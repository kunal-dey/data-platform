"""Fetch Screener.in company pages → period tables by Iceberg table name."""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed

from utils.dlt_lake_config import load_equity_universe_symbols

log = logging.getLogger(__name__)

BASE_URL = "https://www.screener.in"
TIMEOUT = int(os.getenv("SCREENER_TIMEOUT", "30"))
CONCURRENCY = int(os.getenv("SCREENER_CONCURRENCY", "4"))
PAUSE = float(os.getenv("SCREENER_PAUSE", "0.25"))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": f"{BASE_URL}/",
}

# section_id, table_index, table_name
TABLES: list[tuple[str, int, str]] = [
    ("quarters", 0, "quarterly_results"),
    ("profit-loss", 0, "profit_loss"),
    ("balance-sheet", 0, "balance_sheet"),
    ("cash-flow", 0, "cash_flow"),
    ("ratios", 0, "ratios"),
    ("shareholding", 0, "shareholding_quarterly"),
    ("shareholding", 1, "shareholding_yearly"),
]
TABLE_NAMES = [name for _, _, name in TABLES]
_CORE_TABLES = frozenset(
    {"quarterly_results", "profit_loss", "balance_sheet", "cash_flow", "ratios"}
)
_SKIP_METRICS = {"raw pdf"}
_PERIOD_RE = re.compile(
    r"^(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|TTM)$"
)
_METRIC_ALIASES = {
    "sales": "sales",
    "expenses": "expenses",
    "operating profit": "operating_profit",
    "opm %": "opm_pct",
    "other income": "other_income",
    "interest": "interest",
    "depreciation": "depreciation",
    "profit before tax": "profit_before_tax",
    "tax %": "tax_pct",
    "net profit": "net_profit",
    "eps in rs": "eps",
    "dividend payout %": "dividend_payout_pct",
    "equity capital": "equity_capital",
    "reserves": "reserves",
    "borrowings": "borrowings",
    "other liabilities": "other_liabilities",
    "total liabilities": "total_liabilities",
    "fixed assets": "fixed_assets",
    "cwip": "cwip",
    "investments": "investments",
    "other assets": "other_assets",
    "total assets": "total_assets",
    "cash from operating activity": "cash_from_operating",
    "cash from investing activity": "cash_from_investing",
    "cash from financing activity": "cash_from_financing",
    "net cash flow": "net_cash_flow",
    "debtor days": "debtor_days",
    "inventory days": "inventory_days",
    "days payable": "days_payable",
    "cash conversion cycle": "cash_conversion_cycle",
    "working capital days": "working_capital_days",
    "roce %": "roce_pct",
    "promoters": "promoters_pct",
    "fiis": "fiis_pct",
    "diis": "diis_pct",
    "public": "public_pct",
    "no. of shareholders": "shareholders",
}


def _proxy() -> str | None:
    return (
        os.getenv("SCREENER_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
        or ""
    ).strip() or None


def _norm_metric(label: str) -> str:
    return re.sub(r"\s+", " ", label.replace("+", "").strip()).lower()


def _col_name(label: str) -> str:
    key = _norm_metric(label)
    if key in _METRIC_ALIASES:
        return _METRIC_ALIASES[key]
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_") or "metric"


def _parse_num(raw: str) -> float | None:
    text = raw.strip().replace(",", "").replace("%", "")
    if not text or text in {"-", "—", "–"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _section_tables(html: str, section_id: str) -> list:
    from selectolax.parser import HTMLParser

    section = HTMLParser(html).css_first(f"section#{section_id}")
    return [] if section is None else section.css("table")


def _parse_period_table(table) -> tuple[list[str], dict[str, list[str]]]:
    rows = table.css("tr")
    if not rows:
        raise ValueError("empty table")
    header = [c.text(strip=True) for c in rows[0].css("th,td")]
    periods = [h for h in header[1:] if _PERIOD_RE.match(h)]
    if not periods:
        raise ValueError(f"no periods in header: {header[:6]}")
    metrics: dict[str, list[str]] = {}
    for tr in rows[1:]:
        cells = [c.text(strip=True) for c in tr.css("th,td")]
        if not cells:
            continue
        key = _norm_metric(cells[0])
        if not key or key in _SKIP_METRICS:
            continue
        values = cells[1 : 1 + len(periods)]
        if len(values) < len(periods):
            values.extend([""] * (len(periods) - len(values)))
        metrics[key] = values[: len(periods)]
    return periods, metrics


def _period_frame(
    symbol: str, periods: list[str], metrics: dict[str, list[str]]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for i, period in enumerate(periods):
        row: dict[str, object] = {
            "symbol": symbol.strip().upper(),
            "financial_period": period,
        }
        for label, vals in metrics.items():
            row[_col_name(label)] = _parse_num(vals[i]) if i < len(vals) else None
        records.append(row)
    return pd.DataFrame(records)


def _company_url(symbol: str, *, consolidated: bool = True) -> str:
    sym = symbol.strip().upper()
    if consolidated:
        return f"{BASE_URL}/company/{sym}/consolidated/"
    return f"{BASE_URL}/company/{sym}/"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def _fetch_html(url: str, proxy: str | None = None) -> str:
    with httpx.Client(
        headers=HEADERS, proxy=proxy, timeout=TIMEOUT, follow_redirects=True
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _url_available(
    symbol: str, *, consolidated: bool = True, proxy: str | None = None
) -> tuple[bool, str]:
    url = _company_url(symbol, consolidated=consolidated)
    proxy = proxy if proxy is not None else _proxy()
    try:
        with httpx.Client(
            headers=HEADERS, proxy=proxy, timeout=TIMEOUT, follow_redirects=True
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        log.debug("Screener probe failed %s: %s", url, e)
        return False, url
    if resp.status_code != 200:
        return False, url
    text = resp.text.lower()
    if "page not found" in text or "could not find" in text:
        return False, url
    if "/company/" not in str(resp.url).lower():
        return False, url
    return True, url


def _listing_symbols(*, limit: int | None = None) -> list[str]:
    """NSE symbols from bronze_listings.equity_universe (not live nselib)."""
    exchange = os.getenv("SCREENER_EXCHANGE", "NSE").strip().upper() or "NSE"
    return load_equity_universe_symbols(exchange=exchange, limit=limit)


def _parse_company_tables(symbol: str, html: str) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for section_id, table_idx, key in TABLES:
        tables = _section_tables(html, section_id)
        if table_idx >= len(tables):
            continue
        try:
            periods, metrics = _parse_period_table(tables[table_idx])
            results[key] = _period_frame(symbol, periods, metrics)
        except ValueError as e:
            log.warning("Skip %s: %s", key, e)
    return results


def _extract_symbol(
    symbol: str,
    *,
    consolidated: bool = True,
    proxy: str | None = None,
) -> dict[str, pd.DataFrame]:
    sym = symbol.strip().upper()
    proxy = proxy if proxy is not None else _proxy()

    def _run(use_consolidated: bool) -> dict[str, pd.DataFrame]:
        ok, url = _url_available(sym, consolidated=use_consolidated, proxy=proxy)
        if not ok:
            raise ValueError(f"Screener page not available for {sym}: {url}")
        log.info("Fetching %s", url)
        return _parse_company_tables(sym, _fetch_html(url, proxy=proxy))

    results = _run(consolidated)
    if consolidated and not (_CORE_TABLES & results.keys()):
        log.info("%s: consolidated empty — retrying standalone", sym)
        results = _run(False)
    if not results:
        raise ValueError(f"No Screener tables extracted for {sym}")
    return results


def _extract_one(
    symbol: str,
    *,
    index: int,
    total: int,
    consolidated: bool,
    proxy: str | None,
    pause_seconds: float,
) -> tuple[str, dict[str, pd.DataFrame] | None, str]:
    if pause_seconds > 0:
        time.sleep(pause_seconds)
    ok, url = _url_available(symbol, consolidated=consolidated, proxy=proxy)
    if not ok:
        log.info("[%s/%s] skip %s %s", index, total, symbol, url)
        return symbol, None, "skip"
    try:
        log.info("[%s/%s] extract %s", index, total, symbol)
        return symbol, _extract_symbol(symbol, consolidated=consolidated, proxy=proxy), "ok"
    except Exception as e:
        log.warning("[%s/%s] error %s: %s", index, total, symbol, e)
        return symbol, None, f"error: {e}"


def fetch_screener_tables(
    symbols: Iterable[str] | None = None,
    *,
    limit: int | None = None,
    consolidated: bool = True,
    concurrency: int | None = None,
    pause_seconds: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Scrape Screener; return {table_name: DataFrame}."""
    proxy = _proxy()
    syms = list(symbols) if symbols is not None else _listing_symbols(limit=limit)
    if limit is not None and symbols is not None:
        syms = syms[: int(limit)]
    workers = max(1, int(concurrency if concurrency is not None else CONCURRENCY))
    pause = PAUSE if pause_seconds is None else float(pause_seconds)
    total = len(syms)
    log.info(
        "Screener fetch: symbols=%s concurrency=%s pause=%ss",
        total,
        workers,
        pause,
    )

    buckets: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLE_NAMES}
    ok_n = skip_n = err_n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _extract_one,
                sym,
                index=i,
                total=total,
                consolidated=consolidated,
                proxy=proxy,
                pause_seconds=pause,
            )
            for i, sym in enumerate(syms, start=1)
        ]
        for fut in as_completed(futures):
            _sym, tables, status = fut.result()
            if status == "ok" and tables:
                ok_n += 1
                for name, df in tables.items():
                    if not df.empty:
                        buckets[name].append(df)
            elif status == "skip":
                skip_n += 1
            else:
                err_n += 1

    log.info("Screener fetch done: ok=%s skip=%s error=%s", ok_n, skip_n, err_n)
    return {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in buckets.items()
    }
