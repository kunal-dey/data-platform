"""Yahoo Finance OHLCV for symbols in bronze_listings.equity_universe."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator, Sequence
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException

from utils.dlt_lake_config import load_glue_catalog

log = logging.getLogger(__name__)

SOURCE = "yfinance"
DEFAULT_INTERVAL = "1h"
DEFAULT_PERIOD = "2y"
DEFAULT_BATCH_SIZE = 25
DEFAULT_RETRIES = 4
DEFAULT_RETRY_BACKOFF = 1.5


def yahoo_ticker(symbol: str, exchange: str) -> str:
    sym = symbol.strip().upper()
    exc = exchange.strip().upper()
    if exc == "NSE":
        return f"{sym}.NS"
    if exc == "BSE":
        return f"{sym}.BO"
    raise ValueError(f"Unsupported exchange for Yahoo mapping: {exchange!r}")


def load_equity_universe_entries(
    *,
    exchange: str | None = None,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """(exchange, symbol) pairs from bronze_listings.equity_universe.

    ``exchange``: ``NSE``, ``BSE``, or empty/``BOTH``/``ALL`` for both.
    """
    from pyiceberg.expressions import EqualTo

    catalog = load_glue_catalog()
    table = catalog.load_table("bronze_listings.equity_universe")
    scan_kwargs: dict[str, Any] = {"selected_fields": ("exchange", "symbol")}
    if exchange is None:
        exc_filter = os.getenv("EQUITY_PRICING_EXCHANGE", "NSE").strip()
    else:
        exc_filter = str(exchange).strip()
    exc_upper = exc_filter.upper()
    if exc_upper in {"", "BOTH", "ALL", "*"}:
        exc_filter = ""
    elif exc_filter:
        scan_kwargs["row_filter"] = EqualTo("exchange", exc_upper)
        exc_filter = exc_upper
    arrow = table.scan(**scan_kwargs).to_arrow()
    exchanges = arrow.column("exchange").to_pylist()
    symbols = arrow.column("symbol").to_pylist()
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for exc, sym in zip(exchanges, symbols, strict=False):
        if not exc or not sym:
            continue
        key = (str(exc).strip().upper(), str(sym).strip().upper())
        if key in seen or not key[1]:
            continue
        seen.add(key)
        pairs.append(key)
    pairs.sort(key=lambda x: (x[0], x[1]))
    if limit is not None:
        pairs = pairs[: int(limit)]
    if not pairs:
        raise RuntimeError(
            "No equity universe rows"
            + (f" for exchange={exc_filter}" if exc_filter else "")
            + " — run listings job first."
        )
    return pairs


def parse_symbol_list(
    symbols: Sequence[str] | None,
    *,
    default_exchange: str = "NSE",
) -> list[tuple[str, str]] | None:
    """Parse Launchpad/env symbol list into (exchange, symbol) pairs.

    Accepts ``RELIANCE`` or ``NSE:RELIANCE``. Empty / None → None (use universe).
    """
    if not symbols:
        return None
    default_exc = (default_exchange or "NSE").strip().upper() or "NSE"
    if default_exc in {"BOTH", "ALL", "*"}:
        default_exc = "NSE"
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in symbols:
        part = str(raw).strip()
        if not part:
            continue
        if ":" in part:
            exc, sym = part.split(":", 1)
            key = (exc.strip().upper(), sym.strip().upper())
        else:
            key = (default_exc, part.upper())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out or None


def resolve_equity_pricing_entries(
    entries: Sequence[tuple[str, str]] | None = None,
    *,
    exchange: str | None = None,
    limit: int | None = None,
    symbols: Sequence[str] | None = None,
) -> list[tuple[str, str]]:
    parsed = parse_symbol_list(
        symbols,
        default_exchange=(
            exchange
            if exchange is not None
            else os.getenv("EQUITY_PRICING_EXCHANGE", "NSE")
        ),
    )
    if entries is not None:
        out = [(e.strip().upper(), s.strip().upper()) for e, s in entries]
    elif parsed is not None:
        out = parsed
    else:
        raw = os.getenv("EQUITY_PRICING_SYMBOLS", "").strip()
        if raw:
            out = parse_symbol_list(
                [p.strip() for p in raw.split(",") if p.strip()],
                default_exchange=(
                    exchange
                    if exchange is not None
                    else os.getenv("EQUITY_PRICING_EXCHANGE", "NSE")
                ),
            ) or []
        else:
            limit_env = os.getenv("EQUITY_PRICING_LIMIT", "").strip()
            resolved_limit = limit if limit is not None else (int(limit_env) if limit_env else None)
            out = load_equity_universe_entries(exchange=exchange, limit=resolved_limit)

    limit_raw = os.getenv("EQUITY_PRICING_LIMIT", "").strip()
    if limit is not None:
        out = out[: int(limit)]
    elif (
        limit_raw
        and entries is None
        and parsed is None
        and not os.getenv("EQUITY_PRICING_SYMBOLS", "").strip()
    ):
        pass  # already applied in load_equity_universe_entries
    elif limit_raw and parsed is None:
        out = out[: int(limit_raw)]
    return out


def _bar_time_iso(ts: datetime | pd.Timestamp) -> str:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat()


def fetch_symbol_ohlcv(
    symbol: str,
    exchange: str,
    *,
    interval: str,
    period: str | None,
    start: str | None,
    end: str | None,
    retries: int | None = None,
    retry_backoff: float | None = None,
) -> pd.DataFrame:
    """Download OHLCV for one listing; empty frame if Yahoo has no data."""
    if retries is None:
        retries = int(os.getenv("EQUITY_PRICING_RETRIES", str(DEFAULT_RETRIES)))
    if retry_backoff is None:
        retry_backoff = float(
            os.getenv("EQUITY_PRICING_RETRY_BACKOFF", str(DEFAULT_RETRY_BACKOFF))
        )
    retries = max(1, retries)

    yf_symbol = yahoo_ticker(symbol, exchange)
    kwargs: dict[str, Any] = {
        "interval": interval,
        "auto_adjust": False,
        "actions": False,
    }
    if start or end:
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period or DEFAULT_PERIOD

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(yf_symbol).history(**kwargs)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except (RequestException, OSError, ValueError) as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                break
            sleep_s = retry_backoff * (2**attempt)
            log.debug(
                "yfinance retry %s attempt=%s/%s sleep=%.1fs: %s",
                yf_symbol,
                attempt + 1,
                retries,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()


def frame_to_records(
    symbol: str,
    exchange: str,
    interval: str,
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        try:
            bar_time = _bar_time_iso(ts)
        except (TypeError, ValueError):
            continue
        volume = row.get("Volume")
        out.append(
            {
                "symbol": symbol,
                "exchange": exchange,
                "interval": interval,
                "bar_time": bar_time,
                "open": float(row["Open"]) if pd.notna(row.get("Open")) else None,
                "high": float(row["High"]) if pd.notna(row.get("High")) else None,
                "low": float(row["Low"]) if pd.notna(row.get("Low")) else None,
                "close": float(row["Close"]) if pd.notna(row.get("Close")) else None,
                "volume": float(volume) if pd.notna(volume) else None,
                "source": SOURCE,
            }
        )
    return out


def fetch_entries_ohlcv(
    entries: Sequence[tuple[str, str]],
    *,
    interval: str | None = None,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fetch many (exchange, symbol) pairs; returns rows + ok/skip/error counts."""
    interval = (interval or os.getenv("EQUITY_PRICING_INTERVAL", DEFAULT_INTERVAL)).strip()
    period = period or os.getenv("EQUITY_PRICING_PERIOD", DEFAULT_PERIOD).strip() or None
    start = start or os.getenv("EQUITY_PRICING_START_DATE", "").strip() or None
    end = end or os.getenv("EQUITY_PRICING_END_DATE", "").strip() or None

    rows: list[dict[str, Any]] = []
    stats = {"ok": 0, "skip": 0, "error": 0, "symbols_requested": len(entries)}

    for exchange, symbol in entries:
        try:
            df = fetch_symbol_ohlcv(
                symbol,
                exchange,
                interval=interval,
                period=period,
                start=start,
                end=end,
            )
            recs = frame_to_records(symbol, exchange, interval, df)
            if recs:
                rows.extend(recs)
                stats["ok"] += 1
            else:
                stats["skip"] += 1
        except Exception as exc:
            stats["error"] += 1
            log.warning(
                "Equity pricing fetch failed %s:%s: %s",
                exchange,
                symbol,
                exc,
            )
    return rows, stats


def iter_equity_pricing_historical_rows(
    entries: Sequence[tuple[str, str]] | None = None,
    **kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Yield all OHLCV rows for resolved universe (single extract, no batching)."""
    pairs = resolve_equity_pricing_entries(entries)
    interval = kwargs.get("interval") or os.getenv("EQUITY_PRICING_INTERVAL", DEFAULT_INTERVAL)
    log.info("Equity historical pricing: symbols=%s interval=%s", len(pairs), interval)
    rows, stats = fetch_entries_ohlcv(pairs, **kwargs)
    log.info(
        "Equity historical fetch done: rows=%s ok=%s skip=%s error=%s",
        len(rows),
        stats["ok"],
        stats["skip"],
        stats["error"],
    )
    yield from rows
