"""Scrape Economic Times company news → DataFrames for Iceberg load."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urljoin

import aiohttp
import pandas as pd
import requests
from aiohttp.resolver import ThreadedResolver
from requests.adapters import HTTPAdapter
from selectolax.parser import HTMLParser
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

QUOTES = "https://economictimes.indiatimes.com/markets/stocks/stock-quotes"
BASE = "https://economictimes.indiatimes.com"
PAUSE = float(os.getenv("STOCK_NEWS_PAUSE", "1.0"))
CONCURRENCY = int(os.getenv("STOCK_NEWS_CONCURRENCY", "1"))
RETRIES = int(os.getenv("STOCK_NEWS_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("STOCK_NEWS_RETRY_BACKOFF", "5.0"))
TIMEOUT = int(os.getenv("STOCK_NEWS_TIMEOUT", "30"))
# How many company fetches to schedule at once (semaphore still caps in-flight).
EXTRACT_CHUNK = int(os.getenv("STOCK_NEWS_EXTRACT_CHUNK", "25"))
EXISTING_URL_BATCH = int(os.getenv("STOCK_NEWS_EXISTING_URL_BATCH", "100"))
# company = N companies per extract→load cycle; letter = one ET index page per cycle.
BATCH_MODE = os.getenv("STOCK_NEWS_BATCH_MODE", "company").strip().lower()
BATCH_SIZE = int(os.getenv("STOCK_NEWS_BATCH_SIZE", "5"))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": QUOTES,
}
CO_RE = re.compile(r"/stocks/companyid-(\d+)\.cms", re.I)
IDX_RE = re.compile(r"/markets/stocks/stock-quotes/([a-z]|\d|numeric-\d)/?$", re.I)

TABLE_NAMES = ["companies", "articles"]


def _proxy() -> str | None:
    return (
        os.getenv("STOCK_NEWS_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
        or ""
    ).strip() or None


def _session(proxy: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)
            )
        ),
    )
    return s


def _get(s: requests.Session, url: str, *, pause: bool = True) -> str:
    r = s.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    if pause:
        time.sleep(PAUSE)
    return r.text


def _indexes(html: str) -> list[str]:
    urls, seen = [], set()
    for a in HTMLParser(html).css('a[href*="/markets/stocks/stock-quotes/"]'):
        href = urljoin(BASE, a.attributes.get("href") or "").rstrip("/")
        if href not in seen and IDX_RE.search(href):
            seen.add(href)
            urls.append(href)
    return urls or [f"{QUOTES}/{c}" for c in "abcdefghijklmnopqrstuvwxyz123456789"]


def _parse_companies(html: str) -> list[dict[str, str]]:
    tree = HTMLParser(html)
    root = tree.css_first("div.companyLC ul.companyList")
    if not root:
        return []
    out, seen = [], set()
    for a in root.css('a[href*="/stocks/companyid-"]'):
        href = urljoin(BASE, a.attributes.get("href") or "")
        m = CO_RE.search(href)
        if not m or href in seen:
            continue
        seen.add(href)
        cid = m.group(1)
        name = re.sub(r"\s+Share Price$", "", a.text(strip=True), flags=re.I)
        out.append(
            {
                "company_id": cid,
                "company_name": name,
                "news_url": f"{BASE}/stocksupdate_news/companyid-{cid}.cms",
                "stocks_url": href,
            }
        )
    return out


def fetch_companies(*, limit: int | None = None, proxy: str | None = None) -> pd.DataFrame:
    proxy = proxy if proxy is not None else _proxy()
    s = _session(proxy)
    indexes = _indexes(_get(s, QUOTES, pause=False))
    log.info("ET company indexes: %s", len(indexes))
    found: dict[str, dict[str, str]] = {}
    for url in indexes:
        page = _parse_companies(_get(s, url))
        for c in page:
            found.setdefault(c["company_id"], c)
        log.info("%s: %s companies", url.rsplit("/", 1)[-1], len(page))
    rows = list(found.values())
    if limit is not None:
        rows = rows[: int(limit)]
    if not rows:
        raise RuntimeError("No ET companies found")
    return pd.DataFrame(rows)


def iter_companies_by_letter(
    *,
    limit: int | None = None,
    proxy: str | None = None,
):
    """Yield ``(index_label, companies_df)`` per ET stock-quotes index page."""
    proxy = proxy if proxy is not None else _proxy()
    s = _session(proxy)
    indexes = _indexes(_get(s, QUOTES, pause=False))
    log.info("ET company indexes: %s", len(indexes))
    remaining = None if limit is None else int(limit)
    seen: set[str] = set()
    for url in indexes:
        if remaining is not None and remaining <= 0:
            break
        page = _parse_companies(_get(s, url))
        rows = []
        for c in page:
            cid = c["company_id"]
            if cid in seen:
                continue
            seen.add(cid)
            rows.append(c)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
        label = url.rsplit("/", 1)[-1]
        log.info("%s: %s companies (batch)", label, len(rows))
        if rows:
            yield label, pd.DataFrame(rows)


def iter_company_batches(
    companies_df: pd.DataFrame,
    *,
    batch_size: int | None = None,
):
    """Yield ``(label, batch_df)`` slices of a companies DataFrame."""
    size = max(1, int(batch_size if batch_size is not None else BATCH_SIZE))
    rows = companies_df.to_dict(orient="records")
    total = len(rows)
    for start in range(0, total, size):
        end = min(start + size, total)
        label = f"companies_{start + 1}-{end}_of_{total}"
        yield label, pd.DataFrame(rows[start:end])


def _parse_news(html: str, company: dict[str, str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for story in HTMLParser(html).css("div.eachStory"):
        h = story.css_first("h3")
        heading = h.text(strip=True) if h else ""
        if not heading:
            continue
        cat_el = story.css_first(".secCategory")
        cat = (cat_el.text(strip=True) if cat_el else "").replace("|", " ").strip()
        if cat and not re.fullmatch(r"[\s|]*news[\s|]*", cat, re.I):
            continue
        link = (
            story.css_first("h3 a")
            or story.css_first('a[href*="articleshow"]')
            or story.css_first("a[href]")
        )
        href = link.attributes.get("href") if link else ""
        url = urljoin(BASE, href) if href else company["news_url"]
        if "bseindia.com" in url and "articleshow" not in url:
            continue
        p = story.css_first("p")
        t = story.css_first("time") or story.css_first(".storyDate")
        items.append(
            {
                "heading": heading,
                "content": p.text(strip=True) if p else "",
                "url": url,
                "date": t.text(strip=True) if t else "",
                "company": company["company_name"],
            }
        )
    return items


async def _fetch_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    company: dict[str, str],
    *,
    index: int,
    total: int,
    proxy: str | None,
) -> list[dict[str, str]]:
    async with sem:
        last_err: Exception | None = None
        for attempt in range(1, RETRIES + 1):
            await asyncio.sleep(PAUSE if attempt == 1 else RETRY_BACKOFF * attempt)
            try:
                async with session.get(company["news_url"], proxy=proxy) as resp:
                    if resp.status in (403, 429):
                        body = (await resp.text())[:120]
                        raise aiohttp.ClientResponseError(
                            resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=f"{resp.reason}; body={body!r}",
                            headers=resp.headers,
                        )
                    resp.raise_for_status()
                    html = await resp.text()
                items = _parse_news(html, company)
                log.info(
                    "[%s/%s] %s: %s stories",
                    index,
                    total,
                    company["company_name"],
                    len(items),
                )
                return items
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < RETRIES:
                    log.warning(
                        "[%s/%s] retry %s/%s %s: %s",
                        index,
                        total,
                        attempt,
                        RETRIES,
                        company["company_name"],
                        e,
                    )
        log.warning(
            "[%s/%s] skip %s: %s",
            index,
            total,
            company["company_name"],
            last_err,
        )
        return []


async def _extract_async(
    companies: list[dict[str, str]],
    *,
    concurrency: int,
    proxy: str | None,
) -> list[dict[str, str]]:
    """Scrape company news pages in bounded chunks (avoids scheduling thousands of tasks)."""
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        family=socket.AF_INET,
        resolver=ThreadedResolver(),
    )
    total = len(companies)
    chunk = max(concurrency, EXTRACT_CHUNK)
    out: list[dict[str, str]] = []
    async with aiohttp.ClientSession(
        headers=HEADERS, timeout=timeout, connector=connector
    ) as session:
        for start in range(0, total, chunk):
            batch = companies[start : start + chunk]
            results = await asyncio.gather(
                *[
                    _fetch_one(
                        session,
                        sem,
                        c,
                        index=start + i,
                        total=total,
                        proxy=proxy,
                    )
                    for i, c in enumerate(batch, 1)
                ]
            )
            for items in results:
                out.extend(items)
    return out


def _collapse_by_url(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    batch: dict[str, dict[str, Any]] = {}
    for item in items:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        row = batch.setdefault(
            url,
            {
                "url": url,
                "heading": item.get("heading") or "",
                "content": item.get("content") or "",
                "date": item.get("date") or "",
                "companies": set(),
            },
        )
        company = (item.get("company") or "").strip()
        if company:
            row["companies"].add(company)
        if item.get("heading"):
            row["heading"] = item["heading"]
        if item.get("content"):
            row["content"] = item["content"]
        if item.get("date"):
            row["date"] = item["date"]

    existing = _existing_article_companies(set(batch))
    merged: list[dict[str, Any]] = []
    for url, row in batch.items():
        companies = set(row["companies"]) | set(existing.get(url, []))
        merged.append(
            {
                "url": url,
                "heading": row["heading"],
                "content": row["content"],
                "date": row["date"],
                "companies": sorted(companies),
            }
        )
    return merged


def _existing_article_companies(urls: set[str]) -> dict[str, list[str]]:
    """Union companies already in bronze_economic_times.articles (Iceberg).

    Batched row filters only — never load the full table (OOM on small hosts).
    """
    if not urls:
        return {}
    if os.getenv("STOCK_NEWS_SKIP_EXISTING_MERGE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return {}
    try:
        from pyiceberg.expressions import In

        from utils.dlt_lake_config import load_glue_catalog
    except Exception as e:
        log.debug("Skip Iceberg article lookup: %s", e)
        return {}

    try:
        catalog = load_glue_catalog()
        table = catalog.load_table("bronze_economic_times.articles")
    except Exception as e:
        log.info("No existing articles table (or read failed): %s", e)
        return {}

    out: dict[str, list[str]] = {}
    url_list = [u for u in urls if u]
    batch_size = max(1, EXISTING_URL_BATCH)
    for start in range(0, len(url_list), batch_size):
        batch = url_list[start : start + batch_size]
        try:
            arrow = table.scan(
                row_filter=In("url", batch),
                selected_fields=("url", "companies"),
            ).to_arrow()
        except Exception as e:
            log.warning("Existing articles batch lookup failed; skipping merge: %s", e)
            return out
        url_col = arrow.column("url").to_pylist()
        co_col = arrow.column("companies").to_pylist()
        for url, companies in zip(url_col, co_col):
            if not url:
                continue
            names: list[str] = []
            if isinstance(companies, list):
                names = [str(x).strip() for x in companies if str(x).strip()]
            elif isinstance(companies, str) and companies.strip():
                names = [companies.strip()]
            out[str(url).strip()] = names
    return out


def _run_coro(coro):
    """Run a coroutine; safe if a loop is already running (some Dagster executors)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def fetch_articles_for_companies(
    companies_df: pd.DataFrame,
    *,
    concurrency: int | None = None,
    proxy: str | None = None,
) -> pd.DataFrame:
    """Scrape news for the given companies DataFrame → articles DataFrame."""
    proxy = proxy if proxy is not None else _proxy()
    workers = max(1, int(concurrency if concurrency is not None else CONCURRENCY))
    company_rows = companies_df.to_dict(orient="records")
    log.info(
        "Scraping ET news: companies=%s concurrency=%s pause=%ss chunk=%s",
        len(company_rows),
        workers,
        PAUSE,
        EXTRACT_CHUNK,
    )
    raw_items = _run_coro(
        _extract_async(company_rows, concurrency=workers, proxy=proxy)
    )
    articles = _collapse_by_url(raw_items)
    log.info("ET news done: articles=%s", len(articles))
    if not articles:
        return pd.DataFrame(
            columns=["url", "heading", "content", "date", "companies"]
        )
    return pd.DataFrame(articles)


def fetch_stock_news_tables(
    *,
    limit: int | None = None,
    concurrency: int | None = None,
    proxy: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ``{companies, articles}`` DataFrames for dlt / CLI."""
    companies_df = fetch_companies(limit=limit, proxy=proxy)
    articles_df = fetch_articles_for_companies(
        companies_df, concurrency=concurrency, proxy=proxy
    )
    return {"companies": companies_df, "articles": articles_df}