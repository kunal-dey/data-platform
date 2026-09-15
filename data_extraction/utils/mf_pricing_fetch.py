"""Shared MF NAV helpers (AMFI daily dump + mfapi historical)."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any

import requests
from requests.exceptions import RequestException

log = logging.getLogger(__name__)

AMFI_NAV_ALL_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_BASE = "https://api.mfapi.in"
SOURCE_AMFI = "amfi"
SOURCE_MFAPI = "mfapi"
DEFAULT_MAX_WORKERS = 5
DEFAULT_RETRIES = 5
DEFAULT_RETRY_BACKOFF = 1.5


def _normalize_nav_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = value.strip()
    for fmt in ("%d-%m-%Y", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized NAV date format: {value!r}")


def fetch_amfi_nav_all(*, timeout: int = 60) -> str:
    """Download the raw AMFI NAVAll text dump."""
    response = requests.get(AMFI_NAV_ALL_URL, timeout=timeout)
    response.raise_for_status()
    return response.text


def _parse_amfi_lines(text: str) -> Iterator[dict[str, Any]]:
    """Parse raw NAVAll lines into structured scheme rows."""
    for line in text.splitlines():
        parts = line.strip().split(";")
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        try:
            nav = float(parts[-2])
            nav_date = _normalize_nav_date(parts[-1])
        except ValueError:
            continue

        isin_growth = parts[1].strip() if parts[1].strip() not in {"", "-"} else None
        isin_div = parts[2].strip() if parts[2].strip() not in {"", "-"} else None
        scheme_name = ";".join(parts[3:-2]).strip()

        yield {
            "scheme_code": int(parts[0]),
            "isin_growth": isin_growth,
            "isin_div_reinvestment": isin_div,
            "scheme_name": scheme_name,
            "nav": nav,
            "nav_date": nav_date,
        }


def iter_mf_pricing_daily(
    nav_all_text: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield latest NAV rows from AMFI (one row per scheme_code)."""
    text = nav_all_text if nav_all_text is not None else fetch_amfi_nav_all()
    for row in _parse_amfi_lines(text):
        yield {
            "scheme_code": row["scheme_code"],
            "isin": row["isin_growth"] or row["isin_div_reinvestment"],
            "scheme_name": row["scheme_name"],
            "nav_date": row["nav_date"],
            "nav": row["nav"],
            "source": SOURCE_AMFI,
        }


def build_isin_scheme_map(nav_all_text: str | None = None) -> dict[str, int]:
    """Map ISIN -> AMFI scheme code."""
    text = nav_all_text if nav_all_text is not None else fetch_amfi_nav_all()
    isin_to_code: dict[str, int] = {}
    for row in _parse_amfi_lines(text):
        for isin in (row["isin_growth"], row["isin_div_reinvestment"]):
            if isin:
                isin_to_code[isin] = row["scheme_code"]
    return isin_to_code


def fetch_scheme_nav_history(
    scheme_code: int,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    timeout: int = 60,
    retries: int | None = None,
    retry_backoff: float | None = None,
) -> dict[str, Any]:
    """Fetch one scheme's NAV history from mfapi (retries SSL/connection flakes)."""
    params: dict[str, str] = {}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    if retries is None:
        retries = int(os.getenv("MF_PRICING_RETRIES", str(DEFAULT_RETRIES)))
    if retry_backoff is None:
        retry_backoff = float(
            os.getenv("MF_PRICING_RETRY_BACKOFF", str(DEFAULT_RETRY_BACKOFF))
        )
    retries = max(1, retries)

    url = f"{MFAPI_BASE}/mf/{scheme_code}"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params or None, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            last_exc = exc
            if attempt + 1 >= retries:
                break
            sleep_s = retry_backoff * (2**attempt)
            log.debug(
                "mfapi retry scheme_code=%s attempt=%s/%s sleep=%.1fs: %s",
                scheme_code,
                attempt + 1,
                retries,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def iter_scheme_pricing_historical(
    scheme_code: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield normalized NAV rows for a single AMFI scheme code."""
    payload = fetch_scheme_nav_history(
        scheme_code,
        start_date=start_date,
        end_date=end_date,
    )
    if payload.get("status") != "SUCCESS" or not payload.get("data"):
        return

    meta = payload.get("meta") or {}
    isin = meta.get("isin_growth") or meta.get("isin_div_reinvestment")
    scheme_name = meta.get("scheme_name")

    for row in payload["data"]:
        yield {
            "scheme_code": int(meta.get("scheme_code", scheme_code)),
            "isin": isin,
            "scheme_name": scheme_name,
            "nav_date": _normalize_nav_date(row["date"]),
            "nav": float(row["nav"]),
            "source": SOURCE_MFAPI,
        }


def resolve_scheme_codes(
    scheme_codes: Iterable[int] | None = None,
) -> list[int]:
    """Resolve scheme codes from args, env, or full AMFI dump."""
    if scheme_codes is not None:
        codes = [int(c) for c in scheme_codes]
    else:
        raw = os.getenv("MF_SCHEME_CODES", "").strip()
        if raw:
            codes = [int(x.strip()) for x in raw.split(",") if x.strip()]
        else:
            codes = sorted(set(build_isin_scheme_map().values()))

    codes = list(dict.fromkeys(codes))
    limit_raw = os.getenv("MF_SCHEME_LIMIT", "").strip()
    if limit_raw:
        codes = codes[: int(limit_raw)]
    return codes


def iter_mf_pricing_historical(
    scheme_codes: Iterable[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    max_workers: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield historical NAV rows for many schemes (mfapi backfill)."""
    codes = resolve_scheme_codes(scheme_codes)
    if not codes:
        log.warning("No MF scheme codes to load historically")
        return

    workers = max_workers
    if workers is None:
        workers = int(os.getenv("MF_PRICING_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))
    workers = max(1, workers)

    start_date = start_date or os.getenv("MF_NAV_START_DATE", "").strip() or None
    end_date = end_date or os.getenv("MF_NAV_END_DATE", "").strip() or None

    log.info(
        "MF historical pricing: schemes=%s workers=%s start=%s end=%s",
        len(codes),
        workers,
        start_date,
        end_date,
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                list,
                iter_scheme_pricing_historical(
                    code,
                    start_date=start_date,
                    end_date=end_date,
                ),
            ): code
            for code in codes
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                yield from future.result()
            except Exception as exc:
                log.warning("MF historical fetch failed for scheme_code=%s: %s", code, exc)
