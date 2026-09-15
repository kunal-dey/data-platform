"""KiteConnect daily equity quotes for bronze_listings.equity_universe."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect

log = logging.getLogger(__name__)

SOURCE = "kite"
INTERVAL = "1d"
DEFAULT_CHUNK = 300
IST = ZoneInfo("Asia/Kolkata")


def _api_key() -> str:
    key = os.getenv("KITE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing KITE_API_KEY — set it in .env")
    return key


def make_kite(access_token: str, *, api_key: str | None = None) -> KiteConnect:
    token = (access_token or "").strip() or os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing Kite access_token — pass it via Launchpad config "
            "or set KITE_ACCESS_TOKEN"
        )
    kite = KiteConnect(api_key=api_key or _api_key())
    kite.set_access_token(token)
    return kite


def kite_instrument(exchange: str, symbol: str) -> str:
    return f"{exchange.strip().upper()}:{symbol.strip().upper()}"


def trading_day_bar_time(*, now: datetime | None = None) -> str:
    """Stable daily PK timestamp (IST trading date as midnight UTC ISO)."""
    local = (now or datetime.now(tz=IST)).astimezone(IST)
    day = datetime(local.year, local.month, local.day, tzinfo=timezone.utc)
    return day.isoformat()


def _chunks(items: list[str], size: int) -> list[list[str]]:
    size = max(1, size)
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_symbol_from_key(instrument_key: str) -> tuple[str, str]:
    """'NSE:RELIANCE-BE' -> ('NSE', 'RELIANCE')."""
    exchange, _, rest = instrument_key.partition(":")
    symbol = re.sub(r"-BE$", "", rest, flags=re.IGNORECASE)
    return exchange.strip().upper(), symbol.strip().upper()


def fetch_kite_quotes(
    kite: KiteConnect,
    instruments: Sequence[str],
    *,
    chunk_size: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch Kite quote() calls (includes OHLC + last_price + volume)."""
    size = chunk_size
    if size is None:
        raw = os.getenv("EQUITY_PRICING_KITE_CHUNK", "").strip()
        size = int(raw) if raw else DEFAULT_CHUNK
    size = max(1, size)

    out: dict[str, dict[str, Any]] = {}
    for block in _chunks(list(instruments), size):
        try:
            out.update(kite.quote(block))
        except Exception as exc:
            log.warning("Kite quote failed for %s instruments: %s", len(block), exc)
            try:
                ltp = kite.ltp(block)
                for key, payload in ltp.items():
                    out.setdefault(
                        key,
                        {
                            "last_price": payload.get("last_price"),
                            "ohlc": {},
                            "volume": None,
                        },
                    )
            except Exception as ltp_exc:
                log.warning("Kite LTP fallback failed: %s", ltp_exc)
    return out


def quotes_to_records(
    quotes: dict[str, dict[str, Any]],
    *,
    bar_time: str | None = None,
) -> list[dict[str, Any]]:
    bar_time = bar_time or trading_day_bar_time()

    def _f(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for key, payload in quotes.items():
        is_be = 1 if re.search(r"-BE$", key.split(":", 1)[-1], re.I) else 0
        ranked.append((is_be, key, payload))
    ranked.sort(key=lambda x: x[0])

    rows_by_pk: dict[tuple[str, str], dict[str, Any]] = {}
    for _is_be, key, payload in ranked:
        exchange, symbol = _parse_symbol_from_key(key)
        pk = (exchange, symbol)
        if pk in rows_by_pk or not symbol:
            continue
        ohlc = payload.get("ohlc") or {}
        last = payload.get("last_price")
        close = ohlc.get("close")
        if close is None:
            close = last
        volume = payload.get("volume")
        if volume is None:
            volume = payload.get("volume_traded")
        rows_by_pk[pk] = {
            "symbol": symbol,
            "exchange": exchange,
            "interval": INTERVAL,
            "bar_time": bar_time,
            "open": _f(ohlc.get("open")),
            "high": _f(ohlc.get("high")),
            "low": _f(ohlc.get("low")),
            "close": _f(close),
            "volume": _f(volume),
            "source": SOURCE,
        }
    return list(rows_by_pk.values())


def fetch_equity_pricing_daily_rows(
    access_token: str,
    *,
    exchange: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
    include_be: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Snapshot daily prices for equity_universe via Kite quote/LTP."""
    from utils.equity_pricing_fetch import resolve_equity_pricing_entries

    if limit is None:
        limit_raw = os.getenv("EQUITY_PRICING_LIMIT", "").strip()
        limit = int(limit_raw) if limit_raw else None

    entries = resolve_equity_pricing_entries(
        exchange=exchange,
        limit=limit,
        symbols=symbols,
    )
    instruments: list[str] = []
    for ex, sym in entries:
        instruments.append(kite_instrument(ex, sym))
        if include_be:
            instruments.append(kite_instrument(ex, f"{sym}-BE"))

    kite = make_kite(access_token)
    quotes = fetch_kite_quotes(kite, instruments)
    final_rows = quotes_to_records(quotes)

    matched = {r["symbol"] for r in final_rows}
    requested_syms = {s for _, s in entries}
    stats = {
        "symbols_requested": len(entries),
        "ok": len(matched & requested_syms),
        "skip": len(requested_syms - matched),
        "error": 0,
        "rows": len(final_rows),
    }
    log.info(
        "Kite daily pricing: requested=%s rows=%s ok=%s skip=%s",
        stats["symbols_requested"],
        stats["rows"],
        stats["ok"],
        stats["skip"],
    )
    return final_rows, stats
