"""Optional dlt source wrapping the hourly NSE CSV ETL."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import dlt

from .etl import iter_hourly_price_rows, read_symbols_csv, yfinance_symbol


@dlt.source(name="hourly_prices")
def hourly_price_source(
    symbols: Sequence[str] | None = None,
    csv_path: str | Path | None = None,
    *,
    table_format: str | None = None,
) -> list:
    """Extract hourly OHLCV for CSV symbols and load with merge on (symbol, timestamp)."""
    tickers = [yfinance_symbol(s) for s in symbols] if symbols else read_symbols_csv(csv_path)
    resource_kw: dict[str, Any] = {"write_disposition": "merge"}
    if table_format:
        resource_kw["table_format"] = table_format

    @dlt.resource(name="hourly_prices", primary_key=["symbol", "timestamp"], **resource_kw)
    def hourly_prices() -> Iterator[dict[str, Any]]:
        yield from iter_hourly_price_rows(tickers)

    return [hourly_prices]
