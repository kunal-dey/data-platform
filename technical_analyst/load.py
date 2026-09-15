"""Simple ETL: CSV of NSE symbols → hourly prices merge-upsert to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .etl import (
    DEFAULT_SYMBOLS_CSV,
    HOURLY_PRICES_CSV,
    iter_hourly_price_rows,
    read_symbols_csv,
    upsert_csv,
    yfinance_symbol,
)


def run(
    csv_path: str | Path | None = None,
    symbols: list[str] | None = None,
    *,
    out_csv: str | Path | None = None,
) -> Path:
    csv_path = Path(csv_path) if csv_path else DEFAULT_SYMBOLS_CSV
    tickers = [yfinance_symbol(s) for s in (symbols or read_symbols_csv(csv_path))]
    dest = Path(out_csv) if out_csv else Path(config.OUTPUT_DIR) / HOURLY_PRICES_CSV.name
    rows = list(iter_hourly_price_rows(tickers))
    path = upsert_csv(dest, rows)
    print(f"Wrote {len(rows)} rows to {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL: merge-upsert hourly NSE prices and next-hour direction to CSV."
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_SYMBOLS_CSV),
        help="CSV with a symbol column (default: data/nse_symbols.csv)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(config.OUTPUT_DIR) / "hourly_prices.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    print(f"Symbols from {args.csv}")
    run(csv_path=args.csv, out_csv=args.out)


if __name__ == "__main__":
    main()
