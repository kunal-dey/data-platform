"""Equity OHLCV history (Yahoo Finance) → Iceberg ``bronze_equity.price_history``."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import dlt
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

from utils.dlt_lake_config import filesystem_destination  # noqa: E402
from utils.equity_pricing_fetch import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_INTERVAL,
    fetch_entries_ohlcv,
    iter_equity_pricing_historical_rows,
    resolve_equity_pricing_entries,
)

DEFAULT_DATASET = "bronze_equity"
TABLE_NAME = "price_history"
RESOURCE_NAME = "equity_pricing_historical"
PK = ["symbol", "exchange", "interval", "bar_time"]
WRITE = {"disposition": "merge", "strategy": "upsert"}

PRICE_COLUMNS: dict[str, Any] = {
    "symbol": {"data_type": "text", "nullable": False},
    "exchange": {"data_type": "text", "nullable": False},
    "interval": {"data_type": "text", "nullable": False},
    "bar_time": {"data_type": "text", "nullable": False},
    "open": {"data_type": "double"},
    "high": {"data_type": "double"},
    "low": {"data_type": "double"},
    "close": {"data_type": "double"},
    "volume": {"data_type": "double"},
    "source": {"data_type": "text"},
}

log = logging.getLogger(__name__)


def _batch_size_from_env(batch_size: int | None) -> int:
    if batch_size is not None:
        return max(1, int(batch_size))
    raw = os.getenv("EQUITY_PRICING_BATCH_SIZE", "").strip()
    return max(1, int(raw)) if raw else DEFAULT_BATCH_SIZE


def _batched_enabled() -> bool:
    return os.getenv("EQUITY_PRICING_BATCHED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _price_history_resource(records: list[dict[str, Any]]):
    @dlt.resource(
        name=RESOURCE_NAME,
        table_name=TABLE_NAME,
        primary_key=PK,
        write_disposition=WRITE,
        table_format="iceberg",
        columns=PRICE_COLUMNS,
    )
    def _batch():
        yield from records

    return _batch


@dlt.resource(
    name=RESOURCE_NAME,
    table_name=TABLE_NAME,
    primary_key=PK,
    write_disposition=WRITE,
    table_format="iceberg",
    columns=PRICE_COLUMNS,
)
def equity_pricing_historical():
    """Full-universe Yahoo backfill (use batched path in Dagster for large runs)."""
    yield from iter_equity_pricing_historical_rows()


@dlt.source(name=RESOURCE_NAME)
def equity_pricing_historical_source():
    return equity_pricing_historical()


def iter_equity_pricing_batches(
    pipeline: dlt.Pipeline,
    *,
    batch_size: int | None = None,
    exchange: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield per-batch progress; merge-upsert each chunk to Iceberg."""
    lg = logger or log
    size = _batch_size_from_env(batch_size)
    entries = resolve_equity_pricing_entries(
        exchange=exchange, limit=limit, symbols=symbols
    )
    total = len(entries)
    interval = os.getenv("EQUITY_PRICING_INTERVAL", DEFAULT_INTERVAL).strip()
    lg.info(
        "Batched equity pricing: symbols=%s batch_size=%s interval=%s exchange=%s",
        total,
        size,
        interval,
        exchange or os.getenv("EQUITY_PRICING_EXCHANGE", "NSE"),
    )
    try:
        pipeline.drop_pending_packages()
    except Exception as exc:
        lg.warning("Could not drop pending dlt packages: %s", exc)

    batches_done = 0
    rows_loaded = 0
    fetch_ok = fetch_skip = fetch_error = 0

    for start in range(0, total, size):
        end = min(start + size, total)
        batch_entries = entries[start:end]
        label = f"symbols_{start + 1}-{end}_of_{total}"
        batches_done += 1
        lg.info("Batch %s (%s): fetch %s symbols", batches_done, label, len(batch_entries))
        records, stats = fetch_entries_ohlcv(batch_entries)
        batch_rows = len(records)
        fetch_ok += stats["ok"]
        fetch_skip += stats["skip"]
        fetch_error += stats["error"]
        load_info = None
        if records:
            load_info = pipeline.run(_price_history_resource(records))
            rows_loaded += batch_rows
            lg.info(
                "Batch %s (%s): loaded %s rows (%s)",
                batches_done,
                label,
                batch_rows,
                load_info,
            )
        else:
            lg.info("Batch %s (%s): no rows", batches_done, label)
        del records
        gc.collect()
        yield {
            "batch": batches_done,
            "label": label,
            "symbols_in_batch": len(batch_entries),
            "batch_rows": batch_rows,
            "rows_loaded_total": rows_loaded,
            "batches_total": (total + size - 1) // size if total else 0,
            "symbols_total": total,
            "batch_size": size,
            "interval": interval,
            "exchange": exchange or os.getenv("EQUITY_PRICING_EXCHANGE", "NSE"),
            "fetch_ok": fetch_ok,
            "fetch_skip": fetch_skip,
            "fetch_error": fetch_error,
            "load_info": str(load_info) if load_info is not None else None,
        }

    lg.info(
        "Batched equity pricing done: batches=%s symbols=%s rows=%s",
        batches_done,
        total,
        rows_loaded,
    )


def load_equity_pricing_batched(
    pipeline: dlt.Pipeline,
    *,
    batch_size: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for last in iter_equity_pricing_batches(
        pipeline, batch_size=batch_size, logger=logger
    ):
        pass
    if last is None:
        return {
            "batch_size": _batch_size_from_env(batch_size),
            "batches": 0,
            "symbols": 0,
            "rows_loaded": 0,
        }
    return {
        "batch_size": last["batch_size"],
        "batches": last["batch"],
        "symbols": last["symbols_total"],
        "rows_loaded": last["rows_loaded_total"],
        "interval": last.get("interval"),
    }


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    pipe = dlt.pipeline(
        pipeline_name="equity_pricing_historical",
        destination=filesystem_destination(),
        dataset_name=DEFAULT_DATASET,
    )
    if _batched_enabled():
        print(load_equity_pricing_batched(pipe))
    else:
        print(pipe.run(equity_pricing_historical_source()))
