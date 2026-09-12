"""Screener.in period tables → Iceberg (filesystem + Glue), schema bronze_screener."""

from __future__ import annotations

import gc
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import dlt
import pandas as pd
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

from utils.dlt_lake_config import filesystem_destination  # noqa: E402
from utils.screener_fetch import (  # noqa: E402
    TABLE_NAMES,
    fetch_screener_tables,
    listing_symbols,
)

DEFAULT_DATASET = "bronze_screener"
PK = ["symbol", "financial_period"]
BATCH_SIZE = int(os.getenv("SCREENER_BATCH_SIZE", "25"))
log = logging.getLogger(__name__)


def _records(df: pd.DataFrame, *, ingested_at: datetime) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    for row in rows:
        row["ingested_at"] = ingested_at
    return rows


def _limit_from_env(limit: int | None) -> int | None:
    if limit is not None:
        return limit
    raw = os.getenv("SCREENER_LIMIT", "").strip()
    return int(raw) if raw else None


def _batched_enabled() -> bool:
    return os.getenv("SCREENER_BATCHED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _screener_batch_source(
    tables: dict[str, pd.DataFrame], *, ingested_at: datetime
):
    def make_resource(table_name: str):
        @dlt.resource(
            name=table_name,
            primary_key=PK,
            write_disposition={"disposition": "merge", "strategy": "upsert"},
            table_format="iceberg",
        )
        def _table() -> Iterator[dict[str, Any]]:
            yield from _records(
                tables.get(table_name, pd.DataFrame()),
                ingested_at=ingested_at,
            )

        return _table

    @dlt.source(name="screener")
    def _source():
        return [make_resource(name) for name in TABLE_NAMES]

    return _source()


def iter_screener_batches(
    pipeline: dlt.Pipeline,
    *,
    limit: int | None = None,
    batch_size: int | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield per-batch progress dicts; merge-upsert each chunk to Iceberg."""
    lg = logger or log
    limit = _limit_from_env(limit)
    size = max(1, int(batch_size if batch_size is not None else BATCH_SIZE))
    symbols = listing_symbols(limit=limit)
    total = len(symbols)
    lg.info("Batched screener load: symbols=%s batch_size=%s", total, size)

    batches_done = 0
    rows_loaded = 0
    for start in range(0, total, size):
        ingested_at = datetime.now(timezone.utc)
        end = min(start + size, total)
        batch_syms = symbols[start:end]
        label = f"symbols_{start + 1}-{end}_of_{total}"
        batches_done += 1
        lg.info("Batch %s (%s): fetch %s symbols", batches_done, label, len(batch_syms))
        fetched = fetch_screener_tables(symbols=batch_syms)
        tables = fetched.tables
        stats = fetched.stats
        batch_rows = sum(len(df) for df in tables.values())
        lg.info(
            "Batch %s fetch stats: ok=%s skip=%s error=%s (of %s)",
            batches_done,
            stats.ok,
            stats.skip,
            stats.error,
            stats.symbols_requested,
        )
        load_info = None
        if batch_rows:
            load_info = pipeline.run(
                _screener_batch_source(tables, ingested_at=ingested_at)
            )
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
        del tables
        gc.collect()
        yield {
            "batch": batches_done,
            "label": label,
            "symbols_in_batch": len(batch_syms),
            "batch_rows": batch_rows,
            "rows_loaded_total": rows_loaded,
            "batches_total": (total + size - 1) // size if total else 0,
            "symbols_total": total,
            "batch_size": size,
            "fetch_ok": stats.ok,
            "fetch_skip": stats.skip,
            "fetch_error": stats.error,
            "load_info": str(load_info) if load_info is not None else None,
        }

    lg.info(
        "Batched screener load done: batches=%s symbols=%s rows=%s",
        batches_done,
        total,
        rows_loaded,
    )


def load_screener_batched(
    pipeline: dlt.Pipeline,
    *,
    limit: int | None = None,
    batch_size: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """CLI helper: run all batches and return final summary."""
    last: dict[str, Any] | None = None
    for last in iter_screener_batches(
        pipeline, limit=limit, batch_size=batch_size, logger=logger
    ):
        pass
    if last is None:
        return {
            "batch_size": max(1, int(batch_size or BATCH_SIZE)),
            "batches": 0,
            "symbols": 0,
            "rows_loaded": 0,
        }
    return {
        "batch_size": last["batch_size"],
        "batches": last["batch"],
        "symbols": last["symbols_total"],
        "rows_loaded": last["rows_loaded_total"],
    }


@dlt.source(name="screener")
def screener_source(
    symbols: list[str] | None = None,
    limit: int | None = None,
):
    """One dlt resource per Screener CSV/table name → bronze_screener.<name>.

    Fetch is deferred until resources run (not at source construction / Dagster load).
    ``ingested_at`` is set at yield time and refreshes only on insert/upsert of that row.
    """
    limit = _limit_from_env(limit)

    cache: dict[str, Any] = {"tables": None, "ingested_at": None}

    def tables() -> dict[str, pd.DataFrame]:
        if cache["tables"] is None:
            cache["tables"] = fetch_screener_tables(
                symbols=symbols, limit=limit
            ).tables
            cache["ingested_at"] = datetime.now(timezone.utc)
        return cache["tables"]

    def make_resource(table_name: str):
        @dlt.resource(
            name=table_name,
            primary_key=PK,
            write_disposition={"disposition": "merge", "strategy": "upsert"},
            table_format="iceberg",
        )
        def _table() -> Iterator[dict[str, Any]]:
            yield from _records(
                tables().get(table_name, pd.DataFrame()),
                ingested_at=cache["ingested_at"],
            )

        return _table

    return [make_resource(name) for name in TABLE_NAMES]


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    pipe = dlt.pipeline(
        pipeline_name="screener",
        destination=filesystem_destination(),
        dataset_name=DEFAULT_DATASET,
    )
    if _batched_enabled():
        print(load_screener_batched(pipe))
    else:
        print(pipe.run(screener_source()))
