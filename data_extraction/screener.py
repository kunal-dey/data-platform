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

from utils.dlt_lake_config import (  # noqa: E402
    align_dataframe_to_iceberg_table,
    filesystem_destination,
    screener_iceberg_metric_columns,
)
from utils.screener_fetch import (  # noqa: E402
    TABLE_METRIC_COLUMNS,
    TABLE_NAMES,
    fetch_screener_tables,
    listing_symbols,
    normalize_screener_frame,
)

DEFAULT_DATASET = "bronze_screener"
PK = ["symbol", "financial_period"]
BATCH_SIZE = int(os.getenv("SCREENER_BATCH_SIZE", "25"))
log = logging.getLogger(__name__)


def _records(
    df: pd.DataFrame,
    *,
    ingested_at: datetime,
    table_name: str,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    allowed = set(_dlt_columns_for_table(table_name).keys())
    cols = [c for c in df.columns if c in allowed]
    slim = df[cols]
    rows = slim.astype(object).where(pd.notna(slim), None).to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for row in rows:
        record = {k: row[k] for k in cols}
        record["ingested_at"] = ingested_at
        out.append(record)
    return out


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


def _dlt_columns_for_table(table_name: str) -> dict[str, Any]:
    """dlt + Iceberg schema: match existing table metrics when present."""
    fqn = f"{DEFAULT_DATASET}.{table_name}"
    fallback = TABLE_METRIC_COLUMNS.get(table_name, ())
    metrics = screener_iceberg_metric_columns(fqn, fallback=fallback)
    cols: dict[str, Any] = {
        "symbol": {"data_type": "text", "nullable": False},
        "financial_period": {"data_type": "text", "nullable": False},
        "ingested_at": {"data_type": "timestamp"},
    }
    for metric in metrics:
        cols[metric] = {"data_type": "double"}
    return cols


def _prepare_table_frame(table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    fqn = f"{DEFAULT_DATASET}.{table_name}"
    frame = normalize_screener_frame(table_name, frame)
    metrics = screener_iceberg_metric_columns(
        fqn, fallback=TABLE_METRIC_COLUMNS.get(table_name, ())
    )
    base = ["symbol", "financial_period"]
    allowed = base + list(metrics)
    extra = [c for c in frame.columns if c not in allowed]
    if extra:
        log.warning("Prepare %s: dropping columns %s", table_name, extra)
        frame = frame.drop(columns=extra)
    for col in metrics:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame[allowed]
    return align_dataframe_to_iceberg_table(fqn, frame)


def _screener_table_resource(
    table_name: str,
    frame: pd.DataFrame,
    *,
    ingested_at: datetime,
):
    columns = _dlt_columns_for_table(table_name)

    @dlt.resource(
        name=table_name,
        primary_key=PK,
        write_disposition={"disposition": "merge", "strategy": "upsert"},
        table_format="iceberg",
        columns=columns,
        schema_contract={"columns": "discard_value"},
    )
    def _table() -> Iterator[dict[str, Any]]:
        yield from _records(frame, ingested_at=ingested_at, table_name=table_name)

    return _table


def _load_screener_tables(
    pipeline: dlt.Pipeline,
    tables: dict[str, pd.DataFrame],
    *,
    ingested_at: datetime,
    logger: logging.Logger | None = None,
) -> list[Any]:
    """Load each Iceberg table in its own dlt run (avoids cross-table schema bleed)."""
    lg = logger or log
    load_infos: list[Any] = []
    for table_name in TABLE_NAMES:
        frame = tables.get(table_name, pd.DataFrame())
        if frame.empty:
            continue
        prepared = _prepare_table_frame(table_name, frame)
        if prepared.empty:
            continue
        info = pipeline.run(
            _screener_table_resource(table_name, prepared, ingested_at=ingested_at),
            schema_contract={"columns": "discard_value"},
        )
        lg.info("Loaded %s: %s rows (%s)", table_name, len(prepared), info)
        load_infos.append(info)
    return load_infos


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
    try:
        pipeline.drop_pending_packages()
        lg.info("Cleared stale dlt pending packages on screener pipeline (if any)")
    except Exception as exc:
        lg.warning("Could not drop pending dlt packages: %s", exc)

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
            load_infos = _load_screener_tables(
                pipeline,
                tables,
                ingested_at=ingested_at,
                logger=lg,
            )
            load_info = load_infos[-1] if load_infos else None
            rows_loaded += batch_rows
            lg.info(
                "Batch %s (%s): loaded %s rows across %s tables",
                batches_done,
                label,
                batch_rows,
                len(load_infos),
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
        columns = _dlt_columns_for_table(table_name)

        @dlt.resource(
            name=table_name,
            primary_key=PK,
            write_disposition={"disposition": "merge", "strategy": "upsert"},
            table_format="iceberg",
            columns=columns,
            schema_contract={"columns": "discard_value"},
        )
        def _table() -> Iterator[dict[str, Any]]:
            frame = _prepare_table_frame(
                table_name, tables().get(table_name, pd.DataFrame())
            )
            yield from _records(
                frame,
                ingested_at=cache["ingested_at"],
                table_name=table_name,
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
