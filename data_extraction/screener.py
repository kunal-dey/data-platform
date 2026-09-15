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

from utils.dlt_iceberg_merge import patch_iceberg_merge_for_screener  # noqa: E402
from utils.dlt_lake_config import (  # noqa: E402
    align_dataframe_to_iceberg_table,
    filesystem_destination,
    align_dlt_pipeline_table_schema,
    compact_screener_iceberg_schema,
    screener_iceberg_table_column_names,
    screener_load_metric_columns,
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
patch_iceberg_merge_for_screener()
BATCH_SIZE = int(os.getenv("SCREENER_BATCH_SIZE", "25"))
SCREENER_SCHEMA_CONTRACT: dict[str, str] = {
    "tables": "evolve",
    "columns": "evolve",
    "data_type": "freeze",
}
log = logging.getLogger(__name__)


def _records(
    df: pd.DataFrame,
    *,
    ingested_at: datetime,
    table_name: str,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    column_order = list(_dlt_columns_for_table(table_name).keys())
    data_cols = [c for c in column_order if c != "ingested_at"]
    slim = df.reindex(columns=data_cols)
    rows = slim.astype(object).where(pd.notna(slim), None).to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for key in column_order:
            if key == "ingested_at":
                record[key] = ingested_at
            else:
                record[key] = row.get(key)
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


def _load_metrics_for_table(table_name: str) -> tuple[str, ...]:
    fqn = f"{DEFAULT_DATASET}.{table_name}"
    canonical = TABLE_METRIC_COLUMNS.get(table_name, ())
    return screener_load_metric_columns(
        fqn, canonical=canonical, table_name=table_name
    )


def _dlt_columns_for_table(table_name: str) -> dict[str, Any]:
    """dlt column schema in Iceberg catalog order (pyiceberg upsert is order-sensitive)."""
    fqn = f"{DEFAULT_DATASET}.{table_name}"
    layout = screener_iceberg_table_column_names(fqn)
    cols: dict[str, Any] = {}
    if layout:
        for name in layout:
            if name in {"symbol", "financial_period"}:
                cols[name] = {"data_type": "text", "nullable": False}
            elif name == "ingested_at":
                cols[name] = {"data_type": "timestamp"}
            else:
                cols[name] = {"data_type": "double"}
        return cols

    metrics = _load_metrics_for_table(table_name)
    cols = {
        "symbol": {"data_type": "text", "nullable": False},
        "financial_period": {"data_type": "text", "nullable": False},
    }
    for metric in metrics:
        cols[metric] = {"data_type": "double"}
    cols["ingested_at"] = {"data_type": "timestamp"}
    return cols


def _prepare_table_frame(table_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    fqn = f"{DEFAULT_DATASET}.{table_name}"
    frame = normalize_screener_frame(table_name, frame)
    metrics = _load_metrics_for_table(table_name)
    return align_dataframe_to_iceberg_table(
        fqn, frame, metric_columns=metrics
    )


def _screener_table_resource(
    table_name: str,
    frame: pd.DataFrame,
    *,
    ingested_at: datetime,
    merge_upsert: bool = False,
):
    columns = _dlt_columns_for_table(table_name)
    write_disposition: Any = (
        {"disposition": "merge", "strategy": "upsert"}
        if merge_upsert
        else "append"
    )

    @dlt.resource(
        name=table_name,
        primary_key=PK if merge_upsert else None,
        write_disposition=write_disposition,
        table_format="iceberg",
        columns=columns,
        schema_contract=SCREENER_SCHEMA_CONTRACT,
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
        fqn = f"{DEFAULT_DATASET}.{table_name}"
        layout = screener_iceberg_table_column_names(fqn)
        if layout:
            keep_metrics = tuple(
                name
                for name in layout
                if name not in {"symbol", "financial_period", "ingested_at"}
            )
        else:
            keep_metrics = tuple(
                m
                for m in TABLE_METRIC_COLUMNS.get(table_name, ())
                if m
                not in {
                    "revenue",
                    "financing_profit",
                    "financing_margin",
                    "gross_npa",
                    "net_npa",
                }
            )
        compact_screener_iceberg_schema(fqn, keep_metrics=keep_metrics)
        column_specs = _dlt_columns_for_table(table_name)
        align_dlt_pipeline_table_schema(
            pipeline,
            table_name,
            column_specs=column_specs,
        )
        try:
            pipeline.drop_pending_packages()
        except Exception as exc:
            lg.warning("Could not drop pending packages before %s: %s", table_name, exc)
        info = pipeline.run(
            _screener_table_resource(
                table_name,
                prepared,
                ingested_at=ingested_at,
                merge_upsert=False,
            ),
            schema_contract=SCREENER_SCHEMA_CONTRACT,
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
    # Destination-stored dlt schema can be wider than Iceberg; do not re-import it each run.
    pipeline.config.restore_from_destination = False
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
            schema_contract=SCREENER_SCHEMA_CONTRACT,
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
