"""Screener.in period tables → Iceberg (filesystem + Glue), schema bronze_screener."""

from __future__ import annotations

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
from utils.screener_fetch import TABLE_NAMES, fetch_screener_tables  # noqa: E402

DEFAULT_DATASET = "bronze_screener"
PK = ["symbol", "financial_period"]


def _records(df: pd.DataFrame, *, ingested_at: datetime) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    for row in rows:
        row["ingested_at"] = ingested_at
    return rows


@dlt.source(name="screener")
def screener_source(
    symbols: list[str] | None = None,
    limit: int | None = None,
):
    """One dlt resource per Screener CSV/table name → bronze_screener.<name>.

    Fetch is deferred until resources run (not at source construction / Dagster load).
    ``ingested_at`` is set at yield time and refreshes only on insert/upsert of that row.
    """
    if limit is None:
        raw = os.getenv("SCREENER_LIMIT", "").strip()
        limit = int(raw) if raw else None

    cache: dict[str, Any] = {"tables": None, "ingested_at": None}

    def tables() -> dict[str, pd.DataFrame]:
        if cache["tables"] is None:
            cache["tables"] = fetch_screener_tables(symbols=symbols, limit=limit)
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
    print(
        dlt.pipeline(
            pipeline_name="screener",
            destination=filesystem_destination(),
            dataset_name=DEFAULT_DATASET,
        ).run(screener_source())
    )
