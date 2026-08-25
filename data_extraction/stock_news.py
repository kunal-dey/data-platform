"""Economic Times stock news → Iceberg (filesystem + Glue), schema bronze_economic_times."""

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
from utils.stock_news_fetch import fetch_stock_news_tables  # noqa: E402

DEFAULT_DATASET = "bronze_economic_times"


def _records(df: pd.DataFrame, *, ingested_at: datetime | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    if ingested_at is not None:
        for row in rows:
            row["ingested_at"] = ingested_at
    return rows


@dlt.source(name="stock_news")
def stock_news_source(limit: int | None = None):
    """Companies (replace) + articles (merge upsert on url) → bronze_economic_times.*"""
    if limit is None:
        raw = os.getenv("STOCK_NEWS_LIMIT", "").strip()
        limit = int(raw) if raw else None

    cache: dict[str, Any] = {"tables": None, "ingested_at": None}

    def tables() -> dict[str, pd.DataFrame]:
        if cache["tables"] is None:
            cache["tables"] = fetch_stock_news_tables(limit=limit)
            cache["ingested_at"] = datetime.now(timezone.utc)
        return cache["tables"]

    @dlt.resource(
        name="companies",
        write_disposition="replace",
        table_format="iceberg",
    )
    def companies() -> Iterator[dict[str, Any]]:
        yield from _records(tables()["companies"])

    @dlt.resource(
        name="articles",
        primary_key="url",
        write_disposition={"disposition": "merge", "strategy": "upsert"},
        table_format="iceberg",
        columns={
            "url": {"data_type": "text", "nullable": False},
            "heading": {"data_type": "text"},
            "content": {"data_type": "text"},
            "date": {"data_type": "text"},
            "companies": {"data_type": "json"},
            "ingested_at": {"data_type": "timestamp"},
        },
    )
    def articles() -> Iterator[dict[str, Any]]:
        yield from _records(tables()["articles"], ingested_at=cache["ingested_at"])

    return [companies, articles]


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    print(
        dlt.pipeline(
            pipeline_name="stock_news",
            destination=filesystem_destination(),
            dataset_name=DEFAULT_DATASET,
        ).run(stock_news_source())
    )
