"""LLM conform ET articles → bronze_economic_times.conform_articles (Iceberg)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

import dlt
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DATA_EXTRACTION_DIR.parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from utils.dlt_lake_config import filesystem_destination  # noqa: E402

DEFAULT_DATASET = "bronze_economic_times"


@dlt.source(name="news_conform")
def news_conform_source(limit: int | None = None):
    """Pending articles only: last 7 days and not already in conform_articles."""
    if limit is None:
        raw = os.getenv("CONFORM_LIMIT", "").strip()
        limit = int(raw) if raw else None

    cache: dict[str, Any] = {"rows": None}

    def rows() -> list[dict[str, Any]]:
        if cache["rows"] is None:
            from stock_news.conform import run_conform_rows

            cache["rows"] = run_conform_rows(limit=limit)
        return cache["rows"]

    @dlt.resource(
        name="conform_articles",
        primary_key="url",
        write_disposition={"disposition": "merge", "strategy": "upsert"},
        table_format="iceberg",
        columns={
            "url": {"data_type": "text", "nullable": False},
            "companies_mentioned": {"data_type": "json"},
            "symbols": {"data_type": "json"},
            "symbol_matches": {"data_type": "json"},
            "source_heading": {"data_type": "text"},
            "source_date": {"data_type": "text"},
            "source_companies": {"data_type": "json"},
            "model_name": {"data_type": "text"},
            "conformed_at": {"data_type": "timestamp"},
        },
    )
    def conform_articles() -> Iterator[dict[str, Any]]:
        yield from rows()

    return conform_articles


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    print(
        dlt.pipeline(
            pipeline_name="stock_news_conform",
            destination=filesystem_destination(),
            dataset_name=DEFAULT_DATASET,
        ).run(news_conform_source())
    )
