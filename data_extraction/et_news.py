"""Economic Times stock news → Iceberg (filesystem + Glue), schema bronze_economic_times."""

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
from utils.stock_news_fetch import (  # noqa: E402
    BATCH_MODE,
    BATCH_SIZE,
    fetch_articles_for_companies,
    fetch_companies,
    iter_companies_by_letter,
    iter_company_batches,
)

DEFAULT_DATASET = "bronze_economic_times"
log = logging.getLogger(__name__)

_ARTICLES_COLUMNS = {
    "url": {"data_type": "text", "nullable": False},
    "heading": {"data_type": "text"},
    "content": {"data_type": "text"},
    "date": {"data_type": "text"},
    "companies": {"data_type": "json"},
    "ingested_at": {"data_type": "timestamp"},
}


def _records(df: pd.DataFrame, *, ingested_at: datetime | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    if ingested_at is not None:
        for row in rows:
            row["ingested_at"] = ingested_at
    return rows


def _limit_from_env(limit: int | None) -> int | None:
    if limit is not None:
        return limit
    raw = os.getenv("STOCK_NEWS_LIMIT", "").strip()
    return int(raw) if raw else None


def _batched_enabled() -> bool:
    return os.getenv("STOCK_NEWS_BATCHED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _articles_resource(rows: list[dict[str, Any]]):
    """Build a one-shot articles resource for a single batch load."""

    @dlt.resource(
        name="articles",
        primary_key="url",
        write_disposition={"disposition": "merge", "strategy": "upsert"},
        table_format="iceberg",
        columns=_ARTICLES_COLUMNS,
    )
    def articles() -> Iterator[dict[str, Any]]:
        yield from rows

    return articles


def load_articles_batched(
    pipeline: dlt.Pipeline,
    *,
    limit: int | None = None,
    batch_mode: str | None = None,
    batch_size: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Scrape + Iceberg-load one batch at a time so peak RAM stays low.

    ``batch_mode``:
      - ``company`` (default): ``batch_size`` companies → extract → load → free
      - ``letter``: one ET index page of companies → extract → load → free
    """
    lg = logger or log
    limit = _limit_from_env(limit)
    mode = (batch_mode or BATCH_MODE or "company").strip().lower()
    size = max(1, int(batch_size if batch_size is not None else BATCH_SIZE))

    if mode == "letter":
        lg.info("Batched article load: mode=letter limit=%s", limit)
        batch_iter = iter_companies_by_letter(limit=limit)
    else:
        companies_df = fetch_companies(limit=limit)
        lg.info(
            "Batched article load: mode=company batch_size=%s companies=%s",
            size,
            len(companies_df),
        )
        # Keep only lightweight row dicts; build each batch DataFrame on the fly.
        company_rows = companies_df.to_dict(orient="records")
        del companies_df
        gc.collect()

        def _company_batch_iter():
            total = len(company_rows)
            for start in range(0, total, size):
                end = min(start + size, total)
                label = f"companies_{start + 1}-{end}_of_{total}"
                yield label, pd.DataFrame(company_rows[start:end])

        batch_iter = _company_batch_iter()

    batches_done = 0
    articles_loaded = 0
    for label, batch_df in batch_iter:
        batches_done += 1
        n_cos = len(batch_df)
        lg.info("Batch %s (%s): scrape %s companies", batches_done, label, n_cos)
        articles_df = fetch_articles_for_companies(batch_df)
        del batch_df
        n_art = len(articles_df)
        if n_art:
            ingested_at = datetime.now(timezone.utc)
            rows = _records(articles_df, ingested_at=ingested_at)
            del articles_df
            info = pipeline.run(_articles_resource(rows))
            del rows
            articles_loaded += n_art
            lg.info(
                "Batch %s (%s): loaded %s articles (%s)",
                batches_done,
                label,
                n_art,
                info,
            )
        else:
            del articles_df
            lg.info("Batch %s (%s): no articles", batches_done, label)
        gc.collect()

    summary = {
        "batch_mode": mode,
        "batch_size": size if mode == "company" else 0,
        "batches": batches_done,
        "articles_loaded": articles_loaded,
    }
    lg.info("Batched article load done: %s", summary)
    return summary


@dlt.source(name="stock_news")
def stock_news_source(limit: int | None = None):
    """Companies (replace) + articles (merge upsert on url) → bronze_economic_times.*"""
    limit = _limit_from_env(limit)

    cache: dict[str, Any] = {
        "companies": None,
        "articles": None,
        "ingested_at": None,
    }

    def companies_df() -> pd.DataFrame:
        if cache["companies"] is None:
            cache["companies"] = fetch_companies(limit=limit)
        return cache["companies"]

    @dlt.resource(
        name="companies",
        write_disposition="replace",
        table_format="iceberg",
    )
    def companies() -> Iterator[dict[str, Any]]:
        yield from _records(companies_df())

    @dlt.resource(
        name="articles",
        primary_key="url",
        write_disposition={"disposition": "merge", "strategy": "upsert"},
        table_format="iceberg",
        columns=_ARTICLES_COLUMNS,
    )
    def articles() -> Iterator[dict[str, Any]]:
        # Fallback when STOCK_NEWS_BATCHED=0 (loads everything in one extract).
        if cache["articles"] is None:
            cache["articles"] = fetch_articles_for_companies(companies_df())
            cache["ingested_at"] = datetime.now(timezone.utc)
        yield from _records(cache["articles"], ingested_at=cache["ingested_at"])

    return [companies, articles]


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    pipe = dlt.pipeline(
        pipeline_name="stock_news",
        destination=filesystem_destination(),
        dataset_name=DEFAULT_DATASET,
    )
    if _batched_enabled():
        print(pipe.run(stock_news_source().with_resources("companies")))
        print(load_articles_batched(pipe))
    else:
        print(pipe.run(stock_news_source()))
