import os
import sys
from typing import Any

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import dlt
from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetObservation,
    AssetSpec,
    MaterializeResult,
)
from dagster_dlt import DagsterDltResource, DagsterDltTranslator, dlt_assets
from dagster_dlt.translator import DltResourceTranslatorData
from dlt.common.runtime.run_context import switch_context
from dotenv import load_dotenv

_PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
INGEST_TO_LANDING_DIR = _PROJECT_ROOT / "data_extraction"
LISTINGS_SCHEMA = "bronze_listings"
SCREENER_SCHEMA = "bronze_screener"
STOCK_NEWS_SCHEMA = "bronze_economic_times"

load_dotenv(_PROJECT_ROOT / ".env")


def _load_module(module_name: str, filename: str):
    data_extraction_dir = str(INGEST_TO_LANDING_DIR)
    if data_extraction_dir not in sys.path:
        sys.path.insert(0, data_extraction_dir)

    path = INGEST_TO_LANDING_DIR / filename
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class SchemaPrefixedDltTranslator(DagsterDltTranslator):
    def __init__(
        self,
        schema: str,
        deps: list[AssetKey] | None = None,
        resource_deps: dict[str, list[AssetKey]] | None = None,
    ):
        super().__init__()
        self._schema = schema
        self._deps = deps or []
        self._resource_deps = resource_deps or {}

    def get_asset_spec(self, data: DltResourceTranslatorData) -> AssetSpec:
        default_spec = super().get_asset_spec(data)
        attrs: dict = {"key": AssetKey([self._schema, data.resource.name])}
        deps = [*self._deps, *self._resource_deps.get(data.resource.name, [])]
        if deps:
            attrs["deps"] = deps
        return default_spec.replace_attributes(**attrs)


_listings = _load_module("data_extraction.listings", "listings.py")
_screener = _load_module("data_extraction.screener", "screener.py")
_stock_news = _load_module("data_extraction.et_news", "et_news.py")
listings_source = _listings.listings_source()
screener_source = _screener.screener_source()
stock_news_source = _stock_news.stock_news_source()

switch_context(str(INGEST_TO_LANDING_DIR))
os.environ.pop("PYICEBERG_HOME", None)

from utils.dlt_lake_config import filesystem_destination  # noqa: E402

_destination = filesystem_destination()

_listings_pipeline = dlt.pipeline(
    pipeline_name="listings",
    destination=_destination,
    dataset_name=LISTINGS_SCHEMA,
)
_screener_pipeline = dlt.pipeline(
    pipeline_name="screener",
    destination=_destination,
    dataset_name=SCREENER_SCHEMA,
)
_stock_news_pipeline = dlt.pipeline(
    pipeline_name="stock_news",
    destination=_destination,
    dataset_name=STOCK_NEWS_SCHEMA,
)


@dlt_assets(
    dlt_source=listings_source,
    dlt_pipeline=_listings_pipeline,
    name="listings",
    group_name="data_extraction",
    dagster_dlt_translator=SchemaPrefixedDltTranslator(LISTINGS_SCHEMA),
)
def listings_assets(context: AssetExecutionContext, dlt: DagsterDltResource):
    yield from dlt.run(context=context)


@dlt_assets(
    dlt_source=screener_source,
    dlt_pipeline=_screener_pipeline,
    name="screener",
    group_name="data_extraction",
    dagster_dlt_translator=SchemaPrefixedDltTranslator(
        SCREENER_SCHEMA,
        deps=[AssetKey([LISTINGS_SCHEMA, "equity_universe"])],
    ),
)
def screener_assets(context: AssetExecutionContext, dlt: DagsterDltResource):
    # Never use dagster_dlt.run here: it triggers one giant in-process scrape and
    # conflicts with Dagster's interrupt handler (ThreadPoolExecutor on old code).
    del dlt
    selected = {key.path[-1] for key in context.selected_asset_keys}
    table_names = set(_screener.TABLE_NAMES)
    if not selected:
        selected = table_names
    progress_key = AssetKey([SCREENER_SCHEMA, sorted(selected)[0]])

    summary: dict[str, Any] | None = None
    for batch in _screener.iter_screener_batches(
        _screener_pipeline,
        logger=context.log,
    ):
        yield AssetObservation(asset_key=progress_key, metadata=batch)
        summary = {
            "batch_size": batch["batch_size"],
            "batches": batch["batch"],
            "symbols": batch["symbols_total"],
            "rows_loaded": batch["rows_loaded_total"],
        }

    if summary is None:
        summary = {
            "batch_size": _screener.BATCH_SIZE,
            "batches": 0,
            "symbols": 0,
            "rows_loaded": 0,
        }
    for name in sorted(selected):
        yield MaterializeResult(
            asset_key=AssetKey([SCREENER_SCHEMA, name]),
            metadata=summary,
        )


@dlt_assets(
    dlt_source=stock_news_source,
    dlt_pipeline=_stock_news_pipeline,
    name="stock_news",
    group_name="data_extraction",
    dagster_dlt_translator=SchemaPrefixedDltTranslator(
        STOCK_NEWS_SCHEMA,
        # Avoid parallel scrapes under multiprocess (OOM / child crash).
        resource_deps={
            "articles": [AssetKey([STOCK_NEWS_SCHEMA, "companies"])],
        },
    ),
)
def stock_news_assets(context: AssetExecutionContext, dlt: DagsterDltResource):
    selected = {key.path[-1] for key in context.selected_asset_keys}
    # One batch extract→load→GC at a time (default). Full in-memory scrape if STOCK_NEWS_BATCHED=0.
    if selected == {"articles"} and _stock_news._batched_enabled():
        summary = _stock_news.load_articles_batched(
            _stock_news_pipeline,
            logger=context.log,
        )
        yield MaterializeResult(
            asset_key=AssetKey([STOCK_NEWS_SCHEMA, "articles"]),
            metadata={
                "batch_mode": summary["batch_mode"],
                "batch_size": summary["batch_size"],
                "batches": summary["batches"],
                "articles_loaded": summary["articles_loaded"],
            },
        )
        return
    yield from dlt.run(context=context)
