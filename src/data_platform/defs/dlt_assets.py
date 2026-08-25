import asyncio
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import dlt
from dagster import AssetExecutionContext, AssetKey, AssetSpec
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

# stock_news / services packages live at project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


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
_stock_news = _load_module("data_extraction.stock_news", "stock_news.py")
_news_conform = _load_module("data_extraction.news_conform", "news_conform.py")
listings_source = _listings.listings_source()
screener_source = _screener.screener_source()
stock_news_source = _stock_news.stock_news_source()
news_conform_source = _news_conform.news_conform_source()

switch_context(str(INGEST_TO_LANDING_DIR))
os.environ.pop("PYICEBERG_HOME", None)

sys.path.insert(0, str(INGEST_TO_LANDING_DIR))
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
_news_conform_pipeline = dlt.pipeline(
    pipeline_name="stock_news_conform",
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
    yield from dlt.run(context=context)


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
    yield from dlt.run(context=context)


@dlt_assets(
    dlt_source=news_conform_source,
    dlt_pipeline=_news_conform_pipeline,
    name="news_conform",
    group_name="data_extraction",
    dagster_dlt_translator=SchemaPrefixedDltTranslator(
        STOCK_NEWS_SCHEMA,
        deps=[
            AssetKey([STOCK_NEWS_SCHEMA, "articles"]),
            AssetKey([LISTINGS_SCHEMA, "equity_universe"]),
        ],
    ),
)
def news_conform_assets(context: AssetExecutionContext, dlt: DagsterDltResource):
    yield from dlt.run(context=context)
