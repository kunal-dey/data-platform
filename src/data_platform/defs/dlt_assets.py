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
SCHEMA = "bronze_listings"

load_dotenv(_PROJECT_ROOT / ".env")


def _load_listings_module():
    data_extraction_dir = str(INGEST_TO_LANDING_DIR)
    if data_extraction_dir not in sys.path:
        sys.path.insert(0, data_extraction_dir)

    module_name = "data_extraction.listings"
    path = INGEST_TO_LANDING_DIR / "listings.py"
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ListingsDltTranslator(DagsterDltTranslator):
    def get_asset_spec(self, data: DltResourceTranslatorData) -> AssetSpec:
        default_spec = super().get_asset_spec(data)
        return default_spec.replace_attributes(
            key=AssetKey([SCHEMA, data.resource.name]),
        )


_listings = _load_listings_module()
listings_source = _listings.listings_source()

switch_context(str(INGEST_TO_LANDING_DIR))
os.environ.pop("PYICEBERG_HOME", None)

_listings_pipeline = dlt.pipeline(
    pipeline_name="listings",
    destination="filesystem",
    dataset_name=SCHEMA,
)


@dlt_assets(
    dlt_source=listings_source,
    dlt_pipeline=_listings_pipeline,
    name="listings",
    group_name="data_extraction",
    dagster_dlt_translator=ListingsDltTranslator(),
)
def listings_assets(context: AssetExecutionContext, dlt: DagsterDltResource):
    yield from dlt.run(context=context)
