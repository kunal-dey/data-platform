import asyncio
import os
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import dlt
from dagster import AssetExecutionContext
from dagster_dlt import DagsterDltResource, dlt_assets
from dlt.common.runtime.run_context import switch_context

_PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
INGEST_TO_LANDING_DIR = _PROJECT_ROOT / "data_extraction"
DEFAULT_DATASET = "nasdaq_listed"


def _load_nasdaq_listings_pipeline():
    data_extraction_dir = str(INGEST_TO_LANDING_DIR)
    if data_extraction_dir not in sys.path:
        sys.path.insert(0, data_extraction_dir)

    module_name = "data_extraction.nasdaq_listings"
    path = INGEST_TO_LANDING_DIR / "nasdaq_listings.py"
    spec = spec_from_file_location(module_name, path)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_rest_api = _load_nasdaq_listings_pipeline()
nasdaq_listings_source = _rest_api.nasdaq_listings_source()

switch_context(str(INGEST_TO_LANDING_DIR))
os.environ.pop("PYICEBERG_HOME", None)

_nasdaq_pipeline = dlt.pipeline(
    pipeline_name="nasdaq_listings",
    destination="filesystem",
    dataset_name=DEFAULT_DATASET,
)


@dlt_assets(
    dlt_source=nasdaq_listings_source,
    dlt_pipeline=_nasdaq_pipeline,
    name="nasdaq_listings",
    group_name="data_extraction",
)
def nasdaq_listings_assets(context: AssetExecutionContext, dlt: DagsterDltResource):
    yield from dlt.run(context=context)
