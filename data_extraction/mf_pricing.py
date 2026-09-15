"""MF NAV history → Iceberg ``bronze_mf.nav_history``.

Daily (AMFI) and historical (mfapi) both merge-upsert into the same table on
``(scheme_code, nav_date)``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import dlt
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

from utils.dlt_lake_config import filesystem_destination  # noqa: E402
from utils.mf_pricing_fetch import (  # noqa: E402
    iter_mf_pricing_daily,
    iter_mf_pricing_historical,
)

DEFAULT_DATASET = "bronze_mf"
TABLE_NAME = "nav_history"
PK = ["scheme_code", "nav_date"]
WRITE = {"disposition": "merge", "strategy": "upsert"}


@dlt.resource(
    name="mf_pricing_daily",
    table_name=TABLE_NAME,
    primary_key=PK,
    write_disposition=WRITE,
    table_format="iceberg",
)
def mf_pricing_daily():
    """Latest AMFI NAV snapshot → ``nav_history``."""
    yield from iter_mf_pricing_daily()


@dlt.resource(
    name="mf_pricing_historical",
    table_name=TABLE_NAME,
    primary_key=PK,
    write_disposition=WRITE,
    table_format="iceberg",
)
def mf_pricing_historical():
    """mfapi NAV backfill → ``nav_history``."""
    yield from iter_mf_pricing_historical()


@dlt.source(name="mf_pricing_daily")
def mf_pricing_daily_source():
    return mf_pricing_daily()


@dlt.source(name="mf_pricing_historical")
def mf_pricing_historical_source():
    return mf_pricing_historical()


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    mode = os.getenv("MF_PRICING_MODE", "daily").strip().lower()
    pipe = dlt.pipeline(
        pipeline_name="mf_pricing",
        destination=filesystem_destination(),
        dataset_name=DEFAULT_DATASET,
    )
    if mode in {"historical", "hist", "backfill"}:
        print(pipe.run(mf_pricing_historical_source()))
    else:
        print(pipe.run(mf_pricing_daily_source()))
