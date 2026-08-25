"""NSE + BSE equity listings → Iceberg (filesystem + Glue)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import dlt
import pandas as pd
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

from utils.listings_fetch import fetch_universe  # noqa: E402

DEFAULT_DATASET = "bronze_listings"
TABLE_NAME = "equity_universe"


@dlt.resource(name=TABLE_NAME, write_disposition="replace", table_format="iceberg")
def equity_universe():
    df = fetch_universe()
    if df.empty:
        raise ValueError("empty equity universe — refusing load")
    yield from df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


@dlt.source(name="listings")
def listings_source():
    return equity_universe()


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    print(
        dlt.pipeline(
            pipeline_name="listings",
            destination="filesystem",
            dataset_name=DEFAULT_DATASET,
        ).run(listings_source())
    )
