"""Kite mutual fund instrument listings → Iceberg (filesystem + Glue).

Loads into ``bronze_listings.mf_listings`` (full replace snapshot).

Uses the public Kite MF instruments CSV (no API token required):
https://api.kite.trade/mf/instruments
"""

from __future__ import annotations

import csv
import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dlt
import pandas as pd
import requests
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

from utils.dlt_lake_config import filesystem_destination  # noqa: E402

DEFAULT_DATASET = "bronze_listings"
TABLE_NAME = "mf_listings"
KITE_MF_INSTRUMENTS_URL = "https://api.kite.trade/mf/instruments"

_BOOL_COLS = ("purchase_allowed", "redemption_allowed")
_FLOAT_COLS = (
    "minimum_purchase_amount",
    "purchase_amount_multiplier",
    "minimum_additional_purchase_amount",
    "minimum_redemption_quantity",
    "redemption_quantity_multiplier",
    "last_price",
)


def fetch_mf_instruments_csv(*, timeout: int = 120) -> str:
    """Download the public Kite MF instruments CSV dump."""
    response = requests.get(KITE_MF_INSTRUMENTS_URL, timeout=timeout)
    response.raise_for_status()
    return response.text


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for col in _BOOL_COLS:
        if col not in out or out[col] in ("", None):
            continue
        out[col] = str(out[col]).strip() in {"1", "true", "True", "yes", "YES"}
    for col in _FLOAT_COLS:
        if col not in out or out[col] in ("", None):
            continue
        try:
            out[col] = float(out[col])
        except (TypeError, ValueError):
            out[col] = None
    return out


def iter_mf_listings(
    csv_text: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield mutual fund instrument records from Kite's public CSV."""
    text = csv_text if csv_text is not None else fetch_mf_instruments_csv()
    reader = csv.DictReader(io.StringIO(text))
    rows = 0
    for row in reader:
        rows += 1
        yield _coerce_row(row)
    if rows == 0:
        raise ValueError("empty MF instrument list from Kite — refusing load")


def get_mf_listings(csv_text: str | None = None) -> list[dict[str, Any]]:
    """Return the full mutual fund instrument list as dicts."""
    return list(iter_mf_listings(csv_text=csv_text))


def get_mf_listings_df(
    csv_text: str | None = None,
    *,
    growth_only: bool = False,
    purchase_allowed_only: bool = False,
) -> pd.DataFrame:
    """Return MF listings as a DataFrame, optionally filtered."""
    df = pd.DataFrame(get_mf_listings(csv_text=csv_text))
    if growth_only:
        df = df.loc[df["dividend_type"] == "growth"]
    if purchase_allowed_only:
        df = df.loc[df["purchase_allowed"] == True]  # noqa: E712
    return df.reset_index(drop=True)


@dlt.resource(
    name=TABLE_NAME,
    write_disposition="replace",
    table_format="iceberg",
    primary_key="tradingsymbol",
)
def mf_listings():
    yield from iter_mf_listings()


@dlt.source(name="mf_listings")
def mf_listings_source():
    return mf_listings()


if __name__ == "__main__":
    os.environ.pop("PYICEBERG_HOME", None)
    print(
        dlt.pipeline(
            pipeline_name="mf_listings",
            destination=filesystem_destination(),
            dataset_name=DEFAULT_DATASET,
        ).run(mf_listings_source())
    )
