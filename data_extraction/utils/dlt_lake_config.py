"""Configure dlt filesystem + Glue Iceberg from process env / .env.

``secrets.toml`` is gitignored and often missing on EC2; use S3_LAKE_BASE + AWS_*.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import dlt
import pandas as pd
from dlt.destinations import filesystem

log = logging.getLogger(__name__)
_DLT_INTERNAL_COLS = frozenset({"_dlt_load_id", "_dlt_id"})


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing {name}. Set it in .env (or the process environment), "
            "or provide data_extraction/.dlt/secrets.toml."
        )
    return value


def apply_dlt_lake_config() -> dict[str, Any]:
    """Inject Iceberg Glue catalog secrets and return filesystem destination kwargs."""
    bucket_url = os.getenv("DESTINATION__FILESYSTEM__BUCKET_URL") or _require("S3_LAKE_BASE")
    access_key = _require("AWS_ACCESS_KEY_ID")
    secret_key = _require("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "ap-south-1").strip() or "ap-south-1"
    catalog_name = os.getenv("ICEBERG_CATALOG_NAME", "data_platform_catalog").strip()

    catalog_config = {
        "type": "glue",
        "uri": "glue",
        "warehouse": bucket_url,
        "glue.region": region,
        "client.access-key-id": access_key,
        "client.secret-access-key": secret_key,
        "client.region": region,
    }

    dlt.secrets["iceberg_catalog.iceberg_catalog_name"] = catalog_name
    dlt.secrets["iceberg_catalog.iceberg_catalog_type"] = "sql"
    dlt.secrets["iceberg_catalog.iceberg_catalog_config"] = catalog_config

    return {
        "bucket_url": bucket_url,
        "credentials": {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region,
        },
        "_catalog_name": catalog_name,
        "_catalog_config": catalog_config,
    }


def filesystem_destination():
    cfg = apply_dlt_lake_config()
    cfg.pop("_catalog_name", None)
    cfg.pop("_catalog_config", None)
    return filesystem(**cfg)


def load_glue_catalog():
    """PyIceberg Glue catalog using the same env as dlt destination."""
    from pyiceberg.catalog import load_catalog

    cfg = apply_dlt_lake_config()
    return load_catalog(cfg["_catalog_name"], **cfg["_catalog_config"])


def load_equity_universe_symbols(*, exchange: str = "NSE", limit: int | None = None) -> list[str]:
    """Distinct symbols from bronze_listings.equity_universe (Iceberg)."""
    from pyiceberg.expressions import EqualTo

    catalog = load_glue_catalog()
    table = catalog.load_table("bronze_listings.equity_universe")
    scan_kwargs: dict[str, Any] = {"selected_fields": ("symbol",)}
    if exchange:
        scan_kwargs["row_filter"] = EqualTo("exchange", exchange.strip().upper())
    arrow = table.scan(**scan_kwargs).to_arrow()
    symbols = sorted(
        {
            str(s).strip().upper()
            for s in arrow.column("symbol").to_pylist()
            if s and str(s).strip()
        }
    )
    if limit is not None:
        symbols = symbols[: int(limit)]
    if not symbols:
        raise RuntimeError(
            "No symbols in bronze_listings.equity_universe"
            + (f" for exchange={exchange}" if exchange else "")
            + " — run listings job first."
        )
    return symbols


def align_dataframe_to_iceberg_table(table_fqn: str, df: pd.DataFrame) -> pd.DataFrame:
    """Match scrape columns to an existing Iceberg table so merge-upsert can run.

    Screener HTML varies by company; without this, a new metric (e.g. ``roe`` on
    ``ratios``) breaks dlt load with schema mismatch errors.
    """
    if df.empty:
        return df
    try:
        catalog = load_glue_catalog()
        iceberg_table = catalog.load_table(table_fqn)
        dest_cols = [
            field.name
            for field in iceberg_table.schema().fields
            if field.name not in _DLT_INTERNAL_COLS
        ]
    except Exception:
        return df

    # ``ingested_at`` is set in screener._records(), not in the scrape frame.
    data_cols = [c for c in dest_cols if c != "ingested_at"]
    out = df.copy()
    for col in data_cols:
        if col not in out.columns:
            out[col] = pd.NA
    extra = [c for c in out.columns if c not in data_cols]
    if extra:
        log.warning(
            "Align %s: dropping columns not in Iceberg schema: %s",
            table_fqn,
            extra,
        )
        out = out.drop(columns=extra)
    return out[data_cols]
