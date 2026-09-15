"""Configure dlt filesystem + Glue Iceberg from process env / .env.

``secrets.toml`` is gitignored and often missing on EC2; use S3_LAKE_BASE + AWS_*.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import dlt
import pandas as pd
from dlt.common.schema.exceptions import TableNotFound
from dlt.common.schema.utils import is_dlt_table_or_column
from dlt.common.storages.exceptions import SchemaNotFoundError
from dlt.destinations import filesystem

log = logging.getLogger(__name__)
_DLT_INTERNAL_COLS = frozenset({"_dlt_load_id", "_dlt_id"})
# Screener bank / NBFC labels that must not be merged into corporate Iceberg tables.
_SCREENER_BANK_METRICS = frozenset(
    {
        "revenue",
        "financing_profit",
        "financing_margin",
        "gross_npa",
        "net_npa",
    }
)


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


def align_dataframe_to_iceberg_table(
    table_fqn: str,
    df: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Match scrape columns to an existing Iceberg table so merge-upsert can run.

    Screener HTML varies by company; without this, a new metric (e.g. ``roe`` on
    ``ratios``) breaks dlt load with schema mismatch errors.

    When ``metric_columns`` is set, only those metrics are loaded (intersected with
    Glue when the catalog lists extra bank-only fields not on the physical table).
    """
    if df.empty:
        return df
    base_keys = ("symbol", "financial_period")
    if metric_columns is not None:
        # metric_columns is full Iceberg metric layout (incl. bank-only cols as null).
        data_cols = list(base_keys) + list(metric_columns)
        out = df.copy()
        for col in data_cols:
            if col not in out.columns:
                out[col] = pd.NA
        extra = [c for c in out.columns if c not in data_cols]
        if extra:
            log.warning("Align %s: dropping columns outside load set: %s", table_fqn, extra)
            out = out.drop(columns=extra)
        return out[data_cols]

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


def screener_iceberg_table_column_names(table_fqn: str) -> tuple[str, ...] | None:
    """Iceberg user columns through ``ingested_at`` (stops before ``_dlt_*`` / bank tail)."""
    try:
        catalog = load_glue_catalog()
        iceberg_table = catalog.load_table(table_fqn)
        layout: list[str] = []
        for field in iceberg_table.schema().fields:
            name = field.name
            if name in _DLT_INTERNAL_COLS:
                break
            layout.append(name)
        return tuple(layout) if layout else None
    except Exception:
        return None


def screener_load_metric_columns(
    table_fqn: str,
    *,
    canonical: tuple[str, ...],
    table_name: str | None = None,
) -> tuple[str, ...]:
    """Metric columns to populate from scrape (excludes keys and ``ingested_at``)."""
    layout = screener_iceberg_table_column_names(table_fqn)
    if layout:
        return tuple(
            name
            for name in layout
            if name not in {"symbol", "financial_period", "ingested_at"}
        )
    return tuple(m for m in canonical if m not in _SCREENER_BANK_METRICS)


def compact_screener_iceberg_schema(
    table_fqn: str,
    *,
    keep_metrics: Sequence[str],
) -> list[str]:
    """Remove bank-only / stray columns from Glue Iceberg (fixes upsert schema mismatch)."""
    allowed = (
        {"symbol", "financial_period", "ingested_at", "_dlt_load_id", "_dlt_id"}
        | set(keep_metrics)
    )
    try:
        catalog = load_glue_catalog()
        table = catalog.load_table(table_fqn)
    except Exception as exc:
        log.warning("Could not compact Iceberg schema for %s: %s", table_fqn, exc)
        return []

    to_drop = [field.name for field in table.schema().fields if field.name not in allowed]
    if not to_drop:
        return []
    try:
        with table.update_schema() as update:
            for name in to_drop:
                update.delete_column(name)
    except Exception as exc:
        log.warning("Iceberg schema compact failed for %s: %s", table_fqn, exc)
        return []
    log.warning("Dropped orphan Iceberg columns on %s: %s", table_fqn, to_drop)
    return to_drop


def align_dlt_pipeline_table_schema(
    pipeline: dlt.Pipeline,
    table_name: str,
    *,
    column_specs: dict[str, Any],
) -> None:
    """Match local dlt table schema to Iceberg layout (order + null bank columns)."""
    pipeline.activate()
    if not pipeline.default_schema_name:
        return
    try:
        schema = pipeline.default_schema
    except SchemaNotFoundError:
        return
    try:
        table = schema.get_table(table_name)
    except TableNotFound:
        return

    dlt_prefix = schema._dlt_tables_prefix
    allowed = set(column_specs.keys())
    for col_name in list(table["columns"]):
        if col_name in allowed or is_dlt_table_or_column(col_name, dlt_prefix):
            continue
        table["columns"].pop(col_name)

    for col_name, spec in column_specs.items():
        if col_name not in table["columns"]:
            table["columns"][col_name] = spec

    ordered: list[str] = []
    for name in column_specs:
        if name in table["columns"]:
            ordered.append(name)
    for name in table["columns"]:
        if is_dlt_table_or_column(name, dlt_prefix) and name not in ordered:
            ordered.append(name)
    table["columns"] = {name: table["columns"][name] for name in ordered}
    pipeline._schema_storage.save_schema(schema)
