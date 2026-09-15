"""Daily equity prices via KiteConnect → Iceberg ``bronze_equity.price_history``.

Optional technical_analyst forecasts → ``bronze_equity.price_predictions``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import dlt
from dotenv import load_dotenv

_DATA_EXTRACTION_DIR = Path(__file__).resolve().parent
if str(_DATA_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_EXTRACTION_DIR))

load_dotenv(_DATA_EXTRACTION_DIR.parent / ".env")

from equity_pricing import (  # noqa: E402
    DEFAULT_DATASET,
    PK,
    PRICE_COLUMNS,
    TABLE_NAME,
    WRITE,
)
from utils.equity_pricing_kite_fetch import (  # noqa: E402
    fetch_equity_pricing_daily_rows,
)
from utils.equity_pricing_predict import (  # noqa: E402
    PRED_COLUMNS,
    PRED_PK,
    PRED_RESOURCE,
    PRED_TABLE,
    PRED_WRITE,
    build_prediction_rows,
)

RESOURCE_NAME = "equity_pricing_daily"
PRED_ASSET_NAME = PRED_RESOURCE
log = logging.getLogger(__name__)


def _price_resource(records: list[dict[str, Any]]):
    @dlt.resource(
        name=RESOURCE_NAME,
        table_name=TABLE_NAME,
        primary_key=PK,
        write_disposition=WRITE,
        table_format="iceberg",
        columns=PRICE_COLUMNS,
    )
    def _rows():
        yield from records

    return _rows


def _prediction_resource(records: list[dict[str, Any]]):
    @dlt.resource(
        name=PRED_RESOURCE,
        table_name=PRED_TABLE,
        primary_key=PRED_PK,
        write_disposition=PRED_WRITE,
        table_format="iceberg",
        columns=PRED_COLUMNS,
    )
    def _rows():
        yield from records

    return _rows


@dlt.resource(
    name=RESOURCE_NAME,
    table_name=TABLE_NAME,
    primary_key=PK,
    write_disposition=WRITE,
    table_format="iceberg",
    columns=PRICE_COLUMNS,
)
def equity_pricing_daily():
    """Stub resource for Dagster asset specs; real load uses load_equity_pricing_daily."""
    if False:  # pragma: no cover
        yield {}


@dlt.resource(
    name=PRED_RESOURCE,
    table_name=PRED_TABLE,
    primary_key=PRED_PK,
    write_disposition=PRED_WRITE,
    table_format="iceberg",
    columns=PRED_COLUMNS,
)
def equity_pricing_predictions():
    """Stub resource for Dagster asset specs."""
    if False:  # pragma: no cover
        yield {}


@dlt.source(name=RESOURCE_NAME)
def equity_pricing_daily_source():
    return [equity_pricing_daily(), equity_pricing_predictions()]


def load_equity_pricing_daily(
    pipeline: dlt.Pipeline,
    *,
    access_token: str,
    exchange: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
    run_predictions: bool = True,
    calibrate_predictions: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    lg = logger or log
    try:
        pipeline.drop_pending_packages()
    except Exception as exc:
        lg.warning("Could not drop pending dlt packages: %s", exc)

    rows, stats = fetch_equity_pricing_daily_rows(
        access_token,
        exchange=exchange,
        limit=limit,
        symbols=symbols,
    )
    price_info = None
    if rows:
        price_info = pipeline.run(_price_resource(rows))
        lg.info("Loaded %s kite daily price rows (%s)", len(rows), price_info)
    else:
        lg.warning("No kite daily price rows to load")

    pred_rows, pred_stats = build_prediction_rows(
        rows,
        enabled=run_predictions,
        calibrate=calibrate_predictions,
        logger=lg,
    )
    pred_info = None
    if pred_rows:
        pred_info = pipeline.run(_prediction_resource(pred_rows))
        lg.info("Loaded %s prediction rows (%s)", len(pred_rows), pred_info)

    return {
        "rows_loaded": len(rows),
        "prediction_rows_loaded": len(pred_rows),
        "symbols_requested": stats["symbols_requested"],
        "fetch_ok": stats["ok"],
        "fetch_skip": stats["skip"],
        "predicted_ok": pred_stats["predicted_ok"],
        "predicted_error": pred_stats["predicted_error"],
        "interval": "1d",
        "source": "kite",
        "price_load_info": str(price_info) if price_info is not None else None,
        "prediction_load_info": str(pred_info) if pred_info is not None else None,
    }
