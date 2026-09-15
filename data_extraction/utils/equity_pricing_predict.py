"""Build technical_analyst prediction rows (separate from price_history)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PRED_TABLE = "price_predictions"
PRED_RESOURCE = "equity_pricing_predictions"
PRED_PK = ["symbol", "exchange", "bar_time"]
PRED_WRITE = {"disposition": "merge", "strategy": "upsert"}
PRED_COLUMNS: dict[str, Any] = {
    "symbol": {"data_type": "text", "nullable": False},
    "exchange": {"data_type": "text", "nullable": False},
    "bar_time": {"data_type": "text", "nullable": False},
    "signal": {"data_type": "text"},
    "direction": {"data_type": "text"},
    "score": {"data_type": "double"},
    "confidence": {"data_type": "double"},
    "path_return": {"data_type": "double"},
    "gap": {"data_type": "double"},
    "origin_price": {"data_type": "double"},
    "origin_time": {"data_type": "text"},
    "source": {"data_type": "text"},
}
PRED_SOURCE = "technical_analyst"


def _predictions_enabled() -> bool:
    return os.getenv("EQUITY_PRICING_PREDICT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _calibrate_enabled() -> bool:
    return os.getenv("EQUITY_PRICING_PREDICT_CALIBRATE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _ensure_project_root_on_path() -> Path:
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "pyproject.toml").is_file()
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _yahoo_ticker(symbol: str, exchange: str) -> str:
    from utils.equity_pricing_fetch import yahoo_ticker

    return yahoo_ticker(symbol, exchange)


def _predict_one(
    symbol: str,
    exchange: str,
    bar_time: str,
    *,
    market_cache: dict[str, dict],
    calibrate: bool,
) -> dict[str, Any]:
    _ensure_project_root_on_path()
    from technical_analyst.api import predict_direction
    from technical_analyst.data import download_market_returns, download_ohlcv

    ticker = _yahoo_ticker(symbol, exchange)
    ohlcv, _currency = download_ohlcv(ticker)
    if ticker not in market_cache:
        market_cache[ticker] = download_market_returns(ticker, quiet=True)

    result = predict_direction(
        ohlcv["close"],
        timestamps=ohlcv.index,
        volumes=ohlcv["volume"],
        opens=ohlcv["open"],
        market_returns=market_cache[ticker],
        cutoff=None,
        calibrate=calibrate,
        quiet=True,
    )
    origin = result.origin_time
    if origin is not None:
        try:
            origin_s = origin.isoformat()
        except AttributeError:
            origin_s = str(origin)
    else:
        origin_s = None

    direction = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}.get(
        result.signal, None
    )
    return {
        "symbol": symbol,
        "exchange": exchange,
        "bar_time": bar_time,
        "signal": result.signal,
        "direction": direction,
        "score": float(result.score),
        "confidence": float(result.confidence),
        "path_return": float(result.path_return),
        "gap": None if result.gap is None else float(result.gap),
        "origin_price": float(result.origin_price),
        "origin_time": origin_s,
        "source": PRED_SOURCE,
    }


def build_prediction_rows(
    price_rows: list[dict[str, Any]],
    *,
    enabled: bool | None = None,
    calibrate: bool | None = None,
    logger: logging.Logger | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    From daily price rows, build rows for ``bronze_equity.price_predictions``.

    Join key with price_history: ``(symbol, exchange, bar_time)``.
    """
    lg = logger or log
    empty_stats = {"predicted_ok": 0, "predicted_skip": 0, "predicted_error": 0}
    if not price_rows:
        return [], empty_stats

    do_predict = _predictions_enabled() if enabled is None else bool(enabled)
    if not do_predict:
        lg.info("Predictions disabled — skipping price_predictions load")
        return [], {**empty_stats, "predicted_skip": len(price_rows)}

    do_calibrate = _calibrate_enabled() if calibrate is None else bool(calibrate)
    market_cache: dict[str, dict] = {}
    stats = {"predicted_ok": 0, "predicted_skip": 0, "predicted_error": 0}

    # One prediction per symbol/exchange using that row's bar_time.
    by_key: dict[tuple[str, str], str] = {}
    for row in price_rows:
        key = (str(row["exchange"]), str(row["symbol"]))
        by_key.setdefault(key, str(row["bar_time"]))

    keys = sorted(by_key.items(), key=lambda x: x[0])
    lg.info(
        "Building technical_analyst predictions for %s symbols (calibrate=%s)",
        len(keys),
        do_calibrate,
    )

    out: list[dict[str, Any]] = []
    for i, ((exchange, symbol), bar_time) in enumerate(keys, start=1):
        try:
            out.append(
                _predict_one(
                    symbol,
                    exchange,
                    bar_time,
                    market_cache=market_cache,
                    calibrate=do_calibrate,
                )
            )
            stats["predicted_ok"] += 1
            if i == 1 or i % 25 == 0 or i == len(keys):
                lg.info(
                    "Prediction progress %s/%s (%s:%s)",
                    i,
                    len(keys),
                    exchange,
                    symbol,
                )
        except Exception as exc:
            stats["predicted_error"] += 1
            lg.warning("Prediction failed for %s:%s: %s", exchange, symbol, exc)

    lg.info(
        "Predictions done: ok=%s error=%s rows=%s",
        stats["predicted_ok"],
        stats["predicted_error"],
        len(out),
    )
    return out, stats
