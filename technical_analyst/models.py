"""Overnight gap (Ridge) + TTM hourly returns, reconstructed from the last known price."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import information_cutoff, localize_cutoff, next_trading_bars

_TTM_CACHE: dict = {}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _tz_cut(index: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if index.tz is not None:
        return ts.tz_localize(index.tz) if ts.tzinfo is None else ts.tz_convert(index.tz)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _asof_return(series: pd.Series | None, cutoff: pd.Timestamp) -> float:
    if series is None or series.empty:
        return 0.0
    hist = series[series.index <= _tz_cut(series.index, cutoff)].dropna()
    if hist.empty:
        return 0.0
    value = float(hist.iloc[-1])
    return value if np.isfinite(value) else 0.0


def _gap_x(row: pd.Series, info_cutoff: pd.Timestamp, market: dict[str, pd.Series]) -> dict[str, float]:
    return {
        "prev_day_return": _finite(row.get("prev_day_return")),
        "ret_5d": _finite(row.get("ret_5d")),
        "ret_20d": _finite(row.get("ret_20d")),
        "vol_5d": _finite(row.get("vol_5d")),
        "vol_20d": _finite(row.get("vol_20d")),
        "volume_change": _finite(row.get("volume_change")),
        "last_gap": _finite(row.get("last_gap")),
        "is_weekend_gap": _finite(row.get("is_weekend_gap")),
        "nifty_day_return": _asof_return(market.get("nifty"), info_cutoff),
        "nifty_it_day_return": _asof_return(market.get("nifty_it"), info_cutoff),
        "usdinr_day_return": _asof_return(market.get("usdinr"), info_cutoff),
        "nasdaq_overnight_return": _asof_return(market.get("nasdaq"), info_cutoff),
        "adr_overnight_return": _asof_return(market.get("adr"), info_cutoff),
    }


def fit_gap_model(features: pd.DataFrame, market: dict[str, pd.Series], origin_ts: pd.Timestamp, asof_mode: str):
    ts = features[config.TIMESTAMP_COLUMN]
    origin_ts = localize_cutoff(origin_ts, pd.DatetimeIndex(ts))
    opens = features[(features["is_session_open"] > 0.5) & (ts < origin_ts)]
    rows, y = [], []
    for loc in opens.index.tolist():
        if loc == 0:
            continue
        open_row, close_row = features.loc[loc], features.loc[loc - 1]
        prev_close, this_open = float(close_row["close"]), float(open_row["open"])
        if prev_close <= 0 or this_open <= 0:
            continue
        target = np.log(this_open / prev_close)
        if not np.isfinite(target):
            continue
        open_ts = pd.Timestamp(open_row[config.TIMESTAMP_COLUMN])
        info_cut = open_ts - pd.Timedelta(minutes=1) if asof_mode == "before_next_open" else pd.Timestamp(
            close_row[config.TIMESTAMP_COLUMN]
        )
        rows.append(_gap_x(close_row, info_cut, market))
        y.append(target)
    if len(y) < 40:
        return None
    model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 16)))])
    model.fit(pd.DataFrame(rows)[config.GAP_FEATURE_COLUMNS], np.asarray(y, dtype=float))
    return model


def predict_gap(model, origin_row, origin_ts, index, market, asof_mode) -> float:
    if model is None:
        return 0.0
    info_cut = information_cutoff(origin_ts, index, asof_mode)
    x = pd.DataFrame([_gap_x(origin_row, info_cut, market)])[config.GAP_FEATURE_COLUMNS]
    pred = float(model.predict(x)[0])
    return pred if np.isfinite(pred) else 0.0


def load_ttm():
    key = (config.CONTEXT_LENGTH, config.PREDICTION_LENGTH)
    if key in _TTM_CACHE:
        return _TTM_CACHE[key]
    from tsfm_public.toolkit.get_model import get_model

    kwargs = dict(
        model_path=config.MODEL_PATH,
        context_length=config.CONTEXT_LENGTH,
        prediction_length=config.PREDICTION_LENGTH,
        prefer_longer_context=False,
    )
    try:
        model = get_model(freq="h", **kwargs)
    except Exception:
        model = get_model(**kwargs)
    _TTM_CACHE[key] = model
    return model


def _ttm_pipeline(cols: list[str]):
    key = ("pipe", config.CONTEXT_LENGTH, config.PREDICTION_LENGTH, tuple(cols), _device())
    if key in _TTM_CACHE:
        return _TTM_CACHE[key]
    from tsfm_public import TimeSeriesForecastingPipeline

    model = load_ttm()
    ctx_len = int(getattr(model.config, "context_length", config.CONTEXT_LENGTH))
    pipe = TimeSeriesForecastingPipeline(
        model=model,
        id_columns=[],
        timestamp_column=config.TIMESTAMP_COLUMN,
        target_columns=cols,
        context_length=ctx_len,
        prediction_length=config.PREDICTION_LENGTH,
        batch_size=config.BATCH_SIZE,
        impute_method=None,
        device=_device(),
        explode_forecasts=False,
        add_known_ground_truth=True,
    )
    _TTM_CACHE[key] = pipe
    return pipe


def _prediction_column(raw: pd.DataFrame) -> str:
    col = f"{config.TARGET_COLUMN}_prediction"
    return col if col in raw.columns else config.TARGET_COLUMN


def _first_step(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return float("nan")
    return float(arr[0])


def ttm_forecast_table(features: pd.DataFrame, *, last_window_only: bool = False) -> pd.DataFrame:
    """TTM path at each origin. Timestamp is the origin bar (last context point)."""
    cols = [c for c in config.FEATURE_COLUMNS if c in features.columns]
    model = load_ttm()
    ctx_len = int(getattr(model.config, "context_length", config.CONTEXT_LENGTH))
    if len(features) < ctx_len:
        raise ValueError(f"Need {ctx_len} feature rows, got {len(features)}.")
    frame = features[[config.TIMESTAMP_COLUMN, *cols]]
    if last_window_only:
        frame = frame.iloc[-ctx_len:]
    raw = _ttm_pipeline(cols)(frame)
    col = _prediction_column(raw)
    out = pd.DataFrame(
        {
            config.TIMESTAMP_COLUMN: pd.to_datetime(raw[config.TIMESTAMP_COLUMN], utc=True),
            "predicted_log_return": raw[col].map(lambda value: np.asarray(value, dtype=float).reshape(-1)),
        }
    )
    out["next_hour_return"] = out["predicted_log_return"].map(_first_step)
    return out


def ttm_returns(features: pd.DataFrame) -> np.ndarray:
    raw = ttm_forecast_table(features, last_window_only=True)
    arr = np.asarray(raw.iloc[-1]["predicted_log_return"], dtype=float).reshape(-1)
    return arr[: config.PREDICTION_LENGTH]


def ttm_first_step_returns(features: pd.DataFrame) -> pd.Series:
    """Next-hour predicted log return at every origin with a full 512-bar context."""
    raw = ttm_forecast_table(features, last_window_only=False)
    return pd.Series(
        raw["next_hour_return"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(raw[config.TIMESTAMP_COLUMN]),
        name="next_hour_return",
    )


def reconstruct_path(
    incremental_returns: np.ndarray,
    origin_price: float,
    origin_ts: pd.Timestamp,
    history_index: pd.DatetimeIndex,
    first_step_gap: float | None = None,
) -> pd.DataFrame:
    r = np.asarray(incremental_returns, dtype=float).copy()
    if first_step_gap is not None and np.isfinite(first_step_gap):
        r[0] = float(first_step_gap)
    origin_rel = np.cumsum(r)
    future = next_trading_bars(history_index, n=len(r))
    if len(future) != len(r):
        future = pd.date_range(pd.Timestamp(origin_ts) + pd.Timedelta(hours=1), periods=len(r), freq="h")
    return pd.DataFrame(
        {
            config.TIMESTAMP_COLUMN: future,
            "predicted_log_return": r,
            "origin_log_return": origin_rel,
            "predicted_price": origin_price * np.exp(origin_rel),
        }
    )


def attach_actuals(path: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    out = path.copy()
    out["actual_price"] = np.nan
    if holdout is None or holdout.empty:
        return out
    n = min(len(out), len(holdout))
    out.loc[out.index[:n], "actual_price"] = holdout["close"].iloc[:n].to_numpy()
    out.loc[out.index[:n], config.TIMESTAMP_COLUMN] = list(holdout.index[:n])
    return out


def apply_calibration(path: pd.DataFrame, origin_price: float, bias, q10, q90) -> pd.DataFrame:
    out = path.copy()
    r = out["origin_log_return"].to_numpy(dtype=float)
    corr = r.copy()
    if bias is not None and len(bias):
        n = min(len(corr), len(bias))
        corr[:n] = r[:n] - bias[:n]
    out["origin_log_return"] = corr
    out["predicted_price"] = origin_price * np.exp(corr)
    if q10 is not None and q90 is not None:
        n = min(len(corr), len(q10), len(q90))
        out["price_q0.1"] = np.nan
        out["price_q0.9"] = np.nan
        out.loc[out.index[:n], "price_q0.1"] = origin_price * np.exp(corr[:n] + q10[:n])
        out.loc[out.index[:n], "price_q0.9"] = origin_price * np.exp(corr[:n] + q90[:n])
    return out


def prepend_origin(path: pd.DataFrame, origin_ts: pd.Timestamp, origin_price: float) -> pd.DataFrame:
    row = {col: np.nan for col in path.columns}
    row[config.TIMESTAMP_COLUMN] = pd.Timestamp(origin_ts)
    row["predicted_log_return"] = 0.0
    row["origin_log_return"] = 0.0
    row["predicted_price"] = float(origin_price)
    if "actual_price" in path.columns:
        row["actual_price"] = float(origin_price)
    if "price_q0.1" in path.columns:
        row["price_q0.1"] = float(origin_price)
        row["price_q0.9"] = float(origin_price)
    return pd.concat([pd.DataFrame([row]), path], ignore_index=True)


def forecast_at_origin(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    origin_pos: int,
    market: dict[str, pd.Series],
    asof_mode: str,
    gap_model=None,
    bias=None,
    q10=None,
    q90=None,
) -> pd.DataFrame:
    hist = features.iloc[: origin_pos + 1].copy()
    origin_ts = pd.Timestamp(hist[config.TIMESTAMP_COLUMN].iloc[-1])
    origin_price = float(hist["close"].iloc[-1])
    returns = ttm_returns(hist)
    gap_hat = None
    if float(hist["is_session_close"].iloc[-1]) > 0.5:
        if gap_model is None:
            gap_model = fit_gap_model(hist, market, origin_ts, asof_mode)
        gap_hat = predict_gap(
            gap_model,
            hist.iloc[-1],
            origin_ts,
            pd.DatetimeIndex(hist[config.TIMESTAMP_COLUMN]),
            market,
            asof_mode,
        )
    path = reconstruct_path(
        returns,
        origin_price,
        origin_ts,
        pd.DatetimeIndex(hist[config.TIMESTAMP_COLUMN]),
        first_step_gap=gap_hat,
    )
    holdout = ohlcv[ohlcv.index > origin_ts].iloc[: config.PREDICTION_LENGTH]
    path = attach_actuals(path, holdout)
    path = apply_calibration(path, origin_price, bias, q10, q90)
    path.attrs["origin_ts"] = origin_ts
    path.attrs["origin_price"] = origin_price
    path.attrs["gap_hat"] = gap_hat
    return path
