"""Stationary hourly features. Target is log return, not price."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def localize_cutoff(cutoff: str | pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(cutoff)
    if index.tz is not None:
        ts = ts.tz_localize(index.tz) if ts.tzinfo is None else ts.tz_convert(index.tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def infer_bars_per_day(index: pd.DatetimeIndex) -> int:
    if index.empty:
        return 7
    return max(int(round(float(pd.Series(index.date).value_counts().median()))), 1)


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(series: pd.Series) -> pd.Series:
    fast = series.ewm(span=config.MACD_FAST, adjust=False).mean()
    slow = series.ewm(span=config.MACD_SLOW, adjust=False).mean()
    macd = fast - slow
    return macd - macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _asof_daily(hourly_index: pd.DatetimeIndex, daily: pd.Series) -> pd.Series:
    if daily is None or daily.empty:
        return pd.Series(0.0, index=hourly_index)
    right = daily.dropna().to_frame("value")
    right["ts"] = right.index
    if hourly_index.tz is not None:
        right["ts"] = right["ts"].dt.tz_localize(hourly_index.tz) if right["ts"].dt.tz is None else right["ts"].dt.tz_convert(hourly_index.tz)
    elif right["ts"].dt.tz is not None:
        right["ts"] = right["ts"].dt.tz_localize(None)
    left = pd.DataFrame({config.TIMESTAMP_COLUMN: hourly_index})
    merged = pd.merge_asof(
        left.sort_values(config.TIMESTAMP_COLUMN),
        right.sort_values("ts"),
        left_on=config.TIMESTAMP_COLUMN,
        right_on="ts",
        direction="backward",
    )
    return pd.Series(merged["value"].to_numpy(), index=hourly_index).fillna(0.0)


def session_masks(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = pd.Series(index.date, index=index)
    is_open = dates.ne(dates.shift(1))
    is_close = dates.ne(dates.shift(-1))
    hours_since = pd.Series(index, index=index).diff().dt.total_seconds() / 3600.0
    is_open.iloc[0] = True
    is_close.iloc[-1] = True
    return is_open.fillna(False), is_close.fillna(False), hours_since.fillna(1.0)


def next_trading_bars(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    if n <= 0 or index.empty:
        return pd.DatetimeIndex([])
    deltas = pd.Series(index).diff().dropna()
    intra = deltas[deltas <= pd.Timedelta(hours=4)]
    step = pd.Timedelta(intra.median()) if not intra.empty else pd.Timedelta(hours=1)
    minutes = index.hour * 60 + index.minute
    start_m, end_m = int(np.percentile(minutes, 5)), int(np.percentile(minutes, 95))
    if end_m <= start_m:
        start_m, end_m = 9 * 60 + 15, 15 * 60 + 15

    def at_min(ts: pd.Timestamp, m: int) -> pd.Timestamp:
        return ts.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0)

    out: list[pd.Timestamp] = []
    t = index[-1] + step
    guard = 0
    while len(out) < n and guard < n * 48:
        guard += 1
        if t.weekday() >= 5:
            t = at_min(t + pd.Timedelta(days=7 - t.weekday()), start_m)
            continue
        minute = t.hour * 60 + t.minute
        if minute < start_m:
            t = at_min(t, start_m)
            continue
        if minute > end_m:
            t = at_min(t + pd.Timedelta(days=1), start_m)
            continue
        out.append(t)
        t = t + step
    return pd.DatetimeIndex(out)


def information_cutoff(origin_ts: pd.Timestamp, index: pd.DatetimeIndex, asof_mode: str) -> pd.Timestamp:
    origin_ts = pd.Timestamp(origin_ts)
    if asof_mode == "last_bar":
        return origin_ts
    seen = index[index <= origin_ts]
    future = next_trading_bars(pd.DatetimeIndex(seen if len(seen) else index), n=1)
    nxt = pd.Timestamp(future[0]) if len(future) else origin_ts + pd.Timedelta(hours=18)
    return nxt - pd.Timedelta(minutes=1)


def build_features(ohlcv: pd.DataFrame, market_returns: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    df = ohlcv.copy()
    bars = infer_bars_per_day(df.index)
    rsi_len = config.RSI_LOOKBACK_DAYS * bars
    z_win = max(config.ZSCORE_LOOKBACK_DAYS * bars, config.MACD_SLOW + config.MACD_SIGNAL)
    w5, w20 = 5 * bars, 20 * bars
    is_open, is_close, hours_since = session_masks(df.index)
    log_close = np.log(df["close"].astype(float))
    log_open = np.log(df["open"].astype(float))

    df[config.TARGET_COLUMN] = log_close.diff()
    df["log_volume_norm"] = _zscore(np.log1p(df["volume"].astype(float)), z_win)
    df["rsi_14d"] = (_rsi(df["close"].astype(float), rsi_len) - 50.0) / 50.0
    df["macd_hist"] = _zscore(_macd_hist(log_close), z_win)
    df["ret_5d"] = log_close.diff(w5)
    df["ret_20d"] = log_close.diff(w20)
    df["vol_5d"] = df[config.TARGET_COLUMN].rolling(w5, min_periods=w5).std(ddof=0)
    df["vol_20d"] = df[config.TARGET_COLUMN].rolling(w20, min_periods=w20).std(ddof=0)
    df["hours_since_prev"] = np.log1p(hours_since.clip(lower=0.0))
    df["is_session_open"] = is_open.astype(float)
    df["is_session_close"] = is_close.astype(float)
    df["volume_change"] = _zscore(np.log1p(df["volume"].astype(float)).diff(), w20)

    prev_close = df["close"].where(is_close).ffill().shift(1)
    overnight_gap = np.where(is_open, log_open - np.log(prev_close), 0.0)
    daily_close = df.loc[is_close, "close"]
    df["prev_day_return"] = np.log(daily_close / daily_close.shift(1)).reindex(df.index).ffill().shift(1)
    df["last_gap"] = pd.Series(overnight_gap, index=df.index).replace(0.0, np.nan).ffill().shift(1)
    df["is_weekend_gap"] = ((hours_since >= 40) & is_open).astype(float)
    df["nifty_log_return"] = (
        _asof_daily(df.index, market_returns["nifty"]) if market_returns and "nifty" in market_returns else 0.0
    )

    out = df.reset_index()
    out = out.rename(columns={out.columns[0]: config.TIMESTAMP_COLUMN})
    out[config.TIMESTAMP_COLUMN] = pd.to_datetime(out[config.TIMESTAMP_COLUMN], utc=False)

    ttm_cols = [c for c in config.FEATURE_COLUMNS if c in out.columns]
    extra = [
        "open",
        "close",
        "is_session_open",
        "is_session_close",
        "prev_day_return",
        "ret_20d",
        "vol_20d",
        "volume_change",
        "last_gap",
        "is_weekend_gap",
    ]
    keep = list(dict.fromkeys([config.TIMESTAMP_COLUMN, *[c for c in extra if c in out.columns], *ttm_cols]))
    out = out[keep].replace([np.inf, -np.inf], np.nan)
    out[ttm_cols] = out[ttm_cols].fillna(0.0)
    out = out.dropna(subset=[config.TARGET_COLUMN, "close"]).reset_index(drop=True)
    if len(out) < config.CONTEXT_LENGTH:
        raise ValueError(f"Need {config.CONTEXT_LENGTH} bars after indicators, got {len(out)}.")
    return out


def session_close_origins(features: pd.DataFrame, n_days: int) -> list[int]:
    flags = features.get("is_session_close", pd.Series(False, index=features.index))
    idxs = [i for i, flag in enumerate(flags.tolist()) if flag]
    return [i for i in idxs if i + 1 >= config.CONTEXT_LENGTH and i + 1 < len(features)][-n_days:]
