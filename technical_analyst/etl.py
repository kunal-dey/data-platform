"""Simple hourly ETL helpers: CSV symbols, price rows, next-hour fall stats."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .data import download_market_returns, download_ohlcv
from .features import build_features
from .models import fit_gap_model, predict_gap, ttm_first_step_returns
from .signal import _label, _signed_strength

DEFAULT_SYMBOLS_CSV = Path(__file__).resolve().parents[1] / "data" / "nse_symbols.csv"
HOURLY_PRICES_CSV = Path(__file__).resolve().parents[1] / "outputs" / "hourly_prices.csv"
DOWNSIDE_WINDOW = 120
CSV_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "close",
    "volume",
    "next_hour_direction",
    "next_hour_return",
    "next_hour_strength",
    "fall_probability",
    "fall_q10",
    "ingested_at",
]


def yfinance_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if not symbol or symbol == "SYMBOL":
        raise ValueError("empty symbol")
    return symbol if "." in symbol else f"{symbol}.NS"


def read_symbols_csv(path: str | Path | None = None) -> list[str]:
    csv_path = Path(path) if path else DEFAULT_SYMBOLS_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Symbol CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    col = "symbol" if "symbol" in frame.columns else frame.columns[0]
    symbols = []
    seen: set[str] = set()
    for raw in frame[col].tolist():
        ticker = yfinance_symbol(raw)
        if ticker not in seen:
            seen.add(ticker)
            symbols.append(ticker)
    if not symbols:
        raise ValueError(f"No symbols in {csv_path}")
    return symbols


def hourly_downside(closes: pd.Series, window: int = DOWNSIDE_WINDOW) -> tuple[float, float]:
    """Empirical P(next hour is down) and 10th-percentile hourly return (how far it can fall)."""
    stats = rolling_hourly_downside(closes, window=window)
    if stats.empty:
        return 0.5, 0.0
    last = stats.iloc[-1]
    fall_p = last["fall_probability"]
    fall_q10 = last["fall_q10"]
    return (
        0.5 if pd.isna(fall_p) else float(fall_p),
        0.0 if pd.isna(fall_q10) else float(fall_q10),
    )


def rolling_hourly_downside(closes: pd.Series, window: int = DOWNSIDE_WINDOW) -> pd.DataFrame:
    """As-of-each-bar downside stats from the trailing hourly returns (no lookahead)."""
    returns = np.log(closes.astype(float)).diff()
    min_periods = max(window // 4, 1) if window else 1
    fall_probability = returns.lt(0).rolling(window, min_periods=min_periods).mean()
    fall_q10 = returns.rolling(window, min_periods=min_periods).quantile(0.10)
    return pd.DataFrame({"fall_probability": fall_probability, "fall_q10": fall_q10}, index=closes.index)


def _parse_ts(value) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    try:
        if pd.isna(value):
            return pd.NaT
    except (ValueError, TypeError):
        pass
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _utc_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_parse_ts(value) for value in values])


def _scalar(value):
    try:
        if value is None or pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _row_at(lookup: pd.DataFrame, key: pd.Timestamp) -> pd.Series | None:
    if lookup.empty:
        return None
    pos = lookup.index.get_indexer([key])
    if len(pos) == 0 or pos[0] < 0:
        return None
    hit = lookup.iloc[pos[0]]
    return hit.iloc[-1] if isinstance(hit, pd.DataFrame) else hit


def _apply_session_close_gaps(features: pd.DataFrame, market: dict, predicted: pd.Series) -> pd.Series:
    """Overnight gap model replaces TTM's first step when the origin is a session close."""
    if predicted.empty or "is_session_close" not in features.columns:
        return predicted
    origin_ts = pd.Timestamp(features[config.TIMESTAMP_COLUMN].iloc[-1])
    gap_model = fit_gap_model(features, market, origin_ts, config.ASOF_MODE)
    if gap_model is None:
        return predicted
    index = pd.DatetimeIndex(features[config.TIMESTAMP_COLUMN])
    out = predicted.copy()
    pred_pos = {_parse_ts(ts): i for i, ts in enumerate(out.index)}
    for loc in features.index[features["is_session_close"].astype(float) > 0.5]:
        ts = _parse_ts(features.at[loc, config.TIMESTAMP_COLUMN])
        pos = pred_pos.get(ts)
        if pos is None:
            continue
        gap = predict_gap(
            gap_model,
            features.loc[loc],
            ts,
            index,
            market,
            config.ASOF_MODE,
        )
        if np.isfinite(gap):
            out.iloc[pos] = float(gap)
    return out


def forecast_hourly_table(ohlcv: pd.DataFrame, market: dict | None = None) -> pd.DataFrame:
    """Next-hour TTM forecast and trailing downside stats for every hourly bar."""
    market = market or {}
    index = _utc_index(ohlcv.index)
    out = pd.DataFrame(
        {
            "next_hour_direction": pd.Series(pd.NA, index=index, dtype="object"),
            "next_hour_return": np.nan,
            "next_hour_strength": np.nan,
            "fall_probability": np.nan,
            "fall_q10": np.nan,
        },
        index=index,
    )
    downside = rolling_hourly_downside(ohlcv["close"])
    downside.index = index
    out["fall_probability"] = downside["fall_probability"].to_numpy(dtype=float)
    out["fall_q10"] = downside["fall_q10"].to_numpy(dtype=float)

    try:
        features = build_features(ohlcv, market)
        print(f"  TTM sliding windows on {len(features)} feature rows...")
        predicted = ttm_first_step_returns(features)
        predicted.index = _utc_index(predicted.index)
        predicted = predicted[~predicted.index.duplicated(keep="last")]
        predicted = _apply_session_close_gaps(features, market, predicted)
        aligned = predicted.reindex(index)
        finite = aligned.to_numpy(dtype=float)
        threshold = float(config.SIGNAL_SLOPE_THRESHOLD)
        directions = []
        strengths = []
        for value in finite:
            if np.isfinite(value):
                directions.append(_label(value, threshold))
                strengths.append(_signed_strength(value, threshold))
            else:
                directions.append(pd.NA)
                strengths.append(np.nan)
        out["next_hour_return"] = finite
        out["next_hour_direction"] = directions
        out["next_hour_strength"] = strengths
    except Exception as exc:
        print(f"  TTM skipped ({exc})")
    return out


def hourly_price_rows(
    ohlcv: pd.DataFrame,
    symbol: str,
    *,
    forecasts: pd.DataFrame | None = None,
    ingested_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """One merge row per hourly bar, with next-hour forecast columns when available."""
    ingested_at = ingested_at or datetime.now(timezone.utc)
    ticker = yfinance_symbol(symbol)
    if forecasts is None:
        forecasts = pd.DataFrame(
            columns=[
                "next_hour_direction",
                "next_hour_return",
                "next_hour_strength",
                "fall_probability",
                "fall_q10",
            ]
        )
    forecast_index = _utc_index(forecasts.index) if len(forecasts) else pd.DatetimeIndex([])
    lookup = forecasts.copy()
    lookup.index = forecast_index
    rows: list[dict[str, Any]] = []
    for ts, row in ohlcv.iterrows():
        key = _parse_ts(ts)
        hit = _row_at(lookup, key)
        rows.append(
            {
                "symbol": ticker,
                "timestamp": key.to_pydatetime(),
                "open": float(row["open"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "next_hour_direction": None if hit is None else _scalar(hit.get("next_hour_direction")),
                "next_hour_return": None if hit is None else _scalar(hit.get("next_hour_return")),
                "next_hour_strength": None if hit is None else _scalar(hit.get("next_hour_strength")),
                "fall_probability": None if hit is None else _scalar(hit.get("fall_probability")),
                "fall_q10": None if hit is None else _scalar(hit.get("fall_q10")),
                "ingested_at": ingested_at,
            }
        )
    return rows


def _next_hour_forecast_table(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    try:
        market = download_market_returns(ticker, quiet=True)
        print(f"  TTM next-hour forecast on {len(ohlcv)} bars...")
        return forecast_hourly_table(ohlcv, market)
    except Exception as exc:
        print(f"  {ticker}: forecast skipped ({exc})")
        return None


def iter_hourly_price_rows(tickers: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Yield raw hourly dicts for CSV upsert (do not iterate a dlt resource)."""
    for ticker in tickers:
        print(f"Hourly ETL {ticker}...")
        try:
            ohlcv, _ = download_ohlcv(ticker)
        except Exception as exc:
            print(f"  {ticker}: no bars, skipped ({exc})")
            continue
        if ohlcv.empty:
            print(f"  {ticker}: no bars, skipped")
            continue
        forecasts = _next_hour_forecast_table(ohlcv, ticker)
        rows = hourly_price_rows(ohlcv, ticker, forecasts=forecasts)
        filled = sum(1 for row in rows if row["next_hour_direction"])
        latest = rows[-1]
        print(
            f"  {len(rows)} bars  forecasted={filled}  last {latest['timestamp']}  "
            f"next={latest['next_hour_direction']}  "
            f"fall_p={0 if latest['fall_probability'] is None or pd.isna(latest['fall_probability']) else latest['fall_probability']:.0%}  "
            f"fall_q10={0 if latest['fall_q10'] is None or pd.isna(latest['fall_q10']) else latest['fall_q10']:+.2%}"
        )
        yield from rows


def upsert_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Merge-upsert hourly rows into a CSV on (symbol, timestamp)."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = [row for row in rows if isinstance(row, dict) and "timestamp" in row]
    incoming = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    if incoming.empty:
        if dest.is_file():
            return dest
        incoming = pd.DataFrame(columns=CSV_COLUMNS)
    existing = pd.DataFrame(columns=CSV_COLUMNS)
    if dest.is_file() and dest.stat().st_size > 0:
        loaded = pd.read_csv(dest)
        if "timestamp" in loaded.columns:
            existing = loaded
    combined = pd.concat([existing, incoming], ignore_index=True)
    if "timestamp" not in combined.columns:
        raise ValueError("hourly rows are missing a timestamp column")
    combined["timestamp"] = pd.to_datetime(combined["timestamp"].map(_parse_ts), utc=True)
    combined["ingested_at"] = pd.to_datetime(combined["ingested_at"].map(_parse_ts), utc=True)
    combined["_ts_key"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    combined["symbol"] = combined["symbol"].astype(str)
    combined = combined.drop_duplicates(subset=["symbol", "_ts_key"], keep="last")
    combined = combined.drop(columns=["_ts_key"])
    combined = combined.sort_values(["symbol", "timestamp"])
    for col in CSV_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined[CSV_COLUMNS].to_csv(dest, index=False)
    return dest
