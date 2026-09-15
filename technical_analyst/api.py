"""Public API: list of prices in, directions out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .features import build_features, localize_cutoff, session_close_origins
from .models import fit_gap_model, forecast_at_origin
from .signal import directional_signal, step_labels, step_returns, step_strengths


@dataclass(frozen=True)
class DirectionResult:
    """Overall path signal plus one direction and score per forecast bar."""

    signal: str
    directions: list[str]
    scores: list[float]
    strengths: list[float]
    score: float
    confidence: float
    path_return: float
    origin_price: float
    origin_time: pd.Timestamp | None = None
    gap: float | None = None
    path: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __iter__(self):
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.directions)

    def __getitem__(self, index):
        return self.steps[index]

    def __str__(self) -> str:
        return f"{self.signal} ({self.score:+.2f})"

    @property
    def steps(self) -> list[dict]:
        return [
            {"direction": direction, "score": score, "strength": strength}
            for direction, score, strength in zip(self.directions, self.scores, self.strengths)
        ]

    def to_records(
        self,
        symbol: str,
        *,
        ingested_at: datetime | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Flatten into a signal row + hourly step rows for dlt / reverse ETL."""
        ingested_at = ingested_at or datetime.now(timezone.utc)
        origin = self.origin_time
        if isinstance(origin, pd.Timestamp):
            origin = origin.to_pydatetime()
        signal_row = {
            "symbol": str(symbol).strip().upper(),
            "origin_time": origin,
            "origin_price": float(self.origin_price),
            "signal": self.signal,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "path_return": float(self.path_return),
            "gap": None if self.gap is None else float(self.gap),
            "ingested_at": ingested_at,
        }
        step_rows = [
            {
                "symbol": signal_row["symbol"],
                "origin_time": origin,
                "horizon": index,
                "direction": step["direction"],
                "score": float(step["score"]),
                "strength": float(step["strength"]),
                "ingested_at": ingested_at,
            }
            for index, step in enumerate(self.steps, start=1)
        ]
        return signal_row, step_rows


def _as_floats(values: Sequence[float] | pd.Series | np.ndarray) -> list[float]:
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    out = [float(x) for x in values]
    if not out:
        raise ValueError("prices must be a non-empty list.")
    if any(not np.isfinite(x) or x <= 0 for x in out):
        raise ValueError("every price must be a finite number greater than 0.")
    return out


def _session_index(n: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now()
    if end.tzinfo is not None:
        end = pd.Timestamp(end.to_pydatetime().replace(tzinfo=None))
    hours = list(range(9, 16))
    stamps: list[pd.Timestamp] = []
    day = end.normalize()
    while len(stamps) < n:
        if day.weekday() < 5:
            for hour in reversed(hours):
                ts = day + pd.Timedelta(hours=hour, minutes=15)
                if ts <= end:
                    stamps.append(ts)
                if len(stamps) >= n:
                    break
        day -= pd.Timedelta(days=1)
    return pd.DatetimeIndex(sorted(stamps[-n:]))


def prices_to_ohlcv(
    prices: Sequence[float] | pd.Series | np.ndarray,
    timestamps: Sequence | pd.DatetimeIndex | None = None,
    *,
    volumes: Sequence[float] | None = None,
    opens: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Turn a price list into the OHLCV frame the models expect."""
    if isinstance(prices, pd.Series) and timestamps is None and isinstance(prices.index, pd.DatetimeIndex):
        timestamps = prices.index
        close = _as_floats(prices)
    else:
        close = _as_floats(prices)
    n = len(close)
    if timestamps is None:
        index = _session_index(n)
    else:
        index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
        if len(index) != n:
            raise ValueError(f"timestamps length {len(index)} != prices length {n}.")
        index = index[~index.duplicated(keep="last")]
        if len(index) != n:
            raise ValueError("timestamps must be unique.")
    if volumes is None:
        volume = np.zeros(n, dtype=float)
    else:
        volume = np.asarray(list(volumes), dtype=float)
        if len(volume) != n:
            raise ValueError("volumes length must match prices.")
    if opens is None:
        open_px = np.asarray(close, dtype=float)
    else:
        open_px = np.asarray(list(opens), dtype=float)
        if len(open_px) != n:
            raise ValueError("opens length must match prices.")
    frame = pd.DataFrame({"open": open_px, "close": close, "volume": volume}, index=index)
    return frame.sort_index()


def _calibrate(resid: pd.DataFrame):
    if resid.empty:
        return None, None, None
    h_max = config.PREDICTION_LENGTH
    bias = np.zeros(h_max)
    q10 = np.zeros(h_max)
    q90 = np.zeros(h_max)
    for h in range(h_max):
        sl = resid.loc[resid["horizon"] == h, "return_residual"]
        if sl.empty:
            continue
        bias[h] = float(sl.mean())
        centered = sl - sl.mean()
        q10[h] = float(-centered.quantile(0.90))
        q90[h] = float(-centered.quantile(0.10))
    return bias, q10, q90


def _residuals(paths: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for path in paths:
        origin_price = path.attrs.get("origin_price")
        if not origin_price:
            continue
        for h, row in path.iterrows():
            actual, pred = row.get("actual_price"), row.get("predicted_price")
            if pd.isna(actual) or pd.isna(pred) or actual <= 0:
                continue
            pred_r = np.log(pred / origin_price)
            act_r = np.log(actual / origin_price)
            rows.append(
                {
                    "horizon": int(h),
                    "return_residual": float(pred_r - act_r),
                    "price_residual": float(pred - actual),
                    "actual_price": float(actual),
                }
            )
    return pd.DataFrame(rows)


def _forecast_ohlcv(
    ohlcv: pd.DataFrame,
    market: dict[str, pd.Series] | None = None,
    *,
    cutoff: str | None = None,
    calibrate: bool = False,
    asof_mode: str | None = None,
    quiet: bool = True,
) -> tuple[pd.DataFrame, dict]:
    market = market or {}
    asof_mode = asof_mode or config.ASOF_MODE
    features = build_features(ohlcv, market)
    origin_pos = len(features) - 1
    if cutoff:
        cut = localize_cutoff(cutoff, ohlcv.index)
        hits = (features[config.TIMESTAMP_COLUMN] < cut).to_numpy().nonzero()[0]
        if len(hits) == 0:
            raise ValueError(f"No bars before cutoff {cutoff}.")
        origin_pos = int(hits[-1])

    bias = q10 = q90 = None
    if calibrate:
        pool = [p for p in session_close_origins(features, config.ROLLING_ORIGIN_DAYS) if p < origin_pos]
        if not quiet:
            print(f"Rolling calibration on {len(pool)} prior session closes...")
        paths = []
        for i, pos in enumerate(pool):
            ts = pd.Timestamp(features[config.TIMESTAMP_COLUMN].iloc[pos])
            if not quiet:
                print(f"  {i + 1}/{len(pool)} {ts}")
            hist = features.iloc[: pos + 1]
            gap_model = fit_gap_model(hist, market, ts, asof_mode)
            paths.append(forecast_at_origin(features, ohlcv, pos, market, asof_mode, gap_model=gap_model))
        bias, q10, q90 = _calibrate(_residuals(paths))

    path = forecast_at_origin(features, ohlcv, origin_pos, market, asof_mode, bias=bias, q10=q10, q90=q90)
    return path, directional_signal(path)


def predict_direction(
    prices: Sequence[float] | pd.Series | np.ndarray,
    timestamps: Sequence | pd.DatetimeIndex | None = None,
    *,
    volumes: Sequence[float] | None = None,
    opens: Sequence[float] | None = None,
    market_returns: dict[str, pd.Series] | None = None,
    cutoff: str | None = None,
    calibrate: bool = False,
    quiet: bool = True,
) -> DirectionResult:
    """Forecast the next 24 bars from a price list and return BUY/HOLD/SELL directions.

    Parameters
    ----------
    prices
        Historical closes, oldest first. Need at least 512 bars.
    timestamps
        Optional bar times. If omitted, weekday session hours are synthesized.
        A pandas Series with a DatetimeIndex can be passed as ``prices`` alone.
    """
    ohlcv = prices_to_ohlcv(prices, timestamps, volumes=volumes, opens=opens)
    path, signal = _forecast_ohlcv(
        ohlcv,
        market_returns,
        cutoff=cutoff,
        calibrate=calibrate,
        quiet=quiet,
    )
    returns = step_returns(path)
    return DirectionResult(
        signal=str(signal["signal"]),
        directions=step_labels(path),
        scores=returns,
        strengths=step_strengths(path),
        score=float(np.clip(signal["path_return"] / config.SIGNAL_PATH_RETURN_THRESHOLD, -1.0, 1.0))
        if config.SIGNAL_PATH_RETURN_THRESHOLD
        else 0.0,
        confidence=float(signal["confidence"]),
        path_return=float(signal["path_return"]),
        origin_price=float(path.attrs["origin_price"]),
        origin_time=path.attrs.get("origin_ts"),
        gap=path.attrs.get("gap_hat"),
        path=path,
    )
