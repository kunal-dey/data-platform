"""BUY / HOLD / SELL from forecast-path shape, not predicted rupee level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _finite_positive(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _ols_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    coef = np.polyfit(np.arange(len(y), dtype=float), y, 1)
    slope = float(coef[0])
    return slope if np.isfinite(slope) else 0.0


def _finite_return(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _signed_strength(score: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return float(np.clip(score / threshold, -1.0, 1.0))


def _label(score: float, threshold: float) -> str:
    if score > threshold:
        return "BUY"
    if score < -threshold:
        return "SELL"
    return "HOLD"


def _stance(label: str) -> str:
    return {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}[label]


def _confidence(score: float, steps: np.ndarray, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    strength = abs(score) / threshold
    if len(steps) == 0:
        return float(np.clip(np.tanh(strength), 0.0, 1.0))
    expected = np.sign(score) if score != 0 else 0.0
    if expected == 0:
        return float(np.clip(1.0 - np.tanh(strength), 0.0, 1.0))
    agreement = float(np.mean(np.sign(steps) == expected))
    return float(np.clip(np.tanh(strength) * agreement, 0.0, 1.0))


def directional_signal(path: pd.DataFrame, threshold: float | None = None) -> dict:
    """Score the 24-bar forecast shape. Last close vs predicted rupees is ignored."""
    threshold = float(threshold if threshold is not None else config.SIGNAL_PATH_RETURN_THRESHOLD)
    step_threshold = float(config.SIGNAL_SLOPE_THRESHOLD)
    pred = _finite_positive(path["predicted_price"])
    logp = np.log(pred)
    steps = np.diff(logp)
    slope = _ols_slope(logp)
    path_return = float(logp[-1] - logp[0]) if len(logp) else 0.0
    label = _label(path_return, threshold)
    n_up = int(np.sum(steps > step_threshold))
    n_down = int(np.sum(steps < -step_threshold))
    n_flat = int(len(steps) - n_up - n_down)
    out = {
        "signal": label,
        "direction": _stance(label),
        "slope": slope,
        "slope_pct_per_bar": slope * 100.0,
        "path_return": path_return,
        "path_return_pct": path_return * 100.0,
        "mean_step": float(np.mean(steps)) if len(steps) else 0.0,
        "up_steps": n_up,
        "down_steps": n_down,
        "flat_steps": n_flat,
        "confidence": _confidence(path_return, steps, threshold),
        "threshold": threshold,
    }
    if "actual_price" in path.columns:
        actual = _finite_positive(path["actual_price"])
        if len(actual) >= 2:
            act_log = np.log(actual)
            act_return = float(act_log[-1] - act_log[0])
            act_label = _label(act_return, threshold)
            out.update(
                {
                    "actual_slope": _ols_slope(act_log),
                    "actual_path_return": act_return,
                    "actual_path_return_pct": act_return * 100.0,
                    "actual_signal": act_label,
                    "actual_direction": _stance(act_label),
                    "direction_match": bool(
                        np.sign(path_return) == np.sign(act_return) and path_return != 0 and act_return != 0
                    ),
                    "signal_match": label == act_label,
                }
            )
    return out


def step_returns(path: pd.DataFrame) -> list[float]:
    r = path["predicted_log_return"].to_numpy(dtype=float)
    return [_finite_return(x) for x in r]


def step_labels(path: pd.DataFrame, threshold: float | None = None) -> list[str]:
    threshold = float(threshold if threshold is not None else config.SIGNAL_SLOPE_THRESHOLD)
    return [_label(x, threshold) for x in step_returns(path)]


def step_strengths(path: pd.DataFrame, threshold: float | None = None) -> list[float]:
    threshold = float(threshold if threshold is not None else config.SIGNAL_SLOPE_THRESHOLD)
    return [_signed_strength(x, threshold) for x in step_returns(path)]


def attach_step_signals(path: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    out = path.copy()
    out["step_signal"] = step_labels(out, threshold=threshold)
    out["step_score"] = step_returns(out)
    out["step_strength"] = step_strengths(out, threshold=threshold)
    return out


def signal_hit_rate(paths: list[pd.DataFrame], threshold: float | None = None) -> dict:
    scored = [
        row
        for row in (directional_signal(path, threshold=threshold) for path in paths)
        if "actual_signal" in row
    ]
    if not scored:
        return {}
    return {
        "n": len(scored),
        "signal_accuracy": float(np.mean([row["signal_match"] for row in scored])),
        "direction_accuracy": float(np.mean([row["direction_match"] for row in scored])),
        "pred_buy": int(sum(row["signal"] == "BUY" for row in scored)),
        "pred_hold": int(sum(row["signal"] == "HOLD" for row in scored)),
        "pred_sell": int(sum(row["signal"] == "SELL" for row in scored)),
    }
