"""CLI: download prices, then call predict_direction."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .api import predict_direction
from .data import download_market_returns, download_ohlcv
from .features import information_cutoff, localize_cutoff
from .models import prepend_origin
from .signal import attach_step_signals

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("tsfm_public").setLevel(logging.WARNING)


def _save_plot(symbol: str, history: pd.DataFrame, path: pd.DataFrame, dest: Path, currency: str, origin_ts, asof_ts, signal: dict | None = None) -> None:
    hist = history.tail(80)
    forecast = path.iloc[1:] if len(path) > 1 else path
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax, ax_r = axes
    ax.plot(hist[config.TIMESTAMP_COLUMN], hist["close"], label="History", color="#1f4e79")
    if origin_ts is not None:
        ax.axvline(origin_ts, color="#666666", linestyle="--", linewidth=1, label="Forecast origin")
    if "actual_price" in path.columns and path["actual_price"].notna().any():
        ax.plot(path[config.TIMESTAMP_COLUMN], path["actual_price"], label="Actual", color="#2e7d32", marker="o", markersize=3.5, zorder=3)
    ax.plot(path[config.TIMESTAMP_COLUMN], path["predicted_price"], label="Predicted", color="#c45911", marker="o", markersize=3.5, zorder=4)
    if "price_q0.1" in path.columns and "price_q0.9" in path.columns:
        ax.fill_between(path[config.TIMESTAMP_COLUMN], path["price_q0.1"], path["price_q0.9"], color="#c45911", alpha=0.18, label="P10-P90")
    title = f"{symbol}  |  last known price x exp(predicted return)"
    if signal:
        title = (
            f"{symbol}  |  {signal['signal']} ({signal['direction']})  |  "
            f"path {signal['path_return_pct']:+.2f}%  |  conf {signal['confidence']:.0%}"
        )
    ax.set_title(title)
    ax.set_ylabel(f"Price ({currency})" if currency else "Price")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    pred0 = float(forecast["predicted_price"].iloc[0]) if len(forecast) else np.nan
    if pred0 > 0:
        ax_r.plot(
            forecast[config.TIMESTAMP_COLUMN],
            np.log(forecast["predicted_price"].astype(float) / pred0) * 100.0,
            label="Predicted path return",
            color="#c45911",
            marker="o",
            markersize=3,
        )
    if "actual_price" in forecast.columns and forecast["actual_price"].notna().any():
        actual = forecast.dropna(subset=["actual_price"])
        if len(actual) and float(actual["actual_price"].iloc[0]) > 0:
            act0 = float(actual["actual_price"].iloc[0])
            ax_r.plot(
                actual[config.TIMESTAMP_COLUMN],
                np.log(actual["actual_price"].astype(float) / act0) * 100.0,
                label="Actual path return",
                color="#2e7d32",
                marker="o",
                markersize=3,
            )
    ax_r.axhline(0.0, color="#888888", linewidth=0.8)
    ax_r.set_ylabel("Path return (%)")
    ax_r.set_xlabel("Time")
    ax_r.legend(loc="upper left")
    ax_r.grid(True, alpha=0.3)
    if asof_ts is not None:
        ax_r.text(0.01, 0.04, f"Info cutoff: {pd.Timestamp(asof_ts)}", transform=ax_r.transAxes, fontsize=8, color="#444")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="24-hour direction forecast from stock prices (Granite TTM-R3)")
    parser.add_argument("symbol", nargs="?", default=config.STOCK_SYMBOL, help="yfinance ticker, e.g. INFY.NS")
    parser.add_argument("--cutoff", default=config.HOLDOUT_BEFORE, help="Use only data before this date (YYYY-MM-DD)")
    parser.add_argument("--live", action="store_true", help="Forecast from the latest bar (no holdout)")
    parser.add_argument("--fast", action="store_true", help="Skip rolling calibration")
    parser.add_argument("--plot", action="store_true", help="Also save the diagnostic png/csv")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    cutoff = None if args.live else args.cutoff
    print(f"Downloading {symbol}...")
    ohlcv, currency = download_ohlcv(symbol)
    print(f"  {len(ohlcv)} hourly bars, {currency}, last {ohlcv.index.max()}")
    market = download_market_returns(symbol)

    result = predict_direction(
        ohlcv["close"],
        timestamps=ohlcv.index,
        volumes=ohlcv["volume"],
        opens=ohlcv["open"],
        market_returns=market,
        cutoff=cutoff,
        calibrate=not args.fast,
        quiet=False,
    )
    print(f"signal: {result.signal}  score={result.score:+.2f}  path {result.path_return:+.2%}  conf {result.confidence:.0%}")
    if result.gap is not None:
        print(f"overnight gap: {result.gap:.2%}")
    print("hour  direction  score     strength")
    for i, step in enumerate(result.steps, 1):
        print(f"{i:>4}  {step['direction']:<9}  {step['score']:+.2%}  {step['strength']:+.2f}")

    if args.plot:
        if result.path is None:
            raise RuntimeError("forecast path missing; cannot plot.")
        path = result.path
        origin_ts = result.origin_time
        origin_price = result.origin_price
        plotted = attach_step_signals(prepend_origin(path, origin_ts, origin_price))
        out_dir = Path(config.OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = symbol.lower().replace(".", "_")
        csv_path = out_dir / f"{stem}_24h_forecast.csv"
        png_path = out_dir / f"{stem}_24h_forecast.png"
        plotted.to_csv(csv_path, index=False)
        history = pd.DataFrame({config.TIMESTAMP_COLUMN: ohlcv.index, "close": ohlcv["close"].to_numpy()})
        if cutoff:
            cut = localize_cutoff(cutoff, ohlcv.index)
            history = history[history[config.TIMESTAMP_COLUMN] < cut]
        _save_plot(
            symbol,
            history,
            plotted,
            png_path,
            currency,
            origin_ts,
            information_cutoff(origin_ts, ohlcv.index, config.ASOF_MODE),
            signal={
                "signal": result.signal,
                "direction": {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}[result.signal],
                "path_return_pct": result.path_return * 100.0,
                "confidence": result.confidence,
            },
        )
        print(f"saved {csv_path}")
        print(f"saved {png_path}")


if __name__ == "__main__":
    main()
