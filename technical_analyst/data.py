"""Optional yfinance downloads for the CLI. The package API does not need this."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from . import config


def _currency(ticker: yf.Ticker) -> str:
    try:
        value = getattr(ticker.fast_info, "currency", None)
        return str(value) if value else "unknown"
    except Exception:
        return "unknown"


def download_ohlcv(symbol: str) -> tuple[pd.DataFrame, str]:
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period=config.PERIOD, interval=config.INTERVAL, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"No yfinance data for {symbol!r}. NSE tickers look like INFY.NS, TCS.NS.")

    df = raw.copy()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    needed = ["open", "close", "volume"]
    if any(c not in df.columns for c in needed):
        raise ValueError(f"Expected {needed}, got {list(df.columns)}")

    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["volume"] >= 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=needed)
    if df.empty:
        raise ValueError(f"All rows dropped after cleaning {symbol!r}.")
    return df[needed], _currency(ticker)


def _daily_log_return(symbol: str) -> pd.Series | None:
    yf_log = logging.getLogger("yfinance")
    prev_level = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    try:
        raw = yf.Ticker(symbol).history(period=config.PERIOD, interval="1d", auto_adjust=True)
    except Exception:
        return None
    finally:
        yf_log.setLevel(prev_level)
    if raw is None or raw.empty:
        return None
    raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
    if "close" not in raw.columns:
        return None
    close = raw["close"].astype(float)
    close = close[close > 0].dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    if close.empty:
        return None
    return np.log(close).diff().rename(symbol)


def infer_adr_ticker(nse_symbol: str) -> str | None:
    base = nse_symbol.split(".")[0].upper()
    return config.ADR_TICKERS.get(base)


def download_market_returns(symbol: str, *, quiet: bool = False) -> dict[str, pd.Series]:
    tickers = dict(config.MARKET_TICKERS)
    adr = infer_adr_ticker(symbol)
    if adr:
        tickers["adr"] = adr
    out: dict[str, pd.Series] = {}
    for name, ticker in tickers.items():
        series = _daily_log_return(ticker)
        if series is not None:
            out[name] = series
            if not quiet:
                print(f"  {name} ({ticker}): {series.dropna().shape[0]} days")
        elif not quiet:
            print(f"  {name} ({ticker}): skipped")
    return out
