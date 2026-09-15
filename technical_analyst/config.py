"""Package defaults. Override at call time; CLI also reads these."""

STOCK_SYMBOL = "INFY.NS"
HOLDOUT_BEFORE = "2026-09-09"

MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r3"
INTERVAL = "1h"
PERIOD = "730d"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 24

RSI_LOOKBACK_DAYS = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ZSCORE_LOOKBACK_DAYS = 20

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "log_return"
FEATURE_COLUMNS = [
    "log_return",
    "log_volume_norm",
    "rsi_14d",
    "macd_hist",
    "ret_5d",
    "vol_5d",
    "hours_since_prev",
    "nifty_log_return",
]
GAP_FEATURE_COLUMNS = [
    "prev_day_return",
    "ret_5d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "volume_change",
    "last_gap",
    "is_weekend_gap",
    "nifty_day_return",
    "nifty_it_day_return",
    "usdinr_day_return",
    "nasdaq_overnight_return",
    "adr_overnight_return",
]
MARKET_TICKERS = {
    "nifty": "^NSEI",
    "nifty_it": "^CNXIT",
    "usdinr": "INR=X",
    "nasdaq": "^IXIC",
}
ADR_TICKERS = {"INFY": "INFY", "TCS": "TCSFY", "WIPRO": "WIT"}

ASOF_MODE = "before_next_open"
BATCH_SIZE = 32
OUTPUT_DIR = "outputs"
ROLLING_ORIGIN_DAYS = 30

SIGNAL_PATH_RETURN_THRESHOLD = 0.005  # 0.5% over the 24-bar path
SIGNAL_SLOPE_THRESHOLD = 0.0015  # ~0.15%/bar for per-step labels
