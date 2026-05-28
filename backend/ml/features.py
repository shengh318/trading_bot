"""Shared feature computation for ML training and inference."""

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = {"open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol", "target"}


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator feature columns to the DataFrame.

    Operates in-place (adds columns). All features use only past data
    (rolling windows) — no lookahead bias.

    Args:
        data: OHLCV DataFrame with columns: open, high, low, close, volume.
              Index should be datetime-like for day_of_week to work.

    Returns:
        Same DataFrame with additional feature columns.
    """
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # --- Returns ---
    data["ret_1"] = close.pct_change(1)
    data["ret_5"] = close.pct_change(5)
    data["ret_21"] = close.pct_change(21)

    # --- Volatility ---
    data["vol_5"] = data["ret_1"].rolling(5).std()
    data["vol_21"] = data["ret_1"].rolling(21).std()

    # --- SMA ratios ---
    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["close_sma_20"] = close / data["sma_20"]
    data["close_sma_50"] = close / data["sma_50"]

    # --- Volume ratio ---
    data["vol_ma_20"] = volume.rolling(20).mean()
    data["vol_ratio"] = volume / data["vol_ma_20"]

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["rsi"] = 100 - (100 / (1 + rs))
    data["rsi"] = data["rsi"].fillna(50.0)

    # --- MACD ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    # --- ATR (14-period) ---
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    data["atr"] = tr.rolling(14).mean()

    # --- Price position in 20-bar range ---
    data["highest_20"] = high.rolling(20).max()
    data["lowest_20"] = low.rolling(20).min()
    data["price_position"] = (
        (close - data["lowest_20"])
        / (data["highest_20"] - data["lowest_20"]).replace(0, np.nan)
    )

    # --- Day of week (0=Monday, 4=Friday) ---
    if hasattr(data.index, "dtype") and data.index.dtype.kind == "M":
        data["day_of_week"] = data.index.dayofweek
    else:
        data["day_of_week"] = 0

    # --- Volatility ratio (short / long) ---
    data["vol_ratio_5_21"] = data["vol_5"] / data["vol_21"]

    # --- Momentum over multiple windows ---
    data["mom_10"] = close.pct_change(10)
    data["mom_20"] = close.pct_change(20)
    data["mom_60"] = close.pct_change(60)

    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names from a DataFrame that has had compute_features() applied."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]
