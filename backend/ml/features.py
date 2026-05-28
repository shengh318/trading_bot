"""Shared feature computation for ML training and inference."""

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = {"open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol", "target"}


def _hurst_exponent(ts: np.ndarray) -> float:
    """Compute Hurst exponent via rescaled range (R/S) method."""
    if len(ts) < 20:
        return 0.5
    ts = np.asarray(ts, dtype=float)
    mean = np.nanmean(ts)
    deviations = ts - mean
    cumsum = np.nancumsum(deviations)
    r = np.nanmax(cumsum) - np.nanmin(cumsum)
    s = np.nanstd(ts)
    if s <= 0 or r <= 0:
        return 0.5
    return np.log(r / s) / np.log(len(ts))


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

    # ── New features ────────────────────────────────────────────────────────

    # --- Bollinger Bands (20,2) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    data["bb_pct_b"] = (close - bb_lower) / bb_range

    # --- ADX (Average Directional Index, 14-period) ---
    high_lag = high.shift(1)
    low_lag = low.shift(1)
    up_move = high - high_lag
    down_move = low_lag - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=data.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=data.index,
    )
    tr_adx = pd.concat([
        high - low,
        (high - low_lag).abs(),
        (low - low_lag).abs(),
    ], axis=1).max(axis=1)
    alpha = 1 / 14
    s_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    s_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    s_tr = tr_adx.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * s_plus_dm / s_tr.replace(0, np.nan)
    minus_di = 100 * s_minus_dm / s_tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    data["adx"] = dx.rolling(14).mean()
    data["plus_di"] = plus_di
    data["minus_di"] = minus_di

    # --- OBV (normalized as ratio to 20-bar average) ---
    obv = (volume * np.sign(close.diff())).fillna(0).cumsum()
    obv_ma = obv.rolling(20).mean().replace(0, np.nan)
    data["obv_ratio"] = obv / obv_ma

    # --- MFI (Money Flow Index, 14-period) ---
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    data["mfi"] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

    # --- Lag features ---
    data["ret_1_lag1"] = data["ret_1"].shift(1)
    data["ret_1_lag2"] = data["ret_1"].shift(2)
    data["ret_1_lag3"] = data["ret_1"].shift(3)
    data["rsi_lag1"] = data["rsi"].shift(1)
    data["vol_21_lag1"] = data["vol_21"].shift(1)

    # --- Hurst exponent (50-bar window, regime indicator) ---
    hurst = close.rolling(50, min_periods=20).apply(
        lambda x: _hurst_exponent(x), raw=True
    )
    data["hurst"] = hurst.fillna(0.5)

    # --- Choppiness index (14-bar) ---
    tr_ch = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr_ch.rolling(14).sum()
    high_max = high.rolling(14).max()
    low_min = low.rolling(14).min()
    h_l_range = (high_max - low_min).replace(0, np.nan)
    data["choppiness"] = (100 * np.log10(atr_sum / h_l_range) / np.log10(14)).fillna(50)

    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names from a DataFrame that has had compute_features() applied."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]
