"""Shared feature computation for ML training and inference."""

import warnings

import numpy as np
import pandas as pd

from backend.cpp_ext import (
    hurst_exponent,
    rolling_hurst as cpp_rolling_hurst,
    rsi as cpp_rsi,
    macd as cpp_macd,
    bollinger as cpp_bollinger,
    atr as cpp_atr,
    mfi as cpp_mfi,
    adx as cpp_adx,
)

EXCLUDED_COLUMNS = {"open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol", "target"}

_EPS = 1e-12


def safe_divide(num, denom):
    """Divide *num* by *denom*, replacing zero / NaN denominators with NaN.

    The caller should call :func:`clean_features` on the full DataFrame
    afterwards to fill remaining NaN values with 0.
    """
    return num / denom.replace(0, np.nan)


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Return ``pct_change`` with ``inf`` replaced by NaN."""
    return series.pct_change(periods).replace([np.inf, -np.inf], np.nan)


def clean_features(df: pd.DataFrame, fill_val: float = 0.0) -> pd.DataFrame:
    """Replace ``inf`` / ``-inf`` with NaN, then fill all NaN with *fill_val*."""
    return df.replace([np.inf, -np.inf], np.nan).fillna(fill_val)


def _hurst_exponent(ts: np.ndarray) -> float:
    """Compute Hurst exponent via rescaled range (R/S) analysis — C++ accelerated."""
    return hurst_exponent(ts)


def compute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator feature columns to the DataFrame.

    Operates on a copy. All features use only past data
    (rolling windows) — no lookahead bias.

    Args:
        data: OHLCV DataFrame with columns: open, high, low, close, volume.
              Index should be datetime-like for day_of_week to work.

    Returns:
        New DataFrame with additional feature columns.
    """
    data = data.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # --- Returns ---
    data["ret_1"] = safe_pct_change(close, 1)
    data["ret_5"] = safe_pct_change(close, 5)
    data["ret_21"] = safe_pct_change(close, 21)

    # --- Volatility ---
    data["vol_5"] = data["ret_1"].rolling(5).std()
    data["vol_21"] = data["ret_1"].rolling(21).std()

    # --- SMA ratios ---
    data["sma_20"] = close.rolling(20).mean()
    data["sma_50"] = close.rolling(50).mean()
    data["close_sma_20"] = safe_divide(close, data["sma_20"])
    data["close_sma_50"] = safe_divide(close, data["sma_50"])

    # --- Volume ratio ---
    data["vol_ma_20"] = volume.rolling(20).mean()
    data["vol_ratio"] = safe_divide(volume, data["vol_ma_20"])

    # --- RSI (14-period) — C++ accelerated ---
    rsi_vals = cpp_rsi(close.values.astype(float), 14)
    rsi_vals = np.where(np.isnan(rsi_vals), 50.0, rsi_vals)
    data["rsi"] = pd.Series(rsi_vals, index=data.index)

    # --- MACD — C++ accelerated ---
    macd_line, macd_signal, macd_hist = cpp_macd(close.values.astype(float), 12, 26, 9)
    data["macd"] = pd.Series(macd_line, index=data.index)
    data["macd_signal"] = pd.Series(macd_signal, index=data.index)
    data["macd_hist"] = pd.Series(macd_hist, index=data.index)

    # --- True Range (computed once, reused for ATR, Choppiness, ADX) ---
    high_low = high - low
    hc_abs = (high - close.shift(1)).abs()
    lc_abs = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, hc_abs, lc_abs], axis=1).max(axis=1)

    # --- ATR (14-period) — C++ accelerated ---
    atr_vals = cpp_atr(high.values.astype(float), low.values.astype(float),
                       close.values.astype(float), 14)
    data["atr"] = pd.Series(atr_vals, index=data.index)

    # --- Price position in 20-bar range ---
    data["highest_20"] = high.rolling(20).max()
    data["lowest_20"] = low.rolling(20).min()
    data["price_position"] = safe_divide(
        close - data["lowest_20"],
        data["highest_20"] - data["lowest_20"],
    )

    # --- Day of week (0=Monday, 4=Friday) ---
    if hasattr(data.index, "dtype") and data.index.dtype.kind == "M":
        data["day_of_week"] = data.index.dayofweek
    else:
        warnings.warn("Non-datetime index detected; day_of_week set to 0 (Monday)")
        data["day_of_week"] = 0

    # --- Volatility ratio (short / long) ---
    data["vol_ratio_5_21"] = safe_divide(data["vol_5"], data["vol_21"])

    # --- Momentum over multiple windows ---
    data["mom_10"] = safe_pct_change(close, 10)
    data["mom_20"] = safe_pct_change(close, 20)
    data["mom_60"] = safe_pct_change(close, 60)

    # ── New features ────────────────────────────────────────────────────────

    # --- Bollinger Bands (20,2) — C++ accelerated ---
    bb_width, bb_pct_b = cpp_bollinger(close.values.astype(float), 20, 2.0)
    data["bb_width"] = pd.Series(bb_width, index=data.index)
    data["bb_pct_b"] = pd.Series(bb_pct_b, index=data.index)

    # --- ADX (Average Directional Index, 14-period) — C++ accelerated ---
    adx_vals, plus_di_vals, minus_di_vals = cpp_adx(
        high.values.astype(float), low.values.astype(float),
        close.values.astype(float), 14,
    )
    data["adx"] = pd.Series(adx_vals, index=data.index)
    data["plus_di"] = pd.Series(plus_di_vals, index=data.index)
    data["minus_di"] = pd.Series(minus_di_vals, index=data.index)

    # --- OBV (normalized as ratio to 20-bar average) using numpy ---
    close_diff = close.diff().values
    close_diff_sign = np.where(np.isnan(close_diff), 0.0, np.sign(close_diff))
    obv_vals = np.cumsum(volume.values * close_diff_sign)
    obv = pd.Series(obv_vals, index=data.index)
    obv_ma = obv.rolling(20).mean().replace(0, np.nan)
    data["obv_ratio"] = safe_divide(obv, obv_ma)

    # --- MFI (Money Flow Index, 14-period) — C++ accelerated ---
    mfi_vals = cpp_mfi(high.values.astype(float), low.values.astype(float),
                        close.values.astype(float), volume.values.astype(float), 14)
    data["mfi"] = pd.Series(mfi_vals, index=data.index)

    # --- Lag features ---
    data["ret_1_lag1"] = data["ret_1"].shift(1)
    data["ret_1_lag2"] = data["ret_1"].shift(2)
    data["ret_1_lag3"] = data["ret_1"].shift(3)
    data["rsi_lag1"] = data["rsi"].shift(1)
    data["vol_21_lag1"] = data["vol_21"].shift(1)

    # --- Hurst exponent (50-bar window, regime indicator) — C++ accelerated ---
    hurst_vals = cpp_rolling_hurst(close.values.astype(float), 50)
    data["hurst"] = pd.Series(hurst_vals, index=data.index).fillna(0.5)

    # --- Choppiness index (14-bar, reuses TR from above) ---
    atr_sum = tr.rolling(14).sum()
    high_max = high.rolling(14).max()
    low_min = low.rolling(14).min()
    h_l_range = (high_max - low_min).replace(0, np.nan)
    atr_sum_safe = atr_sum.replace(0, np.nan)
    data["choppiness"] = 100 * np.log10(safe_divide(atr_sum_safe, h_l_range)) / np.log10(14)

    data = clean_features(data)
    return data


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names from a DataFrame that has had compute_features() applied."""
    return [c for c in df.columns if c not in EXCLUDED_COLUMNS]


def merge_context_features(
    df: pd.DataFrame,
    context_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute features on context symbols and left-merge onto *df* by timestamp.

    Context feature columns are prefixed with ``{symbol}_`` to avoid name
    collisions.  Missing context dates are forward-filled (last observation
    carried forward) so the model always has context for every bar.

    Args:
        df: Main symbol DataFrame (after ``compute_features()``).
        context_data: Mapping ``{symbol: OHLCV DataFrame}``.

    Returns:
        New DataFrame with context feature columns added.
    """
    df = df.copy()
    for ctx_name, ctx_df in context_data.items():
        # Normalize timezone so join doesn't fail on tz-aware vs tz-naive mismatch
        ctx_feat = compute_features(ctx_df)
        if hasattr(ctx_feat.index, "tz") and ctx_feat.index.tz is not None:
            ctx_feat.index = ctx_feat.index.tz_localize(None)
        ctx_feat = ctx_feat[get_feature_columns(ctx_feat)]
        ctx_feat = ctx_feat.add_prefix(f"{ctx_name}_")
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.join(ctx_feat, how="left")
        for col in ctx_feat.columns:
            df[col] = df[col].ffill()
    return df
