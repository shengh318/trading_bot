"""Shared feature computation for ML training and inference."""

import warnings

import numpy as np
import pandas as pd

EXCLUDED_COLUMNS = {"open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol", "target"}

_EPS = 1e-12


def safe_divide(num, denom):
    """Divide *num* by *denom*, replacing zero / NaN denominators with NaN.

    The caller should call :func:`clean_features` on the full DataFrame
    afterwards to fill remaining NaN values with 0.
    """
    denom = denom.replace(0, np.nan)
    return num / denom


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Return ``pct_change`` with ``inf`` replaced by NaN."""
    return series.pct_change(periods).replace([np.inf, -np.inf], np.nan)


def clean_features(df: pd.DataFrame, fill_val: float = 0.0) -> pd.DataFrame:
    """Replace ``inf`` / ``-inf`` with NaN, then fill all NaN with *fill_val*."""
    return df.replace([np.inf, -np.inf], np.nan).fillna(fill_val)


def _hurst_exponent(ts: np.ndarray) -> float:
    """Compute Hurst exponent via multi-lag rescaled range (R/S) regression.

    Regresses log(E[R/S]) vs log(lag) across multiple lags.
    H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending.
    """
    if len(ts) < 20:
        return 0.5
    ts = np.asarray(ts, dtype=float)
    if np.nanmin(ts) == np.nanmax(ts):
        return 0.5
    max_lag = len(ts) // 2
    if max_lag < 3:
        return 0.5
    lags = range(2, max_lag)
    tau: list[float] = []
    for lag in lags:
        chunks = len(ts) // lag
        if chunks < 1:
            continue
        trimmed = ts[: chunks * lag]
        reshaped = trimmed.reshape((chunks, lag))
        mean_adj = reshaped - reshaped.mean(axis=1, keepdims=True)
        cumsum = mean_adj.cumsum(axis=1)
        rs = (cumsum.max(axis=1) - cumsum.min(axis=1)) / reshaped.std(axis=1, ddof=0)
        rs = rs[~np.isnan(rs)]
        if len(rs) > 0:
            tau.append(float(rs.mean()))
    if len(tau) < 3:
        return 0.5
    lags_used = list(lags[: len(tau)])
    if len(lags_used) < 3:
        return 0.5
    reg = np.polyfit(np.log(lags_used), np.log(tau), 1)
    h = float(reg[0])
    return max(0.0, min(1.0, h))


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

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    # When avg_loss = 0 and avg_gain > 0, RS = inf → RSI = 100 (correct).
    # When both = 0 (flat prices), RS = NaN → fill with 1 → RSI = 50.
    rs = avg_gain / avg_loss
    rs = rs.fillna(1.0)
    data["rsi"] = 100 - (100 / (1 + rs))

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

    # --- Bollinger Bands (20,2) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    data["bb_width"] = safe_divide(bb_upper - bb_lower, bb_mid)
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    data["bb_pct_b"] = safe_divide(close - bb_lower, bb_range)

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
    plus_di = 100 * safe_divide(s_plus_dm, s_tr)
    minus_di = 100 * safe_divide(s_minus_dm, s_tr)
    dx = 100 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    data["adx"] = dx.ewm(alpha=alpha, adjust=False).mean()
    data["plus_di"] = plus_di
    data["minus_di"] = minus_di

    # --- OBV (normalized as ratio to 20-bar average) ---
    obv = (volume * np.sign(close.diff())).fillna(0).cumsum()
    obv_ma = obv.rolling(20).mean().replace(0, np.nan)
    data["obv_ratio"] = safe_divide(obv, obv_ma)

    # --- MFI (Money Flow Index, 14-period) ---
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    # When neg_mf = 0 and pos_mf > 0, MFR = inf → MFI = 100 (correct).
    # When both = 0 (flat prices), MFR = NaN → fill with 1 → MFI = 50.
    mfr = pos_mf / neg_mf
    mfr = mfr.fillna(1.0)
    data["mfi"] = 100 - (100 / (1 + mfr))

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
