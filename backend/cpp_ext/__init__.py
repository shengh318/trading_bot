"""C++ acceleration module with Python fallback.

Provides accelerated versions of:
  - hurst_exponent        R/S analysis
  - rolling_ols           sliding-window OLS (prefix-sum)
  - kalman_fit            RLS Kalman filter
  - compute_time_to_mean  z-score zero-crossing
  - triple_barrier_label  de Prado triple barrier
  - rolling_slope         sliding-window slope (O(1) update)

  NEW:
  - estimate_ols          single OLS regression (nan-safe)
  - rolling_hurst         sliding-window Hurst exponent
  - rolling_half_life     sliding-window half-life
  - adf_test              ADF unit root test
  - eg_coint_test         Engle-Granger cointegration test
  - bai_perron_breaks     Bai-Perron breakpoint detection
  - cusum_breaks          CUSUM break detection
  - rsi                   RSI computation
  - macd                  MACD line/signal/histogram
  - bollinger             Bollinger Band width and %b
  - atr                   Average True Range
  - mfi                   Money Flow Index
  - adx                   ADX, +DI, -DI

If the compiled C++ module is not available, pure-Python fallbacks are used.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

try:
    from .cpp_accel import (
        hurst_exponent as _c_hurst,
        rolling_ols as _c_rolling_ols,
        kalman_fit as _c_kalman,
        compute_time_to_mean as _c_time_to_mean,
        triple_barrier_label as _c_triple_barrier,
        rolling_slope as _c_rolling_slope,
        _cpp_estimate_ols as _c_estimate_ols,
        _cpp_rolling_hurst as _c_rolling_hurst,
        _cpp_rolling_half_life as _c_rolling_half_life,
        _cpp_adf_test as _c_adf_test,
        _cpp_eg_coint_test as _c_eg_coint_test,
        _cpp_bai_perron_breaks as _c_bai_perron_breaks,
        _cpp_cusum_breaks as _c_cusum_breaks,
        _cpp_rsi as _c_rsi,
        _cpp_macd as _c_macd,
        _cpp_bollinger as _c_bollinger,
        _cpp_atr as _c_atr,
        _cpp_mfi as _c_mfi,
        _cpp_adx as _c_adx,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


# ─── Python (numpy) fallbacks ──────────────────────────────────────

def _py_hurst_exponent(ts: np.ndarray) -> float:
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
        std_vals = reshaped.std(axis=1, ddof=0)
        std_vals = np.where(std_vals > 0, std_vals, np.nan)
        rs = (cumsum.max(axis=1) - cumsum.min(axis=1)) / std_vals
        rs = rs[~np.isnan(rs)]
        if len(rs) > 0:
            tau.append(float(rs.mean()))
    if len(tau) < 3:
        return 0.5
    lags_used = list(lags[: len(tau)])
    if len(lags_used) < 3:
        return 0.5
    reg = np.polyfit(np.log(lags_used), np.log(tau), 1)
    return max(0.0, min(1.0, float(reg[0])))


def _py_rolling_ols(
    a: np.ndarray, b: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(a)
    out_len = n - window + 1
    betas = np.empty(out_len, dtype=float)
    ints = np.empty(out_len, dtype=float)
    for i in range(out_len):
        wa = a[i : i + window]
        wb = b[i : i + window]
        A = np.column_stack([np.ones_like(wb), wb])
        coeffs, _, _, _ = np.linalg.lstsq(A, wa, rcond=None)
        ints[i] = coeffs[0]
        betas[i] = coeffs[1]
    return betas, ints


def _py_kalman_fit(
    y: np.ndarray, x: np.ndarray, lambda_: float = 0.99, delta: float = 1e-4
) -> np.ndarray:
    n = len(y)
    if n < 10:
        return np.full(n, np.nan)
    y = y.astype(np.float64)
    x = x.astype(np.float64)
    theta = np.array([0.0, 0.0])
    P = np.eye(2) * 100.0
    betas = np.zeros(n)
    for t in range(n):
        phi = np.array([1.0, x[t]])
        y_pred = theta @ phi
        innovation = y[t] - y_pred
        g = P @ phi / (lambda_ + phi @ P @ phi)
        theta = theta + g * innovation
        P = (P - np.outer(g, phi @ P)) / lambda_
        P += np.eye(2) * delta
        betas[t] = float(theta[1])
    return betas


def _py_compute_time_to_mean(
    zscore: np.ndarray, max_horizon: int = 60
) -> np.ndarray:
    n = len(zscore)
    result = np.full(n, float(max_horizon), dtype=float)
    for i in range(n):
        if not np.isfinite(zscore[i]):
            continue
        end = min(i + max_horizon + 1, n)
        for j in range(i + 1, end):
            if np.isfinite(zscore[j]) and zscore[i] * zscore[j] <= 0.0:
                result[i] = float(j - i)
                break
    return result


def _py_triple_barrier_label(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    pct: float = 0.02, max_bars: int = 10,
) -> np.ndarray:
    n = len(close)
    labels = np.zeros(n, dtype=int)
    for i in range(n - 1):
        entry = close[i]
        tp = entry * (1 + pct)
        sl = entry * (1 - pct)
        end = min(i + max_bars + 1, n)
        for j in range(i + 1, end):
            if high[j] >= tp:
                labels[i] = 1
                break
            if low[j] <= sl:
                labels[i] = 0
                break
    return labels


def _py_rolling_slope(vals: np.ndarray, window: int) -> np.ndarray:
    n = len(vals)
    slopes = np.full(n, np.nan, dtype=float)
    x = np.arange(window, dtype=float)
    for i in range(window, n):
        slopes[i] = np.polyfit(x, vals[i - window : i], 1)[0]
    return slopes


# ─── NEW Python fallbacks ──────────────────────────────────────────

def _py_estimate_ols(a: np.ndarray, b: np.ndarray, add_const: bool = True) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    a_clean = a[mask]
    b_clean = b[mask]
    if add_const:
        b_with_const = np.column_stack([np.ones_like(b_clean), b_clean])
        beta, _, _, _ = np.linalg.lstsq(b_with_const, a_clean, rcond=None)
        return float(beta[1])
    beta, _, _, _ = np.linalg.lstsq(b_clean.reshape(-1, 1), a_clean, rcond=None)
    return float(beta[0])


def _py_rolling_hurst(ts: np.ndarray, window: int) -> np.ndarray:
    n = len(ts)
    result = np.full(n, np.nan, dtype=float)
    if window < 20 or n < window:
        return result
    for i in range(window - 1, n):
        chunk = ts[i - window + 1 : i + 1]
        chunk = chunk[pd.notna(chunk)]  # type: ignore[name-defined]  # noqa: F821
        if len(chunk) >= 30:
            result[i] = _py_hurst_exponent(chunk)
    return result


def _py_rolling_half_life(spread: np.ndarray, window: int) -> np.ndarray:
    n = len(spread)
    result = np.full(n, np.inf, dtype=float)
    if window < 6 or n < window:
        return result
    for i in range(window, n + 1):
        chunk = spread[i - window : i]
        s = np.asarray(chunk, dtype=float)
        lagged = s[:-1]
        delta = s[1:] - s[:-1]
        mask = np.isfinite(lagged) & np.isfinite(delta)
        valid = mask.sum()
        if valid < 5:
            continue
        lagged_c = lagged[mask]
        delta_c = delta[mask]
        A = np.column_stack([np.ones_like(lagged_c), lagged_c])
        try:
            theta, _, _, _ = np.linalg.lstsq(A, delta_c, rcond=None)
        except np.linalg.LinAlgError:
            continue
        theta_val = theta[1]
        if theta_val >= 0:
            continue
        hl = -math.log(2) / theta_val
        if np.isfinite(hl) and hl < 1e6:
            result[i - 1] = hl
    return result


def _py_adf_test(series: np.ndarray, maxlag: int = 1, autolag: bool = True) -> tuple[float, float]:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series, maxlag=maxlag, autolag="AIC" if autolag else None)
    return float(result[0]), float(result[1])


def _py_eg_coint_test(
    a: np.ndarray, b: np.ndarray, maxlag: int = 1, autolag: bool = True
) -> tuple[float, float, float]:
    from statsmodels.tsa.stattools import coint
    hr = _py_estimate_ols(a, b, add_const=True)
    _, p_val, _ = coint(a, b, maxlag=maxlag, autolag="AIC" if autolag else None)
    return float(p_val), float(hr)


def _py_bai_perron_breaks(
    y: np.ndarray, max_breaks: int = 5, min_segment: int = 30
) -> np.ndarray:
    n = len(y)
    breaks: list[int] = []
    segments = [(0, n)]

    def find_break(start: int, end: int) -> tuple[int, float]:
        seg_len = end - start
        if seg_len < 2 * min_segment:
            return -1, 0.0
        total_sum = np.sum(y[start:end])
        total_sum_sq = np.sum(y[start:end] ** 2)
        seg_n = float(seg_len)
        pooled_mean = total_sum / seg_n
        rss_pooled = total_sum_sq - seg_n * pooled_mean**2
        if rss_pooled <= 0:
            return -1, 0.0
        pooled_bic = seg_n * np.log(rss_pooled / seg_n) + 2 * np.log(seg_n)

        best_bic = np.inf
        best_pos = -1
        cumsum = np.cumsum(y[start:end])
        cumsumsq = np.cumsum(y[start:end] ** 2)

        for pos in range(min_segment, seg_len - min_segment):
            n1, n2 = float(pos), float(seg_len - pos)
            sum1 = cumsum[pos - 1]
            sum2 = total_sum - sum1
            sum_sq1 = cumsumsq[pos - 1]
            sum_sq2 = total_sum_sq - sum_sq1
            mean1 = sum1 / n1
            mean2 = sum2 / n2
            rss1 = sum_sq1 - n1 * mean1**2
            rss2 = sum_sq2 - n2 * mean2**2
            rss = rss1 + rss2
            if rss <= 0:
                continue
            bic = seg_n * np.log(rss / seg_n) + 4 * np.log(seg_n)
            if bic < best_bic:
                best_bic = bic
                best_pos = start + pos
        improvement = pooled_bic - best_bic
        return best_pos, improvement

    while len(breaks) < max_breaks:
        best_seg_idx = -1
        best_break = -1
        best_improvement = 0.0
        for s_idx, (seg_start, seg_end) in enumerate(segments):
            pos, improvement = find_break(seg_start, seg_end)
            if pos >= 0 and improvement > best_improvement:
                best_improvement = improvement
                best_break = pos
                best_seg_idx = s_idx
        if best_break < 0 or best_improvement <= 0:
            break
        breaks.append(best_break)
        seg_start, seg_end = segments.pop(best_seg_idx)
        segments.append((seg_start, best_break))
        segments.append((best_break, seg_end))
        segments.sort()

    return np.array(breaks, dtype=int)


def _py_cusum_breaks(
    y: np.ndarray, confidence: float = 0.95
) -> tuple[bool, np.ndarray, np.ndarray]:
    n = len(y)
    if n < 30:
        return False, np.array([], dtype=int), np.full(n - 1, np.nan)

    rr = np.empty(n - 1)
    cum_sum = 0.0
    for t in range(1, n):
        mean_t = cum_sum / t
        cum_sum += y[t]
        err = y[t] - mean_t
        denom = np.sqrt(1.0 + 1.0 / (t + 1))
        rr[t - 1] = err / denom

    sigma = np.std(rr)
    if sigma == 0:
        return False, np.array([], dtype=int), rr

    std_rr = rr / sigma
    cusum = np.cumsum(std_rr)
    z = {0.90: 0.850, 0.95: 0.948, 0.99: 1.143}.get(confidence, 0.948)
    bound = z * np.sqrt(n - 1)
    break_indices = np.where(np.abs(cusum) > bound)[0]
    has_break = len(break_indices) > 0

    # Deduplicate consecutive indices
    if has_break and len(break_indices) > 1:
        deduped = [int(break_indices[0])]
        for idx in break_indices[1:]:
            if idx != deduped[-1] + 1:
                deduped.append(int(idx))
        break_indices = np.array(deduped, dtype=int)

    return has_break, break_indices, cusum


def _py_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    alpha = 1.0 / period
    for i in range(period + 1, len(close)):
        avg_gain[i] = (1.0 - alpha) * avg_gain[i - 1] + alpha * gain[i - 1]
        avg_loss[i] = (1.0 - alpha) * avg_loss[i - 1] + alpha * loss[i - 1]
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.where(avg_gain > 0, 1e12, 1.0))
    rsi_vals = 100.0 - 100.0 / (1.0 + rs)
    rsi_vals[:period] = np.nan
    return rsi_vals


def _py_macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(close) < slow:
        n = len(close)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    ema_fast = _py_ema(close, fast)
    ema_slow = _py_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _py_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _py_ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    result = np.full(len(x), np.nan)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = (x[i] - result[i - 1]) * alpha + result[i - 1]
    return result


def _py_bollinger(
    close: np.ndarray, period: int = 20, n_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    bb_width = np.full(n, np.nan)
    bb_pct_b = np.full(n, np.nan)
    if n < period:
        return bb_width, bb_pct_b
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        mean = np.nanmean(window)
        valid = window[np.isfinite(window)]
        if len(valid) < period // 2:
            continue
        std_val = np.nanstd(window, ddof=0)
        upper = mean + n_std * std_val
        lower = mean - n_std * std_val
        if mean != 0:
            bb_width[i] = (upper - lower) / mean
        if upper != lower:
            bb_pct_b[i] = (close[i] - lower) / (upper - lower)
    return bb_width, bb_pct_b


def _py_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    n = len(high)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    result[period] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period + 1, n):
        result[i] = (1.0 - alpha) * result[i - 1] + alpha * tr[i - 1]
    return result


def _py_mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray, period: int = 14,
) -> np.ndarray:
    n = len(close)
    result = np.full(n, np.nan)
    if n < period + 1:
        return result
    typical = (high + low + close) / 3.0
    raw_mf = typical * volume
    pos_mf = np.full(n, np.nan)
    neg_mf = np.full(n, np.nan)
    for i in range(1, n):
        if typical[i] > typical[i - 1]:
            pos_mf[i] = raw_mf[i]
            neg_mf[i] = 0.0
        elif typical[i] < typical[i - 1]:
            pos_mf[i] = 0.0
            neg_mf[i] = raw_mf[i]
        else:
            pos_mf[i] = 0.0
            neg_mf[i] = 0.0
    for i in range(period, n):
        p_sum = np.nansum(pos_mf[i - period + 1 : i + 1])
        n_sum = np.nansum(neg_mf[i - period + 1 : i + 1])
        mfr = p_sum / n_sum if n_sum > 0 else (1e12 if p_sum > 0 else 1.0)
        result[i] = 100.0 - 100.0 / (1.0 + mfr)
    return result


def _py_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(high)
    adx_vals = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    if n < period + 1:
        return adx_vals, plus_di, minus_di

    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))

    dm_plus = np.where((up > down) & (up > 0), up, 0.0)
    dm_minus = np.where((down > up) & (down > 0), down, 0.0)

    alpha = 1.0 / period
    s_tr = np.full(n, np.nan)
    s_dm_plus = np.full(n, np.nan)
    s_dm_minus = np.full(n, np.nan)

    s_tr[period] = np.mean(tr[:period])
    s_dm_plus[period] = np.mean(dm_plus[:period])
    s_dm_minus[period] = np.mean(dm_minus[:period])

    for i in range(period + 1, n):
        s_tr[i] = (1.0 - alpha) * s_tr[i - 1] + alpha * tr[i - 1]
        s_dm_plus[i] = (1.0 - alpha) * s_dm_plus[i - 1] + alpha * dm_plus[i - 1]
        s_dm_minus[i] = (1.0 - alpha) * s_dm_minus[i - 1] + alpha * dm_minus[i - 1]

    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    for i in range(period, n):
        if s_tr[i] > 0:
            pdi[i] = 100.0 * s_dm_plus[i] / s_tr[i]
            mdi[i] = 100.0 * s_dm_minus[i] / s_tr[i]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        if pdi[i] + mdi[i] > 0:
            dx[i] = 100.0 * np.abs(pdi[i] - mdi[i]) / (pdi[i] + mdi[i])

    adx_vals[2 * period] = np.nanmean(dx[period : 2 * period])
    for i in range(2 * period + 1, n):
        adx_vals[i] = (1.0 - alpha) * adx_vals[i - 1] + alpha * dx[i - 1]

    return adx_vals, pdi, mdi


# ─── Public API — Existing ─────────────────────────────────────────

def hurst_exponent(ts: np.ndarray) -> float:
    if _HAS_CPP:
        return _c_hurst(ts)
    return _py_hurst_exponent(ts)


def rolling_ols(
    a: np.ndarray, b: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    if _HAS_CPP:
        return _c_rolling_ols(a, b, window)
    return _py_rolling_ols(a, b, window)


def kalman_fit(
    y: np.ndarray, x: np.ndarray, lambda_: float = 0.99, delta: float = 1e-4
) -> np.ndarray:
    if _HAS_CPP:
        return _c_kalman(y, x, lambda_, delta)
    return _py_kalman_fit(y, x, lambda_, delta)


def compute_time_to_mean(
    zscore: np.ndarray, max_horizon: int = 60
) -> np.ndarray:
    if _HAS_CPP:
        return _c_time_to_mean(zscore, max_horizon)
    return _py_compute_time_to_mean(zscore, max_horizon)


def triple_barrier_label(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    pct: float = 0.02, max_bars: int = 10,
) -> np.ndarray:
    if _HAS_CPP:
        return _c_triple_barrier(close, high, low, pct, max_bars)
    return _py_triple_barrier_label(close, high, low, pct, max_bars)


def rolling_slope(vals: np.ndarray, window: int) -> np.ndarray:
    if _HAS_CPP:
        return _c_rolling_slope(vals, window)
    return _py_rolling_slope(vals, window)


# ─── Public API — NEW Functions ────────────────────────────────────

def estimate_ols(a: np.ndarray, b: np.ndarray, add_const: bool = True) -> float:
    """Single OLS regression (nan-safe).

    Returns the slope coefficient (beta) from regression a ~ b.
    If add_const=True, includes intercept. NaN values are filtered.
    """
    if _HAS_CPP:
        return _c_estimate_ols(a, b, add_const)
    return _py_estimate_ols(a, b, add_const)


def rolling_hurst(ts: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window Hurst exponent, nan-safe.

    For each position i, computes Hurst on ts[i-window+1 : i+1].
    Windows with <30 non-NaN values yield NaN.
    Input NaN values are filtered within each window.
    """
    if _HAS_CPP:
        return _c_rolling_hurst(ts, window)
    return _py_rolling_hurst(ts, window)


def rolling_half_life(spread: np.ndarray, window: int) -> np.ndarray:
    """Sliding-window half-life via OLS on lagged spread.

    For each window, regresses Δspread_t = α + θ * spread_{t-1}.
    Half-life = -ln(2) / θ.  Returns inf for non-mean-reverting windows.
    """
    if _HAS_CPP:
        return _c_rolling_half_life(spread, window)
    return _py_rolling_half_life(spread, window)


def adf_test(series: np.ndarray, maxlag: int = 1, autolag: bool = True) -> tuple[float, float]:
    """ADF unit root test (constant only).

    Returns (t-statistic, p-value).  If autolag=True, selects lag via AIC
    from 0..maxlag.  Uses MacKinnon 1994 critical values (case 2, n=1).
    """
    if _HAS_CPP:
        return _c_adf_test(series, maxlag, autolag)
    return _py_adf_test(series, maxlag, autolag)


def eg_coint_test(
    a: np.ndarray, b: np.ndarray, maxlag: int = 1, autolag: bool = True
) -> tuple[float, float, float]:
    """Engle-Granger cointegration test.

    Steps:
      1. OLS: a = α + β * b
      2. Spread = a - α - β * b
      3. ADF test on spread (with EG critical values)

    Returns (t-statistic, p-value, hedge_ratio).
    """
    if _HAS_CPP:
        return _c_eg_coint_test(a, b, maxlag, autolag)
    return _py_eg_coint_test(a, b, maxlag, autolag)


def bai_perron_breaks(
    y: np.ndarray, max_breaks: int = 5, min_segment: int = 30
) -> np.ndarray:
    """Bai-Perron sequential breakpoint detection.

    Uses O(1) cumsum per position for BIC computation.
    Returns array of break indices (sorted).
    """
    if _HAS_CPP:
        return _c_bai_perron_breaks(y, max_breaks, min_segment)
    return _py_bai_perron_breaks(y, max_breaks, min_segment)


def cusum_breaks(
    y: np.ndarray, confidence: float = 0.95
) -> tuple[bool, np.ndarray, np.ndarray]:
    """CUSUM break detection via recursive residuals.

    Returns (has_break, break_indices, cusum_series).
    """
    if _HAS_CPP:
        return _c_cusum_breaks(y, confidence)
    return _py_cusum_breaks(y, confidence)


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Relative Strength Index) with Wilder smoothing."""
    if _HAS_CPP:
        return _c_rsi(close, period)
    return _py_rsi(close, period)


def macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    if _HAS_CPP:
        return _c_macd(close, fast, slow, signal)
    return _py_macd(close, fast, slow, signal)


def bollinger(
    close: np.ndarray, period: int = 20, n_std: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger Band width (upper-lower)/mid and %b."""
    if _HAS_CPP:
        return _c_bollinger(close, period, n_std)
    return _py_bollinger(close, period, n_std)


def atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    if _HAS_CPP:
        return _c_atr(high, low, close, period)
    return _py_atr(high, low, close, period)


def mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    volume: np.ndarray, period: int = 14,
) -> np.ndarray:
    """Money Flow Index."""
    if _HAS_CPP:
        return _c_mfi(high, low, close, volume, period)
    return _py_mfi(high, low, close, volume, period)


def adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, -DI (Average Directional Index)."""
    if _HAS_CPP:
        return _c_adx(high, low, close, period)
    return _py_adx(high, low, close, period)
