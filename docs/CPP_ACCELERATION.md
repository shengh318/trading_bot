# C++ Acceleration Module

19 computationally-intensive kernels implemented as a pybind11
extension in `backend/cpp_ext/`.  Each function has a pure-Python
numpy fallback — the module works whether or not the C++ library
compiles.

## Build

### Windows (Visual Studio 2022)

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd backend\cpp_ext && ..\..\.venv\Scripts\python setup.py build_ext --inplace"
```

### macOS / Linux

```bash
cd backend/cpp_ext && python setup.py build_ext --inplace
```

## Kernel Reference

All functions are in `backend.cpp_ext` and auto-detect the C++ build.

### Phase 1 — Core Math (6 kernels)

| Function | Signature | Description |
|----------|-----------|-------------|
| `hurst_exponent(ts)` | `ndarray -> float` | R/S analysis Hurst exponent |
| `rolling_ols(a, b, window)` | `(ndarray, ndarray, int) -> (betas, intercepts)` | Sliding OLS via prefix-sum |
| `kalman_fit(y, x, lambda_, delta)` | `(ndarray, ndarray, float, float) -> ndarray` | RLS Kalman filter |
| `compute_time_to_mean(zscore, max_horizon)` | `(ndarray, int) -> ndarray` | Z-score zero-crossing distance |
| `triple_barrier_label(close, high, low, pct, max_bars)` | `(ndarray, ndarray, ndarray, float, int) -> ndarray` | De Prado triple barrier |
| `rolling_slope(vals, window)` | `(ndarray, int) -> ndarray` | O(1) sliding-window slope |

### Phase 2 — Statistical Tests (5 kernels)

| Function | Signature | Description |
|----------|-----------|-------------|
| `estimate_ols(a, b, add_const)` | `(ndarray, ndarray, bool) -> float` | NaN-safe OLS slope |
| `rolling_hurst(ts, window)` | `(ndarray, int) -> ndarray` | Sliding-window Hurst |
| `rolling_half_life(spread, window)` | `(ndarray, int) -> ndarray` | Sliding half-life via OLS |
| `adf_test(series, maxlag, autolag)` | `(ndarray, int, bool) -> (tstat, pval)` | ADF test with MacKinnon CVs |
| `eg_coint_test(a, b, maxlag, autolag)` | `(ndarray, ndarray, int, bool) -> (tstat, pval, hedge_ratio)` | Engle-Granger cointegration |

### Phase 3 — Feature Computation (8 kernels)

| Function | Signature | Description |
|----------|-----------|-------------|
| `rsi(close, period)` | `(ndarray, int) -> ndarray` | RSI with Wilder smoothing |
| `macd(close, fast, slow, signal)` | `(ndarray, int, int, int) -> (macd, signal, hist)` | MACD line/signal/histogram |
| `bollinger(close, period, n_std)` | `(ndarray, int, float) -> (width, %b)` | Bollinger Band width and %b |
| `atr(high, low, close, period)` | `(ndarray, ndarray, ndarray, int) -> ndarray` | Average True Range |
| `mfi(high, low, close, volume, period)` | `(ndarray, ndarray, ndarray, ndarray, int) -> ndarray` | Money Flow Index |
| `adx(high, low, close, period)` | `(ndarray, ndarray, ndarray, int) -> (adx, +di, -di)` | Average Directional Index |
| `bai_perron_breaks(y, max_breaks, min_segment)` | `(ndarray, int, int) -> ndarray` | Bai-Perron structural breaks |
| `cusum_breaks(y, confidence)` | `(ndarray, float) -> (has_break, indices, cusum)` | CUSUM break detection |

## Python Examples

```python
import numpy as np
from backend.cpp_ext import (
    hurst_exponent, adf_test, eg_coint_test,
    rsi, macd, bollinger, atr, mfi, adx,
)

# Hurst exponent
hurst_exponent(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
# ~ 1.0 (trending)

# ADF test
adf_test(np.random.randn(100), maxlag=5)
# (tstat, pval)

# EG cointegration
a = np.cumsum(np.random.randn(200)) + 100
b = a * 0.5 + np.random.randn(200) * 0.5
eg_coint_test(a, b)
# (tstat, pval, hedge_ratio)

# RSI
close = np.cumsum(np.random.randn(100)) + 100
rsi(close)
```

## Who Uses What

| Module | Kernels Used |
|--------|-------------|
| `backend.stats_arb.spread` | `hurst_exponent`, `estimate_ols`, `adf_test` |
| `backend.stats_arb.regime` | `hurst_exponent`, `rolling_hurst`, `cusum_breaks`, `bai_perron_breaks` |
| `backend.stats_arb.cointegration` | `eg_coint_test` |
| `backend.stats_arb.hedge_ratio` | `estimate_ols`, `rolling_ols`, `kalman_fit` |
| `backend.stats_arb.correlation` | `rolling_slope` |
| `backend.stats_arb.walk_forward` | `estimate_ols`, `adf_test`, `eg_coint_test` |
| `backend.stats_arb.ml_models` | `compute_time_to_mean`, `rolling_half_life` |
| `backend.stats_arb.strategy` | `hurst_exponent`, `rolling_hurst` |
| `backend.ml.features` | `rsi`, `macd`, `bollinger`, `atr` |
| `backend.ml.train` | `triple_barrier_label` |
| `backend.strategies.corr_coint_strat` | `hurst_exponent`, `rolling_ols`, `rolling_hurst` |
