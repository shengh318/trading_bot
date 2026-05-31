# C++ Acceleration Module

19 pybind11 kernels with pure-Python numpy fallbacks. Each function works whether or not the C++ library compiles.

## Build

### Windows (VS 2022)
```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cd backend\cpp_ext && ..\..\.venv\Scripts\python setup.py build_ext --inplace"
```

### macOS / Linux
```bash
cd backend/cpp_ext && python setup.py build_ext --inplace
```

## Kernel Reference

All functions in `backend.cpp_ext`, auto-detect C++ build.

### Phase 1 — Core Math

| Function | Signature | Description |
|----------|-----------|-------------|
| `hurst_exponent(ts)` → float | `ndarray` | R/S analysis |
| `rolling_ols(a, b, window)` → (betas, intercepts) | `(ndarray, ndarray, int)` | Prefix-sum O(1) |
| `kalman_fit(y, x, lambda_, delta)` → ndarray | RLS Kalman filter |
| `compute_time_to_mean(zscore, max_horizon)` → ndarray | Z-score zero-crossing |
| `triple_barrier_label(close, high, low, pct, max_bars)` → ndarray | De Prado triple barrier |
| `rolling_slope(vals, window)` → ndarray | O(1) sliding slope |

### Phase 2 — Statistical Tests

| Function | Signature | Description |
|----------|-----------|-------------|
| `estimate_ols(a, b, add_const)` → float | NaN-safe OLS slope |
| `rolling_hurst(ts, window)` → ndarray | Sliding Hurst |
| `rolling_half_life(spread, window)` → ndarray | Sliding half-life |
| `adf_test(series, maxlag, autolag)` → (tstat, pval) | ADF + MacKinnon CVs |
| `eg_coint_test(a, b, maxlag, autolag)` → (tstat, pval, hr) | Engle-Granger |

### Phase 3 — Feature Computation

| Function | Signature | Description |
|----------|-----------|-------------|
| `rsi(close, period)` → ndarray | SMA-based RSI |
| `macd(close, fast, slow, signal)` → (macd, signal, hist) | EMA-based MACD |
| `bollinger(close, period, n_std)` → (width, %b) | Prefix-sum BB |
| `atr(high, low, close, period)` → ndarray | Wilder smoothing |
| `mfi(high, low, close, volume, period)` → ndarray | Money Flow Index |
| `adx(high, low, close, period)` → (adx, +di, -di) | ADX |
| `bai_perron_breaks(y, max_breaks, min_segment)` → ndarray | Sequential BIM |
| `cusum_breaks(y, confidence)` → (has_break, indices, cusum) | Recursive residuals |

## Python Examples

```python
import numpy as np
from backend.cpp_ext import hurst_exponent, adf_test, eg_coint_test, rsi

hurst_exponent(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))  # ~ 1.0

adf_test(np.random.randn(100), maxlag=5)  # (tstat, pval)

a = np.cumsum(np.random.randn(200)) + 100
b = a * 0.5 + np.random.randn(200) * 0.5
eg_coint_test(a, b)  # (tstat, pval, hedge_ratio)

close = np.cumsum(np.random.randn(100)) + 100
rsi(close)  # ndarray of RSI values
```

## Usage Map

| Module | Kernels Used |
|--------|-------------|
| `stats_arb.spread` | `hurst_exponent`, `estimate_ols`, `adf_test` |
| `stats_arb.regime` | `hurst_exponent`, `rolling_hurst`, `cusum_breaks`, `bai_perron_breaks` |
| `stats_arb.cointegration` | `eg_coint_test` |
| `stats_arb.hedge_ratio` | `estimate_ols`, `rolling_ols`, `kalman_fit` |
| `stats_arb.correlation` | `rolling_slope` |
| `stats_arb.walk_forward` | `estimate_ols`, `adf_test`, `eg_coint_test` |
| `stats_arb.ml_models` | `compute_time_to_mean`, `rolling_half_life` |
| `stats_arb.strategy` | `hurst_exponent`, `rolling_hurst` |
| `ml.features` | `rsi`, `macd`, `bollinger`, `atr` |
| `ml.train` | `triple_barrier_label` |
| `strategies.corr_coint_strat` | `hurst_exponent`, `rolling_ols`, `rolling_hurst` |
