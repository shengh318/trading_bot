# Time-Series Momentum (TSMOM) — Research & Backtesting Framework

## Overview

The TSMOM framework is a 13-phase research and backtesting system for
time-series momentum strategies.  It supports multi-asset portfolios,
volatility-adjusted position sizing, trend filtering, walk-forward
validation, benchmarking, regime analysis, parameter grid search, and
an ML extension.

## 13 Research Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | `TSMOMDataLoader` | yfinance download + parquet caching, multi-symbol alignment |
| 2 | `TSMOMFeatures` | Momentum (21/63/126/252d), vol (20/60d), ATR, drawdown, Sharpe, Sortino |
| 3 | `TSMOMTrendFilter` | Price>MA200, MA50>MA200, breakout, volatility-adjusted filters |
| 4 | `TSMOMSignalGenerator` | Binary / weighted / vol-adjusted momentum signals |
| 5 | `TSMOMPositionSizer` | Equal-weight, vol-target, risk-parity sizing |
| 6 | `TSMOMPortfolioConstruction` | Top-N monthly rebalance, long-only or long/short |
| 7 | `TSMOMWalkForward` | Expanding/rolling window walk-forward with no lookahead |
| 8 | `TSMOMBacktestEngine` | Multi-asset backtest with t-cost, slippage, full metrics |
| 9 | `TSMOMBenchmark` | Alpha, beta, IR, tracking error vs SPY/QQQ |
| 10 | `TSMOMRegimeDetector` | Bull/bear/sideways/high-vol classification |
| 11 | `TSMOMVisualizer` | Equity curve, drawdown, rolling Sharpe, allocation charts |
| 12 | `TSMOMParameterResearch` | Grid search over lookbacks, filters, freq, top-N, sizing |
| 13 | ML Extension | XGBoost/RF/LightGBM trend-persistence prediction |

## Quick Start (API)

```bash
# Start the backend
.venv/bin/uvicorn backend.api.main:app --reload
```

POST `/api/tsmom/analyze` with:

```json
{
  "symbols": "SPY,QQQ,VOO,NVDA,AMD,META",
  "start": "2015-01-01",
  "initial_capital": 100000,
  "long_only": true,
  "top_n": 3,
  "rebalance_freq": "ME",
  "momentum_lookbacks": [21, 63, 126, 252],
  "trend_filters": ["price_above_ma200"],
  "target_volatility": 0.15
}
```

Returns: metrics (Sharpe, return, drawdown), benchmark comparison,
regime analysis, walk-forward results, and optional ML performance.

## Direct Python Usage

```python
from backend.strategies.tsmom_strategy import (
    TSMOMConfig, TSMOMDataLoader, TSMOMBacktestEngine,
    TSMOMBenchmark, TSMOMRegimeDetector,
    TSMOMParameterResearch, ParameterResearchConfig,
)

cfg = TSMOMConfig(
    symbols=["SPY", "QQQ", "VOO"],
    start="2010-01-01",
    momentum_lookbacks=[21, 63, 126],
    trend_filters=["price_above_ma200"],
    sizing_method="vol_target",
    top_n=2,
)
loader = TSMOMDataLoader(cfg)
prices = loader.load_prices()

engine = TSMOMBacktestEngine(cfg)
result = engine.run(prices)
print(result.summary_metrics)

benchmark = TSMOMBenchmark(cfg)
bench = benchmark.compare(result)

regime = TSMOMRegimeDetector(cfg)
regimes = regime.detect_regimes(prices)
regime_analysis = regime.analyze_by_regime(result.equity_curve, regimes)
```

## Parameter Research (Grid Search)

```python
research = TSMOMParameterResearch(cfg)
param_cfg = ParameterResearchConfig()
results = research.run_grid(prices, param_cfg)
df = research.to_dataframe(results)
print(df.sort_values("sharpe_ratio", ascending=False))
```

## CLI (via tsmom_routes)

```bash
.venv/bin/python -m backend.api.tsmom_routes
```

Runs a default multi-asset TSMOM backtest and prints metrics to stdout.

## TSMOMStrategy (for backtest UI / live)

Registered as `"TSMOM Strategy"` in the strategy registry.  Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `momentum_lookback` | 126 | Primary momentum lookback (days) |
| `long_only` | True | Restrict to long signals only |
| `use_trend_filter` | True | Apply trend filter |
| `trend_filter_type` | `price_above_ma200` | Filter type |
| `volatility_position_sizing` | False | Scale positions by vol |
| `signal_type` | `binary` | `binary` or `weighted` |
| `stop_loss_pct` | 0.0 | Stop-loss (0 = disabled) |

## Tests

```bash
.venv/bin/python -m pytest backend/tests/test_tsmom_strategy.py -v
```
