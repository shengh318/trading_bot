# Strategies Reference

Six strategies registered in `backend/strategies/registry.py`.

## Strategy Table

| Strategy | Type | Description |
|---|---|---|
| **SmaCrossover** | Rule-based | Buy when short SMA crosses above long SMA, sell on cross below |
| **Simple Strat 1** | Rule-based | Mean reversion DCA — buys on drops from open, sells on green days with stop loss |
| **ML Strategy** | ML-based | Loads trained .joblib model, computes 38 features, confidence-based sizing, Kelly, trailing stops, online learning |
| **CorrCointStrategy** | Pairs | Correlation + cointegration mean-reversion — z-score entry/exit, cooldown, correlation gate, stop-loss |
| **AutoCointStrategy** | Pairs | Auto-discovers best cointegrated pair from comma-separated symbol list at init, then trades z-score mean reversion |
| **TSMOM Strategy** | Trend-following | Time-Series Momentum — multi-lookback momentum signals, trend filters, volatility-adjusted sizing |

## Adding a New Strategy

1. Create `backend/strategies/your_strat.py`, subclass `Strategy`, implement `init()` and `next()`
2. Import in `backend/strategies/registry.py`
3. Add to `_REGISTRY` (description + params) and `_CLASS_LOADER`
4. Auto-available in backtest UI, live UI, and API

```python
from backend.strategies.base import Strategy, Signal

class MyStrategy(Strategy):
    def __init__(self, param1: int = 10):
        self.param1 = param1
    def init(self, data):
        pass
    def next(self, i, data, portfolio):
        return Signal.BUY if i % 2 == 0 else Signal.HOLD
```

## TSMOM Framework

Time-Series Momentum research & backtesting system with 13 phases:

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | `TSMOMDataLoader` | yfinance + parquet caching |
| 2 | `TSMOMFeatures` | Momentum, volatility, risk metrics |
| 3 | `TSMOMTrendFilter` | Price>MA200, MA50>MA200, breakout |
| 4 | `TSMOMSignalGenerator` | Binary/weighted signals |
| 5 | `TSMOMPositionSizer` | Equal, vol-target, risk-parity |
| 6 | `TSMOMPortfolioConstruction` | Top-N rebalance |
| 7 | `TSMOMWalkForward` | Expanding/rolling windows |
| 8 | `TSMOMBacktestEngine` | Multi-asset, t-cost, slippage |
| 9 | `TSMOMBenchmark` | Alpha/beta/IR vs SPY/QQQ |
| 10 | `TSMOMRegimeDetector` | Bull/bear/sideways/high-vol |
| 11 | `TSMOMVisualizer` | Charts (matplotlib/seaborn) |
| 12 | `TSMOMParameterResearch` | Grid search over all params |
| 13 | ML Extension | XGBoost/RF for trend prediction |

### API

POST `/api/tsmom/analyze` with:
```json
{"symbols": "SPY,QQQ,VOO,NVDA,AMD,META", "start": "2015-01-01",
 "initial_capital": 100000, "long_only": true, "top_n": 3}
```

### Direct Python

```python
from backend.strategies.tsmom_strategy import (
    TSMOMConfig, TSMOMDataLoader, TSMOMBacktestEngine,
    TSMOMBenchmark, TSMOMParameterResearch,
)
cfg = TSMOMConfig(symbols=["SPY", "QQQ"], start="2010-01-01")
prices = TSMOMDataLoader(cfg).load_prices()
result = TSMOMBacktestEngine(cfg).run(prices)
print(result.summary_metrics)
```

### TSMOMStrategy Params

| Param | Default | Description |
|-------|---------|-------------|
| `momentum_lookback` | 126 | Primary lookback |
| `long_only` | True | Long signals only |
| `use_trend_filter` | True | Apply trend qualifier |
| `trend_filter_type` | `price_above_ma200` | Filter name |
| `volatility_position_sizing` | False | Scale by vol |
| `signal_type` | `binary` | `binary` or `weighted` |
| `stop_loss_pct` | 0.0 | 0 = disabled |

## Strategy Params

### SmaCrossover
| Param | Default | Description |
|-------|---------|-------------|
| `short_window` | 10 | Fast SMA |
| `long_window` | 50 | Slow SMA |

### Simple Strat 1
| Param | Default | Description |
|-------|---------|-------------|
| `buy_size` | 100 | $ per buy |
| `entry_drop` | 1.0 | % drop from open |
| `profit_target` | 20.0 | % gain to sell |
| `stop_loss` | 3.0 | % stop-loss |

### ML Strategy
| Param | Default | Description |
|-------|---------|-------------|
| `model_name` | `multi_symbol_model` | Model to load |
| `confidence_threshold` | 0.55 | Min confidence |
| `base_buy_size` | 1000 | $ per trade |
| `use_kelly` | False | Kelly sizing |
| `max_hold_bars` | 30 | Max holding period |
| `trailing_stop_pct` | 0.05 | Trailing stop |

### CorrCointStrategy
| Param | Default | Description |
|-------|---------|-------------|
| `symbol_b` | `""` | Second leg |
| `z_entry_strong` | 2.0 | Z-score to enter |
| `z_stop` | 2.5 | Stop-loss z-score |
| `base_pair_capital` | 10000 | Capital per pair |
| `cooldown_days` | 10 | Cooldown after exit |

### AutoCointStrategy
| Param | Default | Description |
|-------|---------|-------------|
| `symbols` | `"NVDA,AMD"` | Tickers to search |
| `z_entry` | 2.0 | Entry z-score |
| `z_exit` | 0.0 | Exit z-score |
| `stop_loss` | 3.0 | Stop-loss multiple |
| `max_holding_days` | 40 | Max position hold |
