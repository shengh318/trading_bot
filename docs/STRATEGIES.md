# Strategies Reference

Six strategies are registered in `backend/strategies/registry.py`.

## 1. SmaCrossover

Simple moving average crossover.  Buy when short SMA > long SMA,
sell when short SMA < long SMA.

```python
params = {"short_window": 10, "long_window": 50}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `short_window` | int | 10 | Fast SMA period |
| `long_window` | int | 50 | Slow SMA period |

## 2. Simple Strat 1

Mean-reversion DCA strategy.  Buys a fixed dollar amount when price
drops from the day's open, sells the entire position on green days,
with a stop-loss.

```python
params = {"buy_size": 100, "entry_drop": 1.0, "profit_target": 20,
          "sell_portion": 100, "stop_loss": 3.0, "max_buys": 1}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `buy_size` | float | 100 | $ per buy |
| `entry_drop` | float | 1.0 | % drop from open to trigger buy |
| `profit_target` | float | 20.0 | % gain to trigger sell |
| `sell_portion` | float | 100.0 | % of position to sell |
| `stop_loss` | float | 3.0 | % stop-loss |
| `max_buys` | int | 1 | Max accumulations per day |

## 3. ML Strategy

Loads a trained `.joblib` model and generates signals from 38
technical indicator features.  Supports confidence-based position
sizing, Kelly Criterion, trailing stops, online learning (SGD
partial_fit), and cross-symbol context features.

```python
params = {"model_name": "multi_symbol_model", "confidence_threshold": 0.55,
          "base_buy_size": 1000, "use_kelly": False,
          "max_hold_bars": 30, "trailing_stop_pct": 0.05}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | str | `multi_symbol_model` | Model name (from ML Lab) |
| `confidence_threshold` | float | 0.55 | Min confidence for BUY |
| `base_buy_size` | float | 1000 | $ per trade |
| `use_kelly` | bool | False | Kelly size position |
| `max_hold_bars` | int | 30 | Max holding period |
| `trailing_stop_pct` | float | 0.05 | Trailing stop |

## 4. CorrCointStrategy

Pairs mean-reversion strategy using correlation + cointegration.
Enters long/short spread when z-score exceeds threshold, exits on
mean reversion, stop-loss, or time limit.  Supports graduated entry,
volatility scaling, P&L stop, half-life weighting, hedge ratio drift
rebalancing, and cooldown.

```python
params = {"symbol_b": "", "hedge_ratio": 1.0, "half_life": 20,
          "z_entry_strong": 2.0, "z_stop": 2.5, "max_holding_days": 40,
          "base_pair_capital": 10000, "cooldown_days": 10}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `symbol_b` | str | `""` | Second leg symbol |
| `hedge_ratio` | float | 1.0 | Spread hedge ratio |
| `z_entry_strong` | float | 2.0 | Z-score to enter |
| `z_stop` | float | 2.5 | Z-score stop-loss |
| `base_pair_capital` | float | 10000 | Capital per pair |
| `cooldown_days` | int | 10 | Cooldown after exit |

## 5. AutoCointStrategy

Auto-discovers the best cointegrated pair from a comma-separated
symbol list at `init()` time.  Downloads prices, runs pairwise
correlation, tests top candidates for EG cointegration, then trades
the best pair via z-score mean reversion.

```python
params = {"symbols": "NVDA,AMD", "z_entry": 2.0, "z_exit": 0.0,
          "stop_loss": 3.0, "max_holding_days": 40,
          "base_pair_capital": 10000}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `symbols` | str | `"NVDA,AMD"` | Comma-separated tickers to search |
| `min_corr` | float | 0.7 | Min correlation filter |
| `z_entry` | float | 2.0 | Z-score entry threshold |
| `z_exit` | float | 0.0 | Z-score exit threshold |
| `stop_loss` | float | 3.0 | Stop-loss multiple of z_entry |
| `max_holding_days` | int | 40 | Max position hold time |
| `base_pair_capital` | float | 10000 | Capital per pair |

## 6. TSMOM Strategy

Time-Series Momentum — trades persistent trends using multi-lookback
momentum signals with trend filters, volatility-adjusted sizing,
long-only or long/short modes.  Suitable for multi-asset portfolios.

```python
params = {"momentum_lookback": 126, "long_only": True,
          "use_trend_filter": True, "trend_filter_type": "price_above_ma200",
          "volatility_position_sizing": False, "target_volatility": 0.15,
          "signal_type": "binary", "max_hold_bars": 0, "stop_loss_pct": 0.0}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `momentum_lookback` | int | 126 | Primary momentum period |
| `long_only` | bool | True | Long signals only |
| `use_trend_filter` | bool | True | Apply trend qualifier |
| `trend_filter_type` | str | `price_above_ma200` | Filter name |
| `volatility_position_sizing` | bool | False | Scale by volatility |
| `target_volatility` | float | 0.15 | Annual vol target |
| `signal_type` | str | `binary` | `binary` or `weighted` |
| `max_hold_bars` | int | 0 | 0 = unlimited |
| `stop_loss_pct` | float | 0.0 | 0 = disabled |

## Adding a New Strategy

1. Create `backend/strategies/your_strat.py`
2. Subclass `Strategy`, implement `init()` and `next()`
3. Import in `backend/strategies/registry.py`
4. Add to `_REGISTRY` (description + params) and `_CLASS_LOADER`
5. Strategy is auto-available in the backtest UI, live UI, and API

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
