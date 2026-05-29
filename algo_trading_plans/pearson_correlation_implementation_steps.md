# Pearson Correlation — Implementation Steps

## Overview

Three deliverables that integrate Pearson correlation into the existing trader bot:

1. **Correlation Web Page** — interactive frontend page to visually inspect any pair's correlation over time
2. **Correlation Features** — add correlation indicators to the ML model's feature set
3. **Correlation Breakdown Strategy** — standalone strategy that trades stock/benchmark decoupling

Implementation order: Web page → Features → Strategy (see why at the bottom).

---

## Deliverable 1 — Correlation Web Page (Frontend + API)

Add a new "Correlation" tab to the existing React frontend that lets you select any two stocks and see their rolling correlation over time in an interactive `lightweight-charts` chart.

### Backend — New API endpoint

**File:** `backend/api/models.py` — add `CorrelationDataResponse`
**File:** `backend/api/ml_routes.py` — add `GET /api/correlation/data`

Downloads data via yfinance, computes rolling correlations and statistics, returns JSON:

```json
{
  "symbol_a": "NVDA",
  "symbol_b": "VOO",
  "correlations": {
    "20": [{"time": "2020-01-15", "value": 0.72}, ...],
    "60": [{"time": "2020-03-15", "value": 0.68}, ...],
    "120": [{"time": "2020-06-15", "value": 0.65}, ...]
  },
  "cumulative_returns": {
    "NVDA": [{"time": "2020-01-02", "value": 100}, ...],
    "VOO": [{"time": "2020-01-02", "value": 100}, ...]
  },
  "statistics": {
    "pearson_r": 0.68, "pearson_p": 0.0001,
    "spearman_r": 0.65, "spearman_p": 0.0002,
    "kendall_tau": 0.48, "kendall_p": 0.0003,
    "rolling_corr_std": 0.12,
    "oos_corr_drop": 0.08
  }
}
```

### Frontend — New page

**New file:** `src/pages/Correlation.tsx`
**Modify:** `src/api/client.ts` — add `CorrelationData` type + `getCorrelationData()` method
**Modify:** `src/App.tsx` — add tab

Page layout:

```
┌──────────────────────────────────────────────────────────────┐
│  Symbol A: [NVDA]  Symbol B: [SPY]  Years: [5]              │
│  Windows: [✓ 20] [✓ 60] [☐ 120]   [Load Data]              │
├──────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐│
│  │  Rolling Correlation  (lightweight-charts LineSeries)    ││
│  │                                                          ││
│  │  corr_20 ──  corr_60 ──  zero line ──                  ││
│  │                                                          ││
│  └──────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────┤
│  Statistics:                                                 │
│  Pearson: 0.68 (p<0.001)  │ Spearman: 0.65 (p<0.001)        │
│  Kendall: 0.48 (p<0.001)  │ Rolling Std: 0.12               │
│  OOS Drop: 0.08            │                                 │
└──────────────────────────────────────────────────────────────┘
```

Uses the same `lightweight-charts` library already in the project (no new npm deps).

---

## Deliverable 2 — Correlation Features (for the ML Strategy)

Add Pearson correlation between the main stock and context symbols (e.g. SPY) as features that the ML model can learn from.

### Files to modify

| File | Change |
|---|---|
| `backend/ml/features.py` | Add `add_correlation_features(df, context_data, windows=[20, 60])` |
| `backend/ml/train.py` | Call it in `prepare_features()` after `merge_context_features()` |
| `backend/strategies/ml_strategy.py` | Call it in `init()` after `merge_context_features()` |

### New function in `features.py`

```python
def add_correlation_features(
    df: pd.DataFrame,
    context_data: dict[str, pd.DataFrame],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [20, 60]
    main_ret = df["close"].pct_change()
    for ctx_name, ctx_df in context_data.items():
        ctx_ret = ctx_df["close"].pct_change()
        for w in windows:
            df[f"corr_{ctx_name}_{w}"] = main_ret.rolling(w).corr(ctx_ret)
        short_w, long_w = min(windows), max(windows)
        df[f"corr_{ctx_name}_delta"] = (
            df[f"corr_{ctx_name}_{short_w}"] - df[f"corr_{ctx_name}_{long_w}"]
        )
        df[f"corr_{ctx_name}_trend"] = df[f"corr_{ctx_name}_{short_w}"].diff(5)
    return df
```

### How to use

After implementing, retrain the ML model with a context symbol:

```bash
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --context-symbols SPY \
  --years 20
```

The model will see `corr_SPY_20`, `corr_SPY_60`, `corr_SPY_delta`, and `corr_SPY_trend` as additional features alongside the 38 existing indicators.

---

## Deliverable 3 — Correlation Breakdown Strategy

Standalone strategy implementing Phase 4 of `pearson_correlation.md` (Correlation Breakdown Signal).

**New file:** `backend/strategies/correlation_breakdown.py`
**Modified:** `backend/strategies/registry.py`

### Signal logic

1. Compute short-window (20d) and long-window (60d) rolling correlation between stock returns and benchmark (SPY) returns
2. `corr_delta = corr_short - corr_long`
3. If `corr_delta < threshold` (default: -0.30): the stock is decoupling from the market
   - If stock outperformed benchmark over 5 days → **BUY** (bullish breakout)
   - If stock underperformed benchmark over 5 days → **SELL** (bearish breakdown)
4. If correlation < 0.7 or correlation is too unstable (rolling std > 0.15): no trade

### Risk management

| Rule | Detail |
|---|---|
| Min correlation | Only trade when rolling correlation > 0.70 |
| Correlation stability | Skip if 60d corr rolling std > 0.15 |
| Max holding period | Exit after `max_hold_bars` (default 15) |
| Trailing stop | 5% trailing stop from peak |

### Registry entry

```python
"CorrelationBreakdown": {
    "class": CorrelationBreakdown,
    "description": "Trades when stock correlation to benchmark (SPY) drops sharply — decoupling signal with directional confirmation from relative performance.",
    "params": [
        {"name": "short_window", "type": "int", "default": 20},
        {"name": "long_window", "type": "int", "default": 60},
        {"name": "threshold", "type": "float", "default": -0.30},
        {"name": "benchmark", "type": "str", "default": "SPY"},
        {"name": "buy_size", "type": "float", "default": 1000.0},
        {"name": "max_hold_bars", "type": "int", "default": 15},
        {"name": "trailing_stop_pct", "type": "float", "default": 0.05},
    ],
}
```

---

## Implementation Order

```
 1. Correlation Web Page (Frontend + API)
    │  Visually explore pairs before building on them
    ▼
 2. Correlation Features (ML)
    │  Retrain to see if correlation improves model performance
    ▼
 3. Correlation Breakdown Strategy
    │  Deploy as standalone strategy
    ▼
    Both can run simultaneously — ML model with correlation awareness
    + breakdown strategy as a separate signal
```

## Future (not in scope yet)

- **Pairs Trading Strategy** — requires modifying the backtest engine to handle two-symbol positions (long A / short B as one unit). The current `Strategy` ABC and `BacktestEngine` are single-symbol.
- **Regime Gate** — a correlation-based filter that wraps the ML strategy and suppresses trend signals in low-correlation regimes and reversion signals in high-correlation regimes.
- **Full Testing Tool** (from `pearson_correlation_testing.md`) — the complete 7-phase validation (cointegration, half-life, Granger causality, bootstrap) as additional pages or a CLI.
