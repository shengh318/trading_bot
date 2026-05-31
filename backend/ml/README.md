# ML Training Pipeline

Trains classifiers on 38 technical indicator features to predict future price direction. Six model types: **Random Forest**, **Gradient Boosting**, **XGBoost**, **LightGBM**, **SGD (online learning)**, **MLP (deep learning)**. Advanced options include stacking, multi-horizon ensembles, regime-aware wrappers, cross-symbol context features, walk-forward validation, Kelly Criterion sizing, triple-barrier labeling, and an auto-optimizer.

Data sourced from **Yahoo Finance** — no Alpaca keys needed.

## Quick Start

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20
```

## All CLI Options

| Argument | Default | Description |
|---|---|---|
| `--symbols` | `NVDA,AMD,VOO,SPY,META` | Comma-separated symbols |
| `--years` | `20` | Years of history |
| `--name` | `multi_symbol_model` | Model name for saving |
| `--model-types` | `rf,gbt` | rf, gbt, xgb, lgb, sgd, mlp |
| `--n-estimators` | `200` | Number of trees |
| `--max-depth` | `10` | Max tree depth |
| `--learning-rate` | `0.1` | Learning rate |
| `--confidence-threshold` | `0.55` | Min confidence for BUY |
| `--base-buy-size` | `1000` | Base $ per buy |
| `--cutoff-date` | None | Train/val split (ISO, e.g. `2023-01-01`) |
| `--val-split` | `0.8` | Training fraction |
| `--beat-baselines` | False | Only save if ML beats both baselines |
| `--grid-search` | False | Hyperparameter search |
| `--walk-forward` | `0` | N-fold walk-forward (0 = single split) |
| `--kelly` | False | Kelly Criterion sizing |
| `--auto-threshold` | False | Auto-tune confidence threshold |
| `--labeling` | `next_bar` | `next_bar` or `triple_barrier` |
| `--triple-barrier-pct` | `0.02` | PT/SL % (2%) |
| `--triple-barrier-max-bars` | `10` | Max bars for triple barrier |
| `--forecast-horizon` | `1` | Days forward for target |
| `--max-hold-bars` | `30` | Max bars before forced exit |
| `--trailing-stop-pct` | `0.05` | Trailing stop (5%) |
| `--stacking` | False | Stacking ensemble |
| `--meta-labeling` | False | Two-stage filtering |
| `--regime-aware` | False | Hurst + choppiness modulation |
| `--multi-horizon` | None | Horizons for ensemble (e.g. `1,5,21`) |
| `--context-symbols` | None | Context symbols (e.g. `SPY,VOO`) |
| `--prune` | `0.0` | Drop bottom N% features |
| `--regularize` | False | Regularization in grid search |
| `--embargo` | `5` | Embargo for purged WF |
| `--model-dir` | `backend/ml/models/` | Output directory |

## Usage Examples

### All model types
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 \
    --model-types rf,gbt,xgb,lgb,sgd,mlp
```

### Grid search
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 \
    --grid-search --model-types rf,gbt,xgb,lgb,mlp
```

### Beat baselines guard
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --beat-baselines
```

### Stacking ensemble
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 \
    --model-types rf,gbt,xgb,lgb,sgd,mlp --stacking
```

### Multi-horizon
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 \
    --multi-horizon 1,5,21
```

### Regime-aware
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --regime-aware
```

### Context symbols
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 \
    --context-symbols SPY,VOO
```

### Walk-forward
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --walk-forward 5
```

### Kelly + trailing stop
```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 \
    --kelly --auto-threshold --forecast-horizon 5 --max-hold-bars 15 --trailing-stop-pct 0.07
```

## Auto-Optimizer

One-command discovery of the optimal flag combination:

```bash
.venv/bin/python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20
```

**Phase 1** — 10-step forward selection (baseline → grid → WF → all types → stacking → multi-horizon → regime → context → Kelly → triple-barrier), each step keeps only flags that improve Sharpe.

**Phase 2** — Champion training with optimal flags + `--beat-baselines`. Saves as `champion_auto.joblib`.

Skip steps with `--skip-walk-forward`, `--skip-stacking`, etc.

## Model Files

Saved to `backend/ml/models/` with automatic versioning:
- `{name}_v{version}.joblib` — the model
- `{name}_v{version}_metadata.joblib` — features, params, metrics
- `{name}.joblib` — latest-version symlink

## Module Contents

| File | Purpose |
|------|---------|
| `features.py` | 38 technical indicators (SMA, EMA, RSI, MACD, BB, ATR, ADX, MFI, etc.) |
| `model.py` | Versioned save/load/list/delete |
| `train.py` | CLI entry point (1573 lines) — all flag combinations |
| `stacking.py` | StackingEnsemble, MetaLabeledModel, MultiHorizonEnsemble, RegimeAwareModel |
| `auto_optimize.py` | 10-step forward selection + champion |
| `logs/` | Training logs + PID tracking |
| `models/` | Saved .joblib files |
