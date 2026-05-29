# ML Trading Strategy — Implementation Plan

## Goal

Train a machine learning model on 5 symbols (NVDA, AMD, VOO, SPY, META) using 20 years of daily data from Yahoo Finance, then use it in the trading bot to generate buy/sell signals with confidence-based position sizing. **The model must beat both SmaCrossover and SimpleStrat 1 on the validation period before it's saved and used.**

---

## How The Model Works

### Concept

An ensemble of classifiers (Random Forest / Gradient Boosting / XGBoost / LightGBM) that predicts:
> "Will tomorrow's closing price be higher than today's closing price?"

**Input (Features):** 38 technical indicators computed from past price/volume data
**Output:** Probability that next bar will be up (0.0–1.0)

A confidence threshold (default 55%) filters out weak signals. Higher confidence → larger position size. Optionally, Kelly Criterion dynamically sizes positions based on running win/loss statistics.

### Why It Makes Good Decisions From Numbers Alone

1. **Statistical pattern matching** — The model studies thousands of historical bars and learns rules like:
   - "When RSI < 35 AND MACD_hist > 0 AND ret_5 > -2%, next bar was up 72% of the time"
   - Hundreds of such rules are combined into a forest of decision trees
2. **No human bias** — No emotions, no gut feelings, no FOMO. Every decision is purely numerical.
3. **Confidence-weighted voting** — 100+ trees vote on each bar. The fraction voting "up" = the confidence score.
4. **Self-validating** — 20% of historical data is held back during training to verify the model's predictions are real patterns, not noise (or uses walk-forward cross-validation for more robust validation).
5. **Feature importance** — After training, we can see exactly which indicators drive the predictions.
6. **Kelly Criterion** — Optionally uses running win/loss statistics to compute optimal bet sizing (capped at 25% of capital).

### Labeling Methods

| Method | Description |
|--------|-------------|
| **next_bar** (default) | Binary: 1 if next bar's close > current close, else 0 |
| **triple_barrier** | de Prado's method: 1 if profit target hit first, 0 if stop-loss or max hold expires first. Configurable `--triple-barrier-pct` and `--triple-barrier-max-bars`. |

---

## Architecture

```
backend/
  ml/
    __init__.py              # Package marker
    features.py              # Shared feature computation (38 technical indicators)
    model.py                 # Joblib save/load + metadata helpers
    train.py                 # CLI training script (you run this)
    models/                  # Saved .joblib model files go here
      multi_symbol_model.joblib
      multi_symbol_model_metadata.joblib
  strategies/
    ml_strategy.py           # Inference strategy — loads saved model, generates signals
    registry.py              # Registers ML Strategy in the UI
```

### Data Flow

```
Training:
  yfinance (20y) → compute 38 features → combine all symbols → train model(s)
  → compare vs SmaCrossover & SimpleStrat 1 on validation set
  → save ONLY if it beats both baselines (when --beat-baselines is set)

  Optional enhancements:
    --walk-forward N     → N-fold expanding-window cross-validation
    --grid-search        → auto-tune hyperparams (n_estimators, depth, lr)
    --auto-threshold     → find optimal confidence threshold via Sharpe
    --kelly              → Kelly Criterion position sizing
    --labeling triple_barrier → de Prado labeling
    --forecast-horizon N → predict N bars ahead instead of 1

Inference (backtest / live):
  Alpaca data → compute 38 features → model.predict_proba() → BUY/SELL/HOLD
  → Exit overrides: trailing stop (% drop from peak) + max hold duration
  → Position sizing: fixed tiers (0.5x/1x/1.5x/2x) or Kelly fraction
```

---

## The Improvement Loop (Key Concept)

The core workflow is an **iterative retraining loop**:

```
1. Train model with some hyperparameters
2. Run all 3 strategies on the validation period
3. Print comparison table
4. If ML model beats BOTH SmaCrossover AND SimpleStrat 1 → save it ✅
5. If not → adjust params, try again 🔄
```

Each training run prints a clear scoreboard:

```
====================================================
  STRATEGY COMPARISON  (Validation Period)
====================================================
Strategy                      Return   Sharpe  Win Rate   Max DD  Trades
----------------------------------------------------
ML - RandomForest              +28.4%    1.24     68.0%   -12.3%     142   ← wins?
SmaCrossover                   +15.2%    0.72     55.0%   -18.5%      45
SimpleStrat 1                  +12.5%    0.45     48.0%   -22.1%      32
====================================================
```

You tune hyperparameters and retrain until the ML model leads across the metrics that matter to you.

---

## Phases

### Phase 0 — Infrastructure ✅ (Done)

| # | Task | File |
|---|------|------|
| 0a | Create `backend/ml/` package with `__init__.py` | `backend/ml/__init__.py` |
| 0b | Extract shared feature computation module | `backend/ml/features.py` |
| 0c | Add `yfinance` to requirements.txt | `backend/requirements.txt` |

### Phase 1 — Training Script ✅ (Done — significantly extended)

| # | Task | File |
|---|------|------|
| 1a | Create model save/load helpers (joblib + metadata) | `backend/ml/model.py` |
| 1b | Create CLI training script with argument parsing | `backend/ml/train.py` |
| 1c | yfinance data download for all symbols + caching | `backend/ml/train.py` |
| 1d | Feature computation + label generation (next_bar up/down) | `backend/ml/train.py` |
| 1e | Train/validation split (time-based, first 80% train, last 20% val) | `backend/ml/train.py` |
| 1f | Train multiple model types — RF, GBT, XGBoost, LightGBM — compare | `backend/ml/train.py` |
| 1g | **Mini-backtest on validation set for ML model** (shared portfolio across symbols, confidence-based sizing) | `backend/ml/train.py` |
| 1h | **Mini-backtest for SmaCrossover + SimpleStrat 1 on same validation set** | `backend/ml/train.py` |
| 1i | **Print strategy comparison table** (Return, Sharpe, Win Rate, Max DD, Trades) | `backend/ml/train.py` |
| 1j | **`--beat-baselines` flag**: only save model if it outperforms both strategies on Sharpe AND total return | `backend/ml/train.py` |
| 1k | **`--grid-search` flag**: auto-try multiple param combos (n_estimators, max_depth, lr), keep best by Sharpe | `backend/ml/train.py` |
| 1l | Print feature importance chart (top 10) | `backend/ml/train.py` |
| 1m | Save best model + metadata + comparison results to `backend/ml/models/` | `backend/ml/train.py` |

#### Extended Features (also built)

| # | Task | File |
|---|------|------|
| 1n | **XGBoost + LightGBM** support alongside RF/GBT | `backend/ml/train.py` |
| 1o | **Walk-forward validation** (`--walk-forward N`) — expanding-window folds with optional cutoff date | `backend/ml/train.py` |
| 1p | **Kelly Criterion** position sizing (`--kelly`) in training backtest | `backend/ml/train.py` |
| 1q | **Auto confidence threshold tuning** (`--auto-threshold`) — maximize Sharpe on validation set | `backend/ml/train.py` |
| 1r | **Triple-barrier labeling** (`--labeling triple_barrier`) — de Prado profit-target / stop-loss labeling | `backend/ml/train.py` |
| 1s | **Configurable forecast horizon** (`--forecast-horizon N`) — predict N bars ahead | `backend/ml/train.py` |
| 1t | **Max hold bars + trailing stop** in training backtest | `backend/ml/train.py` |
| 1u | **Walk-forward retrain on full data** after fold averaging | `backend/ml/train.py` |

#### CLI Usage

**Basic training (you decide if it's good enough):**
```bash
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --name my_model \
  --n-estimators 200 \
  --max-depth 10
```

**Only save if it beats both baselines:**
```bash
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --name champion_model \
  --beat-baselines
```

**Auto grid-search until a winner is found:**
```bash
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --name champion_model \
  --grid-search
```
This tries RF + GBT + XGBoost + LightGBM with varying tree counts (100/200/300), depths (5/8/12), learning rates (0.05/0.1/0.2), and keeps only the best combination that beats the baselines.

**Walk-forward validation with auto-threshold:**
```bash
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --walk-forward 3 \
  --auto-threshold
```

**Triple-barrier labeling with Kelly sizing:**
```bash
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --labeling triple_barrier \
  --triple-barrier-pct 0.03 \
  --triple-barrier-max-bars 15 \
  --kelly
```

**All available flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | NVDA,AMD,VOO,SPY,META | Comma-separated symbols |
| `--years` | 20 | Years of history |
| `--name` | multi_symbol_model | Model name for saving |
| `--model-types` | rf,gbt | Model types: rf, gbt, xgb, lgb |
| `--n-estimators` | 200 | Number of trees |
| `--max-depth` | 10 | Max tree depth |
| `--learning-rate` | 0.1 | Learning rate (GBT/XGB/LGB) |
| `--confidence-threshold` | 0.55 | Min confidence for BUY |
| `--base-buy-size` | 1000 | Base buy $ amount |
| `--cutoff-date` | None | Explicit train/val boundary |
| `--val-split` | 0.8 | Training fraction |
| `--beat-baselines` | False | Only save if ML beats both |
| `--grid-search` | False | Grid search hyperparams |
| `--walk-forward` | 0 | N-fold walk-forward validation |
| `--kelly` | False | Kelly Criterion sizing |
| `--auto-threshold` | False | Auto-tune confidence threshold |
| `--labeling` | next_bar | next_bar or triple_barrier |
| `--triple-barrier-pct` | 0.02 | TP/SL % for triple barrier |
| `--triple-barrier-max-bars` | 10 | Max hold for triple barrier |
| `--forecast-horizon` | 1 | N-day forward prediction |
| `--max-hold-bars` | 30 | Max bars in position |
| `--trailing-stop-pct` | 0.05 | Trailing stop loss % |
| `--regularize` | False | Add regularization params to grid search |
| `--embargo` | 5 | Embargo buffer rows for purged walk-forward |
| `--stacking` | False | Build StackingEnsemble from all model types |
| `--meta-labeling` | False | Two-stage: primary + meta-labeler filter |
| `--prune` | 0.0 | Drop bottom N% features by importance and retrain |
| `--context-symbols` | None | Cross-symbol features (e.g. SPY,VOO) |
| `--multi-horizon` | None | Multi-horizon ensemble horizons (e.g. 1,5,21) |
| `--regime-aware` | False | Regime-aware switching via Hurst/choppiness |
| `--model-types` | rf,gbt | Extended: rf, gbt, xgb, lgb, sgd, mlp |

### What "Beating" Means

The `--beat-baselines` flag checks that the ML model outperforms **both** SmaCrossover and SimpleStrat 1 on:
- **Sharpe ratio** (risk-adjusted return) — must be higher
- **Total return %** — must be higher

Both criteria must pass.

### Phase 2 — Inference Strategy ✅ (Done)

| # | Task | File |
|---|------|------|
| 2a | `MLStrategy.init()` loads model from disk instead of training | `backend/strategies/ml_strategy.py` |
| 2b | Use shared `features.py` for feature computation in inference | `backend/strategies/ml_strategy.py` |
| 2c | Confidence-based position sizing (tiers: 0.5x / 1x / 1.5x / 2x of base) | `backend/strategies/ml_strategy.py` |
| 2d | Kelly Criterion position sizing (optional, `use_kelly=True`) | `backend/strategies/ml_strategy.py` |
| 2e | Trailing stop exit (configurable `trailing_stop_pct`) | `backend/strategies/ml_strategy.py` |
| 2f | Max hold bars exit (configurable `max_hold_bars`) | `backend/strategies/ml_strategy.py` |
| 2g | Auto-override strategy params from training metadata | `backend/strategies/ml_strategy.py` |
| 2h | Update registry.py with params: model_name, confidence_threshold, base_buy_size, use_kelly, max_hold_bars, trailing_stop_pct | `backend/strategies/registry.py` |

#### Confidence-Based Position Sizing

| Confidence | Position Size (fixed) | Kelly (optional) |
|-----------|----------------------|-------------------|
| 50–55% | HOLD (below threshold) | HOLD |
| 55–65% | 0.5 × base_buy_size | Kelly fraction × cash |
| 65–75% | 1.0 × base_buy_size | Kelly fraction × cash |
| 75–85% | 1.5 × base_buy_size | Kelly fraction × cash |
| 85–100% | 2.0 × base_buy_size | Kelly fraction × cash |

When Kelly is enabled, bet size = `kelly_fraction × portfolio.cash` (capped at 25% of capital). Kelly fraction adapts as the strategy accumulates win/loss history.

### Phase 3 — Verification ✅ (Done)

| # | Task |
|---|------|
| 3a | Verify strategy loads and shows in `GET /api/strategies` |
| 3b | Verify backtest runs with ML strategy |

### Phase 4 — Documentation ✅ (Done)

| # | Task |
|---|------|
| 4a | Update AGENTS.md with training + inference workflow instructions |

---

## Features (38 Technical Indicators)

| Category | Features |
|----------|----------|
| **Returns** | ret_1, ret_5, ret_21 |
| **Volatility** | vol_5, vol_21, vol_ratio_5_21 |
| **SMA ratios** | close_sma_20, close_sma_50 |
| **Volume** | vol_ratio |
| **Momentum** | mom_10, mom_20, mom_60 |
| **Oscillators** | rsi, macd, macd_signal, macd_hist, mfi |
| **Volatility/ATR** | atr, bb_width, bb_pct_b |
| **Trend** | adx, plus_di, minus_di |
| **Price structure** | price_position, obv_ratio, choppiness |
| **Regime** | hurst (50-bar Hurst exponent) |
| **Lag features** | ret_1_lag1, ret_1_lag2, ret_1_lag3, rsi_lag1, vol_21_lag1 |
| **Calendar** | day_of_week |

---

## Training Workflow (Your Steps)

```bash
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Basic training
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20

# 3. Review the comparison table printed at the end
#    If ML model beats both → it's saved, you're done ✅
#    If not → try adjusting params:

# 4. Try grid-search
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search

# 5. Try walk-forward with auto-threshold
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --walk-forward 3 --auto-threshold

# 6. Require beating baselines to save
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --beat-baselines

# 7. Triple-barrier + Kelly
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --labeling triple_barrier --kelly

# 8. Once saved, start the backend and backtest in the UI
.venv\Scripts\uvicorn backend.api.main:app --reload
```

---

## Phase A — Anti-Overfitting Foundation ✅ (Done)

**Goal:** Make the training pipeline robust against overfitting so backtest performance is more likely to hold up live.

| # | Task | File | Status |
|---|------|------|--------|
| A1 | **Overfitting detection** — print train accuracy alongside validation accuracy, flag warnings if gap > 10%. Add confusion matrix with precision/recall/f1 on validation set. | `backend/ml/train.py` | ✅ |
| A2 | **Regularization grid search** — expand `--grid-search` to include `min_samples_leaf`, `max_features`, `subsample`, `reg_lambda`, `reg_alpha` across all model types. These directly penalize model complexity. | `backend/ml/train.py` | ✅ |
| A3 | **Purged walk-forward with embargo** — de Prado's method. Add embargo buffer and purge overlapping label windows between train/test folds to prevent data leakage. | `backend/ml/train.py` | ✅ |

### Why This Phase Exists

The #1 reason ML trading strategies fail is overfitting. A model can look incredible in backtest (90% win rate!) and then bomb in live trading because it memorized noise. These three changes directly address that:

- **A1 (Detection):** You can't fix what you don't measure. By printing train vs val accuracy side by side, you immediately see if the model is overfitting.
- **A2 (Regularization):** Forces the model to learn simpler, more general patterns instead of memorizing noise.
- **A3 (Purged CV):** Most walk-forward implementations leak future information into training. Embargo + purging fixes that.

### CLI Additions (Phase A & B)

```bash
# Train with purged walk-forward and 5-day embargo
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --walk-forward 3 \
  --embargo 5

# Grid search with full regularization params
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --grid-search \
  --regularize

# Train with stacking ensemble from all model types
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --model-types rf,gbt,xgb,lgb \
  --stacking

# Train with meta-labeling (filters low-conviction signals)
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --meta-labeling

# Train with feature pruning (keeps top 80% of features by importance)
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --prune 0.2

# Full pipeline: regularized grid search + stacking + pruning + meta-labeling
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --grid-search \
  --regularize \
  --stacking \
  --prune 0.2 \
  --meta-labeling
```

New flags: `--embargo N` (default 5), `--regularize`, `--stacking`, `--prune FLOAT`, `--meta-labeling`

---

## Phase B — Accuracy Improvements ✅ (Done)

**Goal:** Improve directional prediction accuracy through ensemble methods and smarter feature usage.

| # | Task | File | Status |
|---|------|------|--------|
| B1 | **Ensemble stacking** — train RF + GBT + XGB + LGB as base models, then a meta-model (LogisticRegression) learns optimal weights. Uses out-of-fold predictions to avoid overfitting the stacker. | `backend/ml/stacking.py`, `backend/ml/train.py` | ✅ |
| B2 | **Feature pruning** — after training, drop bottom N% of features by importance, retrain. Compare both on validation, keep whichever is better. | `backend/ml/train.py` | ✅ |
| B3 | **Meta-labeling** (de Prado) — two-stage model: primary predicts *direction*, secondary predicts *will the primary be correct?* Only take trades when both agree. | `backend/ml/stacking.py`, `backend/ml/train.py` | ✅ |

### How Stacking Works

```
Level 0 (Base models):   RF     GBT     XGB     LGB
                          \      /        \      /
Level 1 (Meta-model):      LogisticRegression (or Ridge)
                                   |
                           Final prediction proba
```

The base models are trained on raw features. Their predictions (probabilities) become the input features for the meta-model. The meta-model learns which base models to trust in which situations.

To prevent the meta-model from overfitting, base model predictions are generated via **5-fold out-of-fold** predictions — each base model is trained 5 times on different splits, predicting the held-out fold each time. The meta-model sees only out-of-sample predictions.

### CLI Additions (Phase B)

```bash
# Train with stacking ensemble
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --stacking

# Train with stacking + feature pruning
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --stacking \
  --prune 0.2

# Full pipeline: regularized grid search + stacking + pruning
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --grid-search \
  --regularize \
  --stacking \
  --prune 0.2
```

New flags: `--stacking` (enable ensemble stacking), `--prune FLOAT` (feature importance threshold, 0 = disabled)

---

## Phase C — Advanced Features ✅ (Done)

**Goal:** Production-grade ML infrastructure — versioned models, retraining API, cross-symbol context, online learning, multi-horizon voting, regime awareness, deep learning, and a dedicated UI tab.

| # | Task | File(s) | Status |
|---|------|---------|--------|
| C1 | **On-demand retraining API** — `POST /api/ml/retrain` with full CLI argument overrides, background subprocess execution, status polling endpoint | `backend/api/ml_routes.py`, `backend/api/main.py`, `backend/api/models.py` | ✅ |
| C2 | **Model versioning** — auto-versioning on save (v1, v2, … via `_{name}_vN.joblib`), `list_models()` with version info, `load_model(name, version=N)`, `delete_model(name, version=N)`, backward-compatible latest files | `backend/ml/model.py` | ✅ |
| C3 | **Cross-symbol features** — `--context-symbols SPY,VOO` flag to inject market-context features into per-symbol predictions. Uses `merge_context_features()` in features.py; context downloaded from yfinance during training; MLStrategy auto-loads context data via yfinance during inference | `backend/ml/features.py`, `backend/ml/train.py`, `backend/strategies/ml_strategy.py` | ✅ |
| C4 | **ML Lab UI tab** — full-featured React page showing trained model cards (return, sharpe, drawdown, win rate, trades), retrain form with all flags (symbols, years, model types, labeling, horizon, grid search, walk-forward, stacking, meta-labeling, kelly, regularize, regime-aware, context symbols, multi-horizon, prune), delete buttons, status polling | `frontend/src/pages/MlLab.tsx`, `frontend/src/App.tsx`, `frontend/src/api/client.ts` | ✅ |
| C5 | **Regime-aware switching** — `RegimeAwareModel` wrapper uses Hurst exponent + choppiness index to modulate predictions: amplify confidence in trending regimes (Hurst > 0.55), reverse signal in mean-reverting regimes (Hurst < 0.45), suppress signals in choppy markets (Choppiness > 60). Flag: `--regime-aware` | `backend/ml/stacking.py`, `backend/ml/train.py` | ✅ |
| C6 | **Deep learning (MLP)** — `mlp` model type using sklearn's `MLPClassifier` with 3 hidden layers (128/64/32), ReLU activation, Adam optimizer, early stopping. Zero extra dependencies. | `backend/ml/train.py` | ✅ |
| C7 | **Online learning** — `sgd` model type (`SGDClassifier` with `log_loss`). MLStrategy: when `online_learning=True`, after each bar it calls `partial_fit()` with the previous bar's actual outcome, enabling the model to adapt during live trading. | `backend/ml/train.py`, `backend/strategies/ml_strategy.py` | ✅ |
| C8 | **Multi-horizon ensemble** — `MultiHorizonEnsemble` trains one model per forecast horizon (e.g. 1-day, 5-day, 21-day) and averages their `predict_proba` outputs. Flag: `--multi-horizon 1,5,21`. Works with stacking and regime-aware. | `backend/ml/stacking.py`, `backend/ml/train.py` | ✅ |

### New CLI Flags (Phase C)

| Flag | Default | Description |
|------|---------|-------------|
| `--context-symbols` | None | Comma-separated context symbols (e.g. SPY,VOO) for cross-symbol features |
| `--multi-horizon` | None | Comma-separated forecast horizons for multi-horizon ensemble (e.g. 1,5,21) |
| `--regime-aware` | False | Wrap model with RegimeAwareModel to adjust predictions by market regime |
| `--model-types` | rf,gbt | Extended: rf, gbt, xgb, lgb, sgd, mlp |

### New Model Types

| Type | Class | Description |
|------|-------|-------------|
| `sgd` | `SGDClassifier` | Online learning via `partial_fit()`. Supports incremental model updates. |
| `mlp` | `MLPClassifier` | Multi-layer perceptron with 128/64/32 hidden layers, ReLU, Adam, early stopping. |

### Phase C CLI Examples

```bash
# Retrain with cross-symbol features (market context)
python -m backend.ml.train \
  --symbols NVDA,AMD,META \
  --years 20 \
  --context-symbols SPY,VOO

# Multi-horizon ensemble (1-day, 5-day, 21-day)
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --model-types rf,gbt,xgb \
  --multi-horizon 1,5,21

# Regime-aware model
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --regime-aware

# SGD online learning model
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --model-types sgd

# MLP deep learning model
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --model-types mlp

# Full pipeline: everything combined
python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --model-types rf,gbt,xgb,lgb \
  --grid-search \
  --regularize \
  --stacking \
  --multi-horizon 1,5,21 \
  --context-symbols SPY,VOO \
  --regime-aware \
  --prune 0.2 \
  --kelly \
  --auto-threshold \
  --walk-forward 3

# Retrain via API
curl -X POST http://localhost:8000/api/ml/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": "NVDA,AMD,VOO,SPY,META",
    "years": 20,
    "name": "champion_v3",
    "model_types": "rf,gbt,xgb,lgb",
    "stacking": true,
    "multi_horizon": "1,5,21",
    "regime_aware": true
  }'
```

### Phase C Architecture

```
POST /api/ml/retrain
  → launches subprocess: python -m backend.ml.train ...
  → returns {status: "started", pid: ...}
  → client polls GET /api/ml/retrain/status/{name}
  → client refreshes model list via GET /api/ml/models

GET /api/ml/models
  → returns list of MlModelInfo with version history

DELETE /api/ml/models/{name}?version=N
  → removes specific version or entire model
```

### Online Learning Flow

```
Live trading bar:
  1. Compute features (main symbol + context symbols)
  2. model.predict_proba() → BUY/SELL/HOLD signal
  3. Wait for next bar (actual outcome known)
  4. model.partial_fit(prev_features, actual_outcome) ← incremental update
```

The model adapts in real time as market conditions change, without requiring a full retrain.

### RegimeAwareModel Logic

| Condition | Market State | Action |
|-----------|-------------|--------|
| Hurst > 0.55 | Trending | Amplify model confidence toward extremes (trend-following) |
| Hurst < 0.45 | Mean-reverting | Reverse the model's signal (reversion to mean) |
| Choppiness > 60 | Choppy / Sideways | Suppress all signals → HOLD |

The Hurst exponent and choppiness index are already computed as features in `compute_features()`, so no additional data is needed.

---

## Phase D — Data Integrity & Anti-Leakage ✅ (Done)

**Goal:** Fix audit-discovered data integrity issues that could silently invalidate historical results.

| # | Task | File | Status |
|---|------|------|--------|
| D1 | **Safe division in features** — replace all raw `/` operations with `safe_divide()`, add `safe_pct_change()`, replace `inf`/`NaN` with 0 at the end of `compute_features()`. Check `isinf` alongside `isnan` in strategy inference. | `backend/ml/features.py`, `backend/strategies/ml_strategy.py` | ✅ |
| D2 | **Purged OOF for stacking** — pass `purge_window` (based on forecast horizon) to `TimeSeriesOOF.split()`, purge training rows near the fold boundary, use `max(embargo, purge_window)` as the effective embargo gap. | `backend/ml/stacking.py`, `backend/ml/train.py` | ✅ |
| D3 | **Rate-limited online learning** — throttle `partial_fit` to every N bars (default 5), add burn-in period, track rolling accuracy over a 50-bar window, pause updates when drift is detected (accuracy < 45%). | `backend/strategies/ml_strategy.py` | ✅ |

### Why Phase D Matters

The three fixes in this phase address the most subtle but dangerous class of bugs:
- **D1 (Division by zero):** NaN/`inf` features silently corrupt training data. Tree-based models handle NaN differently, so a model could be "best" purely because it happened to handle a specific NaN pattern well — not because it learned anything meaningful.
- **D2 (Purged OOF):** The walk-forward outer split correctly purges and embargoes, but the inner OOF split in stacking re-introduced leakage at the ensemble level. This defeated the purpose of walk-forward.
- **D3 (Online learning throttle):** Updating on every bar causes the model to constantly drift from what was validated. A bad week gets "learned" and the model starts systematically avoiding profitable trades.

### CLI Additions (Phase D)

No new flags. Behavioral changes:
- `--stacking` now uses purged OOF splits automatically; purge_window derived from `forecast_horizon` and `triple_barrier_max_bars`
- `--model-types sgd` with online learning in `MLStrategy` now uses `online_learning_every_n=5` by default (configurable in strategy constructor)
- Feature computation is now guaranteed to produce finite values — no NaN/`inf` can leak to the model

### Retrain Required

All models trained before Phase D should be retrained, since the feature pipeline now produces different (correct) values for edge cases that previously produced NaN/`inf`. The fix is backward-compatible in code but the training data has changed.
