# ML Trading Strategy — Implementation Plan

## Goal

Train a machine learning model on 5 symbols (NVDA, AMD, VOO, SPY, META) using 20 years of daily data from Yahoo Finance, then use it in the trading bot to generate buy/sell signals with confidence-based position sizing. **The model must beat both SmaCrossover and SimpleStrat 1 on the validation period before it's saved and used.**

---

## How The Model Works

### Concept

A **Random Forest / Gradient Boosting classifier** that predicts:
> "Will tomorrow's closing price be higher than today's closing price?"

**Input (Features):** 24 technical indicators computed from past price/volume data
**Output:** Probability that next bar will be up (0.0–1.0)

A confidence threshold (default 55%) filters out weak signals. Higher confidence → larger position size.

### Why It Makes Good Decisions From Numbers Alone

1. **Statistical pattern matching** — The model studies thousands of historical bars and learns rules like:
   - "When RSI < 35 AND MACD_hist > 0 AND ret_5 > -2%, next bar was up 72% of the time"
   - Hundreds of such rules are combined into a forest of decision trees
2. **No human bias** — No emotions, no gut feelings, no FOMO. Every decision is purely numerical.
3. **Confidence-weighted voting** — 100+ trees vote on each bar. The fraction voting "up" = the confidence score.
4. **Self-validating** — 20% of historical data is held back during training to verify the model's predictions are real patterns, not noise.
5. **Feature importance** — After training, we can see exactly which indicators drive the predictions.

---

## Architecture

```
backend/
  ml/
    __init__.py              # Package marker
    features.py              # Shared feature computation (24 technical indicators)
    model.py                 # Joblib save/load + metadata helpers
    train.py                 # CLI training script (you run this)
    models/                  # Saved .joblib model files go here
  strategies/
    ml_strategy.py           # Inference strategy — loads saved model, generates signals
    registry.py              # Registers ML Strategy in the UI
```

### Data Flow

```
Training:
  yfinance (20y) → compute features → combine all 5 symbols → train model
  → compare vs SmaCrossover & SimpleStrat 1 on validation set
  → save ONLY if it beats both baselines

Inference (backtest / live):
  Alpaca data → compute same features → model.predict_proba() → BUY/SELL/HOLD
```

---

## 🔁 The Improvement Loop (Key Concept)

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
=== Strategy Comparison (Validation Period) ===
Strategy             Return    Sharpe    Win Rate   Max DD    Trades
ML Strategy (RF)     +28.4%    1.24      68%        -12.3%    142   ← wins?
SmaCrossover         +15.2%    0.72      55%        -18.5%     45
SimpleStrat 1        +12.5%    0.45      48%        -22.1%     32
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

### Phase 1 — Training Script (You Run This)

| # | Task | File |
|---|------|------|
| 1a | Create model save/load helpers (joblib + metadata) | `backend/ml/model.py` |
| 1b | Create CLI training script with argument parsing | `backend/ml/train.py` |
| 1c | yfinance data download for all 5 symbols + caching | `backend/ml/train.py` |
| 1d | Feature computation + label generation (up/down) | `backend/ml/train.py` |
| 1e | Train/validation split (time-based, first 80% train, last 20% val) | `backend/ml/train.py` |
| 1f | Train both RandomForestClassifier and GradientBoostingClassifier, compare | `backend/ml/train.py` |
| 1g | **Mini-backtest on validation set for ML model** (simulates trading with confidence-based sizing) | `backend/ml/train.py` |
| 1h | **Mini-backtest for SmaCrossover + SimpleStrat 1 on same validation set** | `backend/ml/train.py` |
| 1i | **Print strategy comparison table** (Return, Sharpe, Win Rate, Max DD, Trades) | `backend/ml/train.py` |
| 1j | **`--beat-baselines` flag**: only save model if it outperforms both strategies on Sharpe AND total return | `backend/ml/train.py` |
| 1k | **`--grid-search` flag**: auto-try multiple param combinations, keep only the best that beats baselines | `backend/ml/train.py` |
| 1l | Print feature importance chart | `backend/ml/train.py` |
| 1m | Save best model + metadata + comparison results to `backend/ml/models/` | `backend/ml/train.py` |

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
This tries RF + GBT with varying tree counts (100/200/300), depths (5/8/12), learning rates (0.05/0.1/0.2), and keeps only the best combination that beats the baselines.

### What "Beating" Means

The `--beat-baselines` flag checks that the ML model outperforms **both** SmaCrossover and SimpleStrat 1 on:
- **Sharpe ratio** (risk-adjusted return) — must be higher
- **Total return %** — must be higher

Both criteria must pass. You can override with `--metric sharpe` or `--metric return` if you prefer one over the other.

### Phase 2 — Inference Strategy

| # | Task | File |
|---|------|------|
| 2a | Rewrite `MLStrategy.init()` to load model from disk instead of training | `backend/strategies/ml_strategy.py` |
| 2b | Use shared `features.py` for feature computation in inference | `backend/strategies/ml_strategy.py` |
| 2c | Add confidence-based position sizing (scale buy $ by confidence level) | `backend/strategies/ml_strategy.py` |
| 2d | Update registry.py with new params: `model_name`, `confidence_threshold`, `base_buy_size` | `backend/strategies/registry.py` |

#### Confidence-Based Position Sizing

| Confidence | Position Size |
|-----------|--------------|
| 50–55% | HOLD (below threshold) |
| 55–65% | 0.5 × base_buy_size |
| 65–75% | 1.0 × base_buy_size |
| 75–85% | 1.5 × base_buy_size |
| 85–100% | 2.0 × base_buy_size |

This maximizes returns by concentrating capital on high-conviction signals.

### Phase 3 — Verification

| # | Task |
|---|------|
| 3a | Verify strategy loads and shows in `GET /api/strategies` |
| 3b | Verify backtest runs with ML strategy (uses dummy model if none trained yet) |

### Phase 4 — Documentation

| # | Task |
|---|------|
| 4a | Update AGENTS.md with training + inference workflow instructions |

---

## Training Workflow (Your Steps)

```bash
# 1. Install dependencies
.venv\Scripts\pip install -r backend\requirements.txt

# 2. Train the model (first attempt)
.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20

# 3. Review the comparison table printed at the end
#    If ML model beats both → it's saved, you're done ✅
#    If not → try adjusting params:

# 4. Retrain with different params
.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --n-estimators 300 --max-depth 8

# 5. Or let it auto-search until it finds a winner:
.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search

# 6. Once saved, start the backend and backtest in the UI
.venv\Scripts\uvicorn backend.api.main:app --reload
```

---

## Future Enhancements (Phase 5+)

- XGBoost / LightGBM support (stronger models to beat baselines more easily)
- Walk-forward cross-validation (more robust comparison)
- Feature selection (auto-remove low-importance features)
- Ensemble: combine predictions from RF + GBT + XGBoost
- On-demand retraining via API (`POST /api/ml/retrain`)
- ML Lab UI tab to view training history, feature importance, model comparison
