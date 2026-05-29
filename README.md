# TraderBot

Backtest and live-trade stock strategies via a web dashboard — works with or without an Alpaca account. Includes an ML training pipeline (RF/GBT/XGB/LGB/SGD/MLP + stacking + multi-horizon + regime-aware) that compares against rule-based baselines before deploying.

## Architecture

```
Frontend (React + Vite + Lightweight Charts)      ──►  Dashboard, Live, Backtest, Strategies
  ▲  HTTP REST / WebSocket
  │
Backend (FastAPI + Python)                        ──►  API layer + simulation + live engine
  │
  ├── Alpaca Broker API                           ──►  Market data + live order execution
  ├── Custom Backtest Engine                      ──►  Bar-by-bar simulation (single/multi symbol)
  ├── Live Trading Engine                         ──►  Real-time strategy execution on Alpaca
  ├── ML Training Pipeline                        ──►  yfinance data → 38 features → RF/GBT/XGB/LGB/SGD/MLP + stacking + regime-aware + multi-horizon
  └── SQLite                                       ──►  Trades, snapshots, backtest runs
```

## Project Structure

```
trader/
├── backend/
│   ├── api/               FastAPI routes, WebSockets, models, deps
│   │   ├── main.py        FastAPI app entry + lifespan + CORS
│   │   ├── routes.py      REST: portfolio, positions, orders, backtest CRUD
│   │   ├── websocket.py   /ws/backtest — streaming backtest + replay
│   │   ├── live_routes.py REST: live engine status / stop
│   │   ├── live_websocket.py  /ws/live — streaming live trading
│   │   ├── alpaca_routes.py   Alpaca account / positions / orders proxy
│   │   ├── models.py      Pydantic response models
│   │   └── deps.py        Shared DB dependency
│   ├── strategies/        Strategy base class + implementations
│   │   ├── base.py        Abstract Strategy + Signal + Portfolio
│   │   ├── registry.py    Strategy registry (discovery + instantiation)
│   │   ├── sma_crossover.py    SMA crossover strategy
│   │   ├── simple_strat_1.py   Mean-reversion DCA strategy
│   │   └── ml_strategy.py      ML-based strategy (loads trained model)
│   ├── ml/                ML training pipeline
│   │   ├── features.py    38 technical indicator features + cross-symbol merging
│   │   ├── model.py       Versioned joblib save/load + metadata + list/delete
│   │   ├── train.py       CLI training: 6 model types, grid search, stacking, multi-horizon, regime-aware, walk-forward, Kelly, context symbols
│   │   ├── stacking.py    Stacking ensemble, meta-labeling, regime-aware wrapper, multi-horizon ensemble
│   │   └── models/        Versioned .joblib model files + metadata
│   ├── backtest/          Custom simulation engine + metrics
│   │   ├── engine.py      Bar-by-bar simulation (single & multi-symbol)
│   │   └── metrics.py     Sharpe, drawdown, win rate, profit factor
│   ├── engine/            Live trading engine
│   │   └── live.py        Real-time Alpaca execution with market-open detection
│   ├── data/              Alpaca data loader & SQLite store
│   │   ├── loader.py      Fetches OHLCV + dividends, caches as Parquet
│   │   ├── store.py       SQLite CRUD for snapshots, positions, orders, runs
│   │   └── cache/         Parquet cache for bars & dividends
│   ├── tests/             173 pytest tests covering all modules
│   └── config.py          Settings & env vars
├── frontend/
│   └── src/
│       ├── pages/         Dashboard, Live, Backtest, Strategies
│       ├── components/    AccountSummary, Clock, PortfolioChart,
│       │                  PositionsTable, OrderHistory, StrategySelector
│       ├── api/           HTTP + WebSocket client (BacktestSocket, LiveSocket)
│       └── theme/         ThemeContext (dark/light mode)
├── pyproject.toml          Pytest asyncio config
├── AGENTS.md               Agent instructions
├── ML_PLAN.md              ML implementation plan & details
└── PLAN.md                 Full architecture & build phases
```

## Setup

### Windows (PowerShell)

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend\requirements.txt

# Install frontend dependencies
cd frontend; npm install; cd ..
```

### macOS / Linux

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Configuration (optional)

Copy `.env.example` to `.env` and fill in your Alpaca API keys for real market data and live trading.
The app works without it — it will use synthetic data when Alpaca is unavailable.

```bash
cp .env.example .env
```

## Running

### Windows (PowerShell)

```powershell
# Terminal 1 — Backend
.venv\Scripts\uvicorn backend.api.main:app --reload

# Terminal 2 — Frontend
cd frontend; npm run dev
```

### macOS / Linux

```bash
# Terminal 1 — Backend
.venv/bin/uvicorn backend.api.main:app --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open http://localhost:5173 in your browser.

## Testing

### Backend (209 tests)

```bash
# Windows
.venv\Scripts\python -m pytest backend\tests\ -v --tb=short

# macOS / Linux
.venv/bin/python -m pytest backend/tests/ -v --tb=short
```

### Frontend (41 tests)

```bash
cd frontend; npm test
```

## ML Model Training

The ML pipeline trains classifiers on 38 technical indicator features to predict future price direction. Six model types are supported: **Random Forest**, **Gradient Boosting**, **XGBoost**, **LightGBM**, **SGD (online learning)**, and **MLP (deep learning)**. Advanced ensemble options include **stacking** (all models combined via meta-classifier), **multi-horizon** (separate models per forecast horizon with averaged probabilities), and **regime-aware** wrappers (Hurst exponent + choppiness index modulation). Cross-symbol context features (e.g. SPY/VOO) can be merged in for richer signal. The pipeline backtests each model and compares against SmaCrossover + SimpleStrat 1 baselines, optionally only saving if outperformed.

Data is sourced from **Yahoo Finance** (via `yfinance`), so **no Alpaca keys are needed** for training.

### Usage

#### Basic training

```bash
# macOS / Linux
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20
```

Trains RF and GBT, prints a strategy comparison table, and saves the best model (by Sharpe ratio). No baseline gate check — saves regardless.

#### All model types (including SGD + MLP)

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --model-types rf,gbt,xgb,lgb,sgd,mlp
```

Trains all six model types: Random Forest, Gradient Boosting, XGBoost, LightGBM, SGD (online-capable), and MLP (deep learning). Each is backtested and compared.

#### Grid search (auto-find best hyperparameters)

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search --model-types rf,gbt,xgb,lgb,mlp
```

Tries combinations of model types, tree counts, depths, and learning rates across RF/GBT/XGB/LGB/MLP. Keeps the combination with the highest validation Sharpe.

#### Only save if it beats baselines

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --beat-baselines
```

The model is saved **only if** it beats both SmaCrossover and SimpleStrat 1 on **total return %** AND **Sharpe ratio**.

#### Stacking ensemble

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --model-types rf,gbt,xgb,lgb,sgd,mlp --stacking
```

Builds a stacking ensemble using all model types as base learners and a LogisticRegression meta-classifier. Compares ensemble performance against individual models.

#### Multi-horizon ensemble

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --multi-horizon 1,5,21
```

Trains separate models for 1-day, 5-day, and 21-day forecast horizons, then averages their `predict_proba` outputs at inference for a multi-timescale signal.

#### Regime-aware switching

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --regime-aware
```

Wraps the trained model with a `RegimeAwareModel` that uses Hurst exponent + choppiness index to modulate predictions: amplify confidence in trending regimes, reverse in mean-reverting regimes, suppress in choppy markets.

#### Cross-symbol context features

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --context-symbols SPY,VOO
```

Downloads additional OHLCV data for SPY and VOO, computes the same 38 features on them, and merges them into the main feature set (left-joined by timestamp, forward-filled). Adds ~76 extra context features.

#### Walk-forward validation

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --walk-forward 5
```

Performs 5-fold walk-forward validation (train on expanding window, validate on next fold) for a more robust performance estimate.

#### Kelly Criterion position sizing

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --kelly --auto-threshold
```

Uses Kelly-optimal position sizes instead of fixed cash tiers, and auto-tunes the confidence threshold on the validation set.

#### Multi-day forecast horizon with trailing stop

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --forecast-horizon 5 --max-hold-bars 15 --trailing-stop-pct 0.07
```

Labels targets with 5-day forward returns, forces exit after 15 bars, and applies a 7% trailing stop.

#### 🧪 Kitchen sink experiment (all features combined — for exploration, not deployment)

> **⚠️ Warning:** This enables 10+ interacting features simultaneously. If it produces a good
> result, it is impossible to tell which parts drove the improvement. If it fails to beat
> baselines, it is impossible to tell which parts hurt. **The recommended workflow is to
> add one enhancement at a time**, validate that each independently improves out-of-sample
> Sharpe, and only combine features that have individually proven their worth. This command
> is for curiosity / exploration, not for production use.

```bash
# macOS / Linux
.venv/bin/python -m backend.ml.train \
  --symbols NVDA,AMD,VOO,SPY,META \
  --years 20 \
  --model-types rf,gbt,sgd,mlp \
  --grid-search --regularize \
  --stacking --meta-labeling \
  --walk-forward 5 --embargo 5 \
  --regime-aware \
  --kelly --auto-threshold \
  --multi-horizon 1,5,21 \
  --context-symbols SPY,VOO \
  --labeling triple_barrier \
  --triple-barrier-pct 0.02 --triple-barrier-max-bars 10 \
  --beat-baselines

# Windows (PowerShell — one line)
.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --model-types rf,gbt,sgd,mlp --grid-search --regularize --stacking --meta-labeling --walk-forward 5 --embargo 5 --regime-aware --kelly --auto-threshold --multi-horizon 1,5,21 --context-symbols SPY,VOO --labeling triple_barrier --triple-barrier-pct 0.02 --triple-barrier-max-bars 10 --beat-baselines
```

Enables every ML feature:
- **Grid search** over RF/GBT/SGD/MLP with regularization — finds optimal hyperparameters
- **Stacking ensemble** blends all model types + **meta-labeling** filters low-conviction signals
- **Walk-forward 5** with **embargo 5** — robust purged cross-validation
- **Regime-aware** — Hurst + choppiness modulation
- **Kelly sizing** + **auto-threshold** — optimal position sizing
- **Multi-horizon** (1, 5, 21 day) — multi-timescale predictions
- **Context symbols** (SPY/VOO) — macro cross-asset features
- **Triple barrier labeling** — profit target / stop-loss / time-based exits
- **Beat baselines** — only saves if it outperforms SMA crossover + SimpleStrat 1

Training time is significantly longer than basic mode.

---

### Recommended Workflow: Add One Enhancement at a Time

> **Why this matters.** Combining 10+ features at once makes it impossible to know which part
> drove the result. The correct approach is to start with a bare baseline, add one enhancement
> per run, validate that each independently **improves out-of-sample Sharpe**, and only combine
> things that have individually proven their worth.

#### Step 1 — Train a baseline to beat

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name baseline
```
Train RF + GBT on default params. Note the validation Sharpe. **This is your reference point.** Every enhancement must improve on it.

---

#### Step 2 — Add grid search (find better hyperparams)

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step2_grid --grid-search
```
If validation Sharpe improves → keep grid search. If not → skip it for this dataset.

---

#### Step 3 — Add walk-forward validation

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step3_wf --walk-forward 5 --grid-search
```
More robust performance estimate. Compare averaged Sharpe across folds to the single-split baseline.

---

#### Step 4 — Try more model types (SGD, MLP)

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step4_models --model-types rf,gbt,xgb,lgb,sgd,mlp --grid-search --walk-forward 5
```
Compare all six types. SGD enables online learning later; MLP may capture nonlinear patterns.
If none beat the baseline, revert to `rf,gbt` (faster, more reliable).

---

#### Step 5 — Add stacking ensemble

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step5_stack --model-types rf,gbt,xgb,lgb --stacking --grid-search --walk-forward 5
```
Does the meta-model blend improve Sharpe over the best single model? If not, stacking adds complexity without benefit.

---

#### Step 6 — Add multi-horizon ensemble

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step6_mh --model-types rf,gbt --multi-horizon 1,5,21 --stacking --walk-forward 5
```
If multi-timescale predictions improve Sharpe → keep it. If it dilutes signal → skip.

---

#### Step 7 — Add regime-aware modulation

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step7_regime --regime-aware --stacking --multi-horizon 1,5,21 --walk-forward 5
```
Does Hurst/choppiness modulation help? Compare the averaged regime-aware predictions to unmodulated.

---

#### Step 8 — Add context symbols (market macro features)

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step8_context --context-symbols SPY,VOO --regime-aware --stacking --multi-horizon 1,5,21 --walk-forward 5
```
Market-wide features (SPY, VOO) add ~76 columns. If Sharpe improves → keep. If it hurts → context may be adding noise for your symbols.

---

#### Step 9 — Add Kelly sizing + auto-threshold

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step9_kelly --kelly --auto-threshold --context-symbols SPY,VOO --regime-aware --stacking --multi-horizon 1,5,21 --walk-forward 5
```
Kelly optimizes position sizing; auto-threshold finds the best confidence cutoff. Both should improve risk-adjusted returns independently.

---

#### Step 10 — Add triple-barrier labeling

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name step10_tb --labeling triple_barrier --triple-barrier-pct 0.02 --triple-barrier-max-bars 10 --kelly --auto-threshold --context-symbols SPY,VOO --regime-aware --stacking --multi-horizon 1,5,21 --walk-forward 5 --beat-baselines
```
de Prado's profit-target / stop-loss labeling. More realistic than next-bar direction. Only saves if it beats both baselines.

---

#### Step 11 — Lock in with `--beat-baselines`

Take your best-performing combination and add `--beat-baselines` to make sure it outperforms SmaCrossover + SimpleStrat 1 before saving:

```bash
python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --name champion --kelly --auto-threshold --context-symbols SPY,VOO --regime-aware --stacking --multi-horizon 1,5,21 --walk-forward 5 --labeling triple_barrier --beat-baselines
```

Only the flags that actually improved Sharpe should be included. If a feature didn't help, leave it out.

---

### All CLI Options

| Argument | Default | Description |
|---|---|---|
| `--symbols` | `NVDA,AMD,VOO,SPY,META` | Comma-separated symbols |
| `--years` | `20` | Years of history |
| `--name` | `multi_symbol_model` | Model name for saving |
| `--model-types` | `rf,gbt` | Comma-separated: rf, gbt, xgb, lgb, sgd, mlp |
| `--n-estimators` | `200` | Number of trees |
| `--max-depth` | `10` | Max tree depth |
| `--learning-rate` | `0.1` | Learning rate (tree-based models) |
| `--confidence-threshold` | `0.55` | Min confidence for BUY signal |
| `--base-buy-size` | `1000` | Base $ amount per buy |
| `--cutoff-date` | None | Train/val split date (ISO, e.g. `2023-01-01`) |
| `--val-split` | `0.8` | Training fraction (only used without `--cutoff-date`) |
| `--beat-baselines` | False | Only save if ML beats both baselines |
| `--grid-search` | False | Auto-try hyperparameter combinations |
| `--walk-forward` | `0` | N-fold walk-forward validation (0 = single split) |
| `--kelly` | False | Use Kelly Criterion position sizing |
| `--auto-threshold` | False | Auto-tune confidence threshold on validation set |
| `--labeling` | `next_bar` | Labeling method: `next_bar` or `triple_barrier` |
| `--triple-barrier-pct` | `0.02` | Profit target / stop-loss % for triple barrier (2%) |
| `--triple-barrier-max-bars` | `10` | Max holding period bars for triple barrier |
| `--forecast-horizon` | `1` | Days forward for target prediction |
| `--max-hold-bars` | `30` | Max bars before forced exit |
| `--trailing-stop-pct` | `0.05` | Trailing stop loss fraction (5%) |
| `--stacking` | False | Build a stacking ensemble from all model types |
| `--meta-labeling` | False | Two-stage: primary predicts direction, meta-model filters false signals |
| `--regime-aware` | False | Wrap model with RegimeAwareModel (Hurst + choppiness modulation) |
| `--multi-horizon` | None | Comma-separated horizons for ensemble (e.g. `1,5,21`) |
| `--context-symbols` | None | Comma-separated context symbols for cross-symbol features (e.g. `SPY,VOO`) |
| `--prune` | `0.0` | Drop bottom N% features by importance, retrain, keep if better |
| `--regularize` | False | Add regularization params (min_samples_leaf, subsample) to grid search |
| `--embargo` | `5` | Embargo buffer rows for purged walk-forward |
| `--model-dir` | `backend/ml/models/` | Output directory for saved models |

### What Gets Saved

Trained models are saved to `backend/ml/models/` with automatic versioning:
- `{name}_v{version}.joblib` — the scikit-learn model (e.g. `multi_symbol_model_v3.joblib`)
- `{name}_v{version}_metadata.joblib` — metadata including feature columns, training params, validation metrics, and baseline comparison results
- `{name}.joblib` / `{name}_metadata.joblib` — symlink-like copies pointing to the latest version for backward compatibility

The latest version is used by default when loading a model by name. Specific versions can be loaded via the REST API or `load_model(name, version=N)`.

### Training on Another Machine

Since the training script downloads data from Yahoo Finance (no Alpaca keys required), you can run it on any machine:

1. Copy the project (or just `backend/`) to the target machine
2. Set up the virtual environment and install dependencies
3. Run any of the commands above

## Pages

| Page | What it does |
|---|---|
| **Dashboard** | See your Alpaca account summary, positions, order history, and portfolio equity chart |
| **Backtest** | Pick a strategy + symbols + date range + cash amount, run a backtest with streaming buy/sell markers. Supports multi-symbol, dividend handling, and intraday timeframes. Replay or clear past runs |
| **Live** | Run a strategy live on Alpaca paper trading — select strategy, symbols, and timeframe; start/stop via WebSocket |
| **Strategies** | Browse available strategies and their configurable parameters |
| **ML Lab** | Train and manage ML models: retrain with custom flags, inspect model cards (return, Sharpe, drawdown, win rate), delete old versions |

### Backtest tips
- Pick a **start date** — it runs through the last full trading day automatically
- Enter a **starting amount** (like $100)
- Select one or more **symbols** (NVDA, AMD, VOO, SPY, META)
- Click **Run** — watch the chart and buy/sell markers stream in real time

## Strategies

| Strategy | Type | Description |
|---|---|---|
| **SmaCrossover** | Rule-based | Buy when short SMA crosses above long SMA, sell on cross below |
| **Simple Strat 1** | Rule-based | Mean reversion with DCA — buys on drops from open, sells on green days with stop loss |
| **ML Strategy** | ML-based | Loads a trained model (RF/GBT/XGB/LGB/SGD/MLP/stacking) and generates signals from 38 technical indicator features with confidence-based position sizing, optional exit management, online learning (SGD partial_fit), and inference-time context feature merging |

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [x] Phase 2: Custom Backtest Engine — bar-by-bar simulation, P&L tracking, equity curves, metrics
- [x] Phase 3: Backend API + WebSocket — REST endpoints, WebSocket streaming, strategy registry
- [x] Phase 4: React Frontend Dashboard — 4 pages, 6 widgets, WebSocket streaming, buy/sell markers, theme toggle, 34 frontend tests
- [x] Phase 5: Live Trading Engine — Alpaca live execution, market-open detection, WebSocket streaming, Live UI page
- [x] Phase 6: ML Training Pipeline — yfinance data download, 38 technical indicator features, RF/GBT training, grid search, baseline comparison, confidence-based position sizing
- [x] Phase 7: Advanced ML — SGD/MLP model types, stacking ensemble, multi-horizon ensemble, regime-aware modulation, cross-symbol context features, versioned model storage, on-demand retraining API, ML Lab UI, online learning (partial_fit)
