# TraderBot

Backtest and live-trade stock strategies via a web dashboard — works with or without an Alpaca account. Includes an ML training pipeline (Random Forest / Gradient Boosting) that compares against rule-based baselines before deploying.

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
  ├── ML Training Pipeline                        ──►  yfinance data → 38 features → RF/GBT/XGB/LGB model
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
│   │   ├── features.py    38 technical indicator feature computation
│   │   ├── model.py       Joblib save/load + metadata helpers
│   │   ├── train.py       CLI training script with grid search + baseline comparison
│   │   └── models/        Saved .joblib model files
│   ├── backtest/          Custom simulation engine + metrics
│   │   ├── engine.py      Bar-by-bar simulation (single & multi-symbol)
│   │   └── metrics.py     Sharpe, drawdown, win rate, profit factor
│   ├── engine/            Live trading engine
│   │   └── live.py        Real-time Alpaca execution with market-open detection
│   ├── data/              Alpaca data loader & SQLite store
│   │   ├── loader.py      Fetches OHLCV + dividends, caches as Parquet
│   │   ├── store.py       SQLite CRUD for snapshots, positions, orders, runs
│   │   └── cache/         Parquet cache for bars & dividends
│   ├── tests/             71+ pytest tests covering all modules
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

### Backend (71+ tests)

```bash
# Windows
.venv\Scripts\python -m pytest backend\tests\ -v --tb=short

# macOS / Linux
.venv/bin/python -m pytest backend/tests/ -v --tb=short
```

### Frontend (34 tests)

```bash
cd frontend; npm test
```

## ML Model Training

The ML pipeline trains a Random Forest, Gradient Boosting, XGBoost, or LightGBM classifier on 38 technical indicator features to predict future price direction (1-day or multi-day horizon). It backtests the model and compares it against SmaCrossover and SimpleStrat 1 baselines, only saving if it outperforms both.

Data is sourced from **Yahoo Finance** (via `yfinance`), so **no Alpaca keys are needed** for training.

### Usage

#### Basic training

```bash
# macOS / Linux
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20
```

Trains RF and GBT, prints a strategy comparison table, and saves the best model (by Sharpe ratio). No baseline gate check — saves regardless.

#### Only save if it beats baselines

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --beat-baselines
```

The model is saved **only if** it beats both SmaCrossover and SimpleStrat 1 on **total return %** AND **Sharpe ratio**.

#### Grid search (auto-find best hyperparameters)

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search
```

Tries combinations of model type (RF/GBT/XGB/LGB), tree counts, depths, and learning rates. Keeps the combination with the highest validation Sharpe.

#### Kelly Criterion position sizing

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --kelly --auto-threshold
```

Uses Kelly-optimal position sizes instead of fixed cash tiers, and auto-tunes the confidence threshold on the validation set.

#### Multi-day forecast horizon

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --forecast-horizon 5 --max-hold-bars 15 --trailing-stop-pct 0.07
```

Labels targets with 5-day forward returns, forces exit after 15 bars, and applies a 7% trailing stop.

#### Walk-forward validation

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --walk-forward 5
```

Performs 5-fold walk-forward validation (train on expanding window, validate on next fold) for a more robust performance estimate.

#### Full production-style training (recommended)

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search --cutoff-date 2023-01-01 --kelly --auto-threshold --forecast-horizon 5 --max-hold-bars 15 --beat-baselines
```

Auto-searches hyperparameters, uses a fixed train/val cutoff, Kelly sizing, multi-day horizon, and only saves if the model beats both baselines.

### All CLI Options

| Argument | Default | Description |
|---|---|---|
| `--symbols` | `NVDA,AMD,VOO,SPY,META` | Comma-separated symbols |
| `--years` | `20` | Years of history |
| `--name` | `multi_symbol_model` | Model name for saving |
| `--model-types` | `rf,gbt` | Comma-separated: rf, gbt, xgb, lgb |
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
| `--forecast-horizon` | `1` | Days forward for target prediction |
| `--max-hold-bars` | `30` | Max bars before forced exit |
| `--trailing-stop-pct` | `0.05` | Trailing stop loss fraction (5%) |
| `--model-dir` | `backend/ml/models/` | Output directory for saved models |

### What Gets Saved

Trained models are saved to `backend/ml/models/` as two files:
- `{name}.joblib` — the scikit-learn model
- `{name}_metadata.joblib` — metadata including feature columns, training params, validation metrics, and baseline comparison results

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
| **ML Strategy** | ML-based | Loads a trained model (RF/GBT/XGB/LGB) and generates signals from 38 technical indicator features with confidence-based position sizing and optional exit management |

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [x] Phase 2: Custom Backtest Engine — bar-by-bar simulation, P&L tracking, equity curves, metrics
- [x] Phase 3: Backend API + WebSocket — REST endpoints, WebSocket streaming, strategy registry
- [x] Phase 4: React Frontend Dashboard — 4 pages, 6 widgets, WebSocket streaming, buy/sell markers, theme toggle, 34 frontend tests
- [x] Phase 5: Live Trading Engine — Alpaca live execution, market-open detection, WebSocket streaming, Live UI page
- [x] Phase 6: ML Training Pipeline — yfinance data download, 24 technical indicator features, RF/GBT training, grid search, baseline comparison, confidence-based position sizing
