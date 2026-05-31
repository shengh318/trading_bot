# TraderBot

Backtest and live-trade stock strategies via a web dashboard — works with or without an Alpaca account. Includes an ML training pipeline (RF/GBT/XGB/LGB/SGD/MLP + stacking + multi-horizon + regime-aware) that compares against rule-based baselines, a full statistical arbitrage research framework for pairs trading (cointegration, spread modeling, regime detection, walk-forward validation, ranking), C++ accelerated computation kernels (19 functions, pybind11), and a Time-Series Momentum (TSMOM) research and backtesting framework.

## Architecture

```
Frontend (React + Vite + Lightweight Charts)      ──►  Dashboard, Live, Backtest, Strategies, ML Lab, Correlation, Pairs, TSMOM
  ▲  HTTP REST / WebSocket
  │
Backend (FastAPI + Python)                        ──►  API layer + simulation + live engine
  │
  ├── Alpaca Broker API                           ──►  Market data + live order execution
  ├── Custom Backtest Engine                      ──►  Bar-by-bar simulation (single/multi symbol, dividend handling)
  ├── Live Trading Engine                         ──►  Real-time strategy execution on Alpaca
  ├── ML Training Pipeline                        ──►  yfinance data → 38 features → RF/GBT/XGB/LGB/SGD/MLP + stacking + regime-aware + multi-horizon + auto-optimizer
  ├── Statistical Arbitrage Framework             ──►  Pairs trading: cointegration, spread modeling, regime detection, walk-forward, ranking, ML extensions
  ├── TSMOM Research Framework                     ──►  Time-Series Momentum: multi-asset backtest, parameter research, ML extension, walk-forward
  ├── C++ Acceleration (pybind11)                  ──►  19 accelerated kernels: Hurst, ADF, EG coint, Kalman, rolling OLS, RSI, MACD, BB, ATR, MFI, ADX, Bai-Perron, CUSUM
  └── SQLite                                       ──►  Trades, snapshots, backtest runs
```

## Project Structure

```
trader/
├── backend/
│   ├── api/               FastAPI routes, WebSockets, models, deps
│   │   ├── main.py        FastAPI app entry + lifespan + CORS (7 routers)
│   │   ├── routes.py      REST: portfolio, positions, orders, backtest CRUD
│   │   ├── websocket.py   /ws/backtest — streaming backtest + replay
│   │   ├── live_routes.py REST: live engine status / stop
│   │   ├── live_websocket.py  /ws/live — streaming live trading
│   │   ├── alpaca_routes.py   Alpaca account / positions / orders proxy
│   │   ├── ml_routes.py       ML model management / retraining / correlation data API
│   │   ├── pairs_routes.py    Pairs trading analysis / ranking / heatmap API
│   │   ├── tsmom_routes.py   TSMOM analysis / research / ML API
│   │   ├── models.py      Pydantic response models (incl. pairs trading models)
│   │   └── deps.py        Shared DB dependency
│   ├── strategies/        Strategy base class + implementations
│   │   ├── base.py        Abstract Strategy + Signal + Portfolio
│   │   ├── registry.py    Strategy registry (discovery + instantiation)
│   │   ├── sma_crossover.py    SMA crossover strategy
│   │   ├── simple_strat_1.py   Mean-reversion DCA strategy
│   │   ├── ml_strategy.py      ML-based strategy (loads trained model)
│   │   ├── corr_coint_strat.py Correlation + cointegration pairs strategy
│   │   ├── auto_coint_strategy.py Auto-discovers best cointegrated pair from symbol list
│   │   └── tsmom_strategy.py  Time-Series Momentum strategy (multi-asset, 13 research phases)
│   ├── ml/                ML training pipeline
│   │   ├── features.py    38 technical indicator features + cross-symbol merging
│   │   ├── model.py       Versioned joblib save/load + metadata + list/delete
│   │   ├── train.py       CLI training: 6 model types, grid search, stacking, multi-horizon, regime-aware, walk-forward, Kelly, context symbols
│   │   ├── stacking.py    Stacking ensemble, meta-labeling, regime-aware wrapper, multi-horizon ensemble
│   │   ├── auto_optimize.py  Forward-selection search + champion training (Phase 1 + Phase 2)
│   │   ├── logs/          Training log files + active PID tracking
│   │   └── models/        Versioned .joblib model files + metadata
│   ├── backtest/          Custom simulation engine + metrics
│   │   ├── engine.py      Bar-by-bar simulation (BacktestEngine + MultiSymbolBacktestEngine, dividend handling)
│   │   └── metrics.py     Sharpe, drawdown, win rate, profit factor
│   ├── engine/            Live trading engine
│   │   └── live.py        Real-time Alpaca execution with market-open detection
│   ├── data/              Alpaca data loader & SQLite store
│   │   ├── loader.py      Fetches OHLCV + dividends, caches as Parquet
│   │   ├── store.py       SQLite CRUD for snapshots, positions, orders, runs
│   │   └── cache/         Parquet cache for bars & dividends
│   ├── stats_arb/         Statistical Arbitrage Research Framework (pairs trading)
│   │   ├── config.py      Global defaults (significance, windows, fees)
│   │   ├── data.py        Data download/cache via yfinance
│   │   ├── correlation.py Rolling correlation analysis with stability scoring
│   │   ├── cointegration.py   Engle-Granger + Johansen tests
│   │   ├── hedge_ratio.py OLS, rolling OLS, Kalman filter estimation
│   │   ├── spread.py      Half-life, Hurst, ADF, OU process analysis
│   │   ├── regime.py      Hurst, volatility, VIX, CUSUM, Chow, Bai-Perron
│   │   ├── walk_forward.py    Purged walk-forward validation + backtest
│   │   ├── strategy.py    Spread mean-reversion trading strategy
│   │   ├── backtest.py    Comprehensive backtest engine + metrics
│   │   ├── ranking.py     Pair ranking with multiple comparison correction
│   │   ├── ml_models.py   ML extensions for spread prediction
│   │   ├── pipeline.py    Full pipeline orchestrator (all phases)
│   │   ├── cli.py         CLI entry point for analysis + ranking + heatmap + auto-discovery
│   │   ├── discover.py    Auto-discovery — screens cointegrated pairs, runs full analysis, ranks by score, filters by profitability, outputs table/JSON/Markdown
│   │   └── visualization.py   Publication-quality plotting
│   ├── cpp_ext/            C++ pybind11 acceleration module (19 kernels)
│   │   ├── module.cpp      1500-line C++ implementation
│   │   ├── __init__.py     Python wrapper + pure-Python fallbacks
│   │   ├── setup.py        Build configuration (VS 2022)
│   │   └── build.bat       One-click build script
│   ├── tests/             774 pytest tests (51 files) covering all modules
│   │   └── manual/        Ad-hoc test scripts
│   └── config.py          Settings & env vars (Alpaca keys, DB path)
├── frontend/
│   └── src/
│       ├── pages/         Dashboard, Live, Backtest, Strategies, ML Lab, Correlation, Pairs, TSMOM
│       ├── components/    AccountSummary, Clock, PortfolioChart,
│       │                  PositionsTable, OrderHistory, StrategySelector
│       ├── api/           HTTP client + WebSocket classes (BacktestSocket, LiveSocket)
│       └── theme/         ThemeContext (dark/light mode with system preference detection)
├── docs/
│   ├── planning/           Archived planning documents
│   └── reports/            Generated pairs reports
├── Module READMEs          → backend/ml/README.md, backend/strategies/README.md, backend/stats_arb/README.md, backend/cpp_ext/README.md
├── pyproject.toml          Pytest asyncio config
├── AGENTS.md               Agent instructions
├── README.md               Project overview (this file)
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

## Module Documentation

| Module | README |
|--------|--------|
| `backend/ml/` | [ML Training Pipeline](backend/ml/README.md) — 6 model types, grid search, stacking, auto-optimizer |
| `backend/strategies/` | [Strategies](backend/strategies/README.md) — all 6 strategies, params, TSMOM framework |
| `backend/stats_arb/` | [Pairs Trading](backend/stats_arb/README.md) — CLI, auto-discovery, full pipeline |
| `backend/cpp_ext/` | [C++ Acceleration](backend/cpp_ext/README.md) — 19 pybind11 kernel reference |

## Testing

### Backend (774 tests)

```bash
# Windows
.venv\Scripts\python -m pytest backend\tests\ -v --tb=short
# macOS / Linux
.venv/bin/python -m pytest backend/tests/ -v --tb=short
```

### Frontend (133 tests)
> **Note:** Latest commit shows frontend `it()` count at ~51; test files = 17, test count fluctuates as refactoring progresses.

```bash
cd frontend; npm test
```

## Pages

| Page | What it does |
|---|---|
| **Dashboard** | See your Alpaca account summary (cash, portfolio value, buying power, P&L), positions table, order history, and portfolio equity chart |
| **Backtest** | Pick a strategy + symbols + date range + cash amount + timeframe, run a backtest with streaming buy/sell markers. Supports multi-symbol, dividend handling. Replay or clear past runs |
| **Live** | Run a strategy live on Alpaca paper trading — select strategy, symbols, and timeframe; start/stop via WebSocket with bar-by-bar streaming |
| **Strategies** | Browse available strategies and their configurable parameters |
| **ML Lab** | Train and manage ML models: retrain with custom flags (symbols, years, model types, stacking, grid search, walk-forward, etc.), inspect model cards (return, Sharpe, drawdown, win rate), delete old versions |
| **Correlation** | Analyze rolling correlation between two symbols with selectable windows (20/60/120d), cumulative returns chart, and summary statistics (Pearson, Spearman, Kendall) |
| **TSMOM** | Time-Series Momentum analysis and research — multi-asset backtest, walk-forward validation, parameter grid search, regime analysis, ML extension, benchmarking |
| **Pairs** | Full pairs trading analysis: cointegration (Engle-Granger + Johansen), spread analysis (half-life, Hurst, z-score), regime detection, walk-forward validation, backtest metrics, pair ranking with multiple comparison correction, and cointegration heatmaps |

### Backtest tips
- Pick a **start date** — it runs through the last full trading day automatically
- Enter a **starting amount** (like $100)
- Select one or more **symbols** (NVDA, AMD, VOO, SPY, META)
- Click **Run** — watch the chart and buy/sell markers stream in real time

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [x] Phase 2: Custom Backtest Engine — bar-by-bar simulation, P&L tracking, equity curves, metrics
- [x] Phase 3: Backend API + WebSocket — REST endpoints, WebSocket streaming, strategy registry
- [x] Phase 4: React Frontend Dashboard — 7 pages, 6 widgets, WebSocket streaming, buy/sell markers, theme toggle, 133 frontend tests
- [x] Phase 5: Live Trading Engine — Alpaca live execution, market-open detection, WebSocket streaming, Live UI page
- [x] Phase 6: ML Training Pipeline — yfinance data download, 38 technical indicator features, RF/GBT training, grid search, baseline comparison, confidence-based position sizing
- [x] Phase 7: Advanced ML — SGD/MLP model types, stacking ensemble, multi-horizon ensemble, regime-aware modulation, cross-symbol context features, versioned model storage, on-demand retraining API, ML Lab UI, online learning (partial_fit), auto-optimizer with forward selection
- [x] Phase 8: Statistical Arbitrage Framework — full pairs research pipeline (cointegration, spread modeling, regime, walk-forward, ranking, ML)
- [x] Phase 9: C++ Acceleration — 19 pybind11 kernels with pure-Python fallbacks
- [x] Phase 10: TSMOM Research Framework — 13 research phases, live/backtest support, REST API
