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
├── docs/                   Documentation
│   ├── TSMOM.md            Time-Series Momentum framework
│   ├── CPP_ACCELERATION.md C++ pybind11 kernel reference
│   ├── PAIRS_DISCOVERY.md  Pairs discovery pipeline guide
│   ├── STRATEGIES.md       All strategies reference
│   ├── planning/           Archived planning documents
│   └── reports/            Generated pairs reports
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

## Documentation

| File | Contents |
|------|----------|
| `docs/TSMOM.md` | Time-Series Momentum framework — CLI, API, Python usage |
| `docs/CPP_ACCELERATION.md` | C++ pybind11 kernel reference — build, API, examples |
| `docs/PAIRS_DISCOVERY.md` | Pairs discovery pipeline — CLI, auto-discover, API |
| `docs/STRATEGIES.md` | All 6 strategies — params, usage, how to add new ones |

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

## ML Model Training

The ML pipeline trains classifiers on 38 technical indicator features to predict future price direction. Six model types are supported: **Random Forest**, **Gradient Boosting**, **XGBoost**, **LightGBM**, **SGD (online learning)**, and **MLP (deep learning)**. Advanced ensemble options include **stacking** (all models combined via meta-classifier), **multi-horizon** (separate models per forecast horizon with averaged probabilities), and **regime-aware** wrappers (Hurst exponent + choppiness index modulation). Cross-symbol context features (e.g. SPY/VOO) can be merged in for richer signal. The pipeline backtests each model and compares against SmaCrossover + SimpleStrat 1 baselines, optionally only saving if outperformed.

Data is sourced from **Yahoo Finance** (via `yfinance`), so **no Alpaca keys are needed** for training.

### Usage

#### Basic training

```bash
# macOS / Linux
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01
```

Trains RF and GBT, prints a strategy comparison table, and saves the best model (by Sharpe ratio). No baseline gate check — saves regardless.

#### All model types (including SGD + MLP)

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --model-types rf,gbt,xgb,lgb,sgd,mlp
```

Trains all six model types: Random Forest, Gradient Boosting, XGBoost, LightGBM, SGD (online-capable), and MLP (deep learning). Each is backtested and compared.

#### Grid search (auto-find best hyperparameters)

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --grid-search --model-types rf,gbt,xgb,lgb,mlp
```

Tries combinations of model types, tree counts, depths, and learning rates across RF/GBT/XGB/LGB/MLP. Keeps the combination with the highest validation Sharpe.

#### Only save if it beats baselines

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --beat-baselines
```

The model is saved **only if** it beats both SmaCrossover and SimpleStrat 1 on **total return %** AND **Sharpe ratio**.

#### Stacking ensemble

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --model-types rf,gbt,xgb,lgb,sgd,mlp --stacking
```

Builds a stacking ensemble using all model types as base learners and a LogisticRegression meta-classifier. Compares ensemble performance against individual models.

#### Multi-horizon ensemble

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --multi-horizon 1,5,21
```

Trains separate models for 1-day, 5-day, and 21-day forecast horizons, then averages their `predict_proba` outputs at inference for a multi-timescale signal.

#### Regime-aware switching

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --regime-aware
```

Wraps the trained model with a `RegimeAwareModel` that uses Hurst exponent + choppiness index to modulate predictions: amplify confidence in trending regimes, reverse in mean-reverting regimes, suppress in choppy markets.

#### Cross-symbol context features

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --context-symbols SPY,VOO
```

Downloads additional OHLCV data for SPY and VOO, computes the same 38 features on them, and merges them into the main feature set (left-joined by timestamp, forward-filled). Adds ~76 extra context features.

#### Walk-forward validation

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --walk-forward 5
```

Performs 5-fold walk-forward validation (train on expanding window, validate on next fold) for a more robust performance estimate.

#### Kelly Criterion position sizing

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --kelly --auto-threshold
```

Uses Kelly-optimal position sizes instead of fixed cash tiers, and auto-tunes the confidence threshold on the validation set.

#### Multi-day forecast horizon with trailing stop

```bash
.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --cutoff-date 2023-01-01 --forecast-horizon 5 --max-hold-bars 15 --trailing-stop-pct 0.07
```

Labels targets with 5-day forward returns, forces exit after 15 bars, and applies a 7% trailing stop.

---

### Single-Command Workflow: Auto-Optimizer + Champion

This is the recommended way to train a production ML model. **One command** discovers the
optimal flag combination and saves a champion model ready for backtesting and live trading.

```bash
# Windows (PowerShell)
.venv\Scripts\python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20

# macOS / Linux
.venv/bin/python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20
```

#### What it does

**Phase 1 — Forward selection (10 steps)**
Tests each enhancement independently, only keeping flags that improve Sharpe:

1. Train baseline (RF + GBT)
2. Add `--grid-search`
3. Add `--walk-forward 5`
4. Try all 6 model types (`rf,gbt,xgb,lgb,sgd,mlp`)
5. Add `--stacking`
6. Add `--multi-horizon 1,5,21`
7. Add `--regime-aware`
8. Add `--context-symbols SPY,VOO`
9. Add `--kelly --auto-threshold`
10. Add triple-barrier labeling

**Phase 2 — Champion training**
Takes the optimal flags from Phase 1, forces `--grid-search` + `--walk-forward 5` for
rigorous validation, trains with `--beat-baselines`, and saves the model as
`backend/ml/models/champion_auto.joblib`. Intermediate step models are cleaned up.

#### After the run

The saved `champion_auto.joblib` is loaded automatically by the **ML Strategy** in the
frontend. To use it:

1. Start the API: `.venv\Scripts\uvicorn backend.api.main:app --reload`
2. Open the frontend, go to **Backtest** or **Live**
3. Select **ML Strategy** — it will load `champion_auto.joblib`

#### Skipping steps

Use `--skip-*` flags to ignore enhancements that don't apply to your dataset:

```bash
# Skip walk-forward and stacking for a quick run on small data
.venv\Scripts\python -m backend.ml.auto_optimize --symbols AAPL,MSFT --years 5 --skip-walk-forward --skip-stacking
```

#### Re-training manually

The command at the end of the auto-optimizer output shows the exact `train.py` invocation
to reproduce the champion. Run it any time with a different `--name` to create variants.

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

## Statistical Arbitrage (Pairs Trading CLI)

The `stats_arb` module can be used independently via the command line:

```bash
# Analyze a single pair
.venv/bin/python -m backend.stats_arb.cli --pair KO,PEP --start 2010-01-01

# With ML-enhanced spread prediction
.venv/bin/python -m backend.stats_arb.cli --pair NVDA,META --ml

# Rank multiple pairs from a JSON file
.venv/bin/python -m backend.stats_arb.cli --pairs-file pairs.json --rank --top-n 10

# Cointegration heatmap for a universe of stocks
.venv/bin/python -m backend.stats_arb.cli --heatmap NVDA,AMD,INTC,AAPL,MSFT,GOOGL

# Purged walk-forward validation
.venv/bin/python -m backend.stats_arb.cli --pair XOM,CVX --purged

# JSON output for programmatic consumption
.venv/bin/python -m backend.stats_arb.cli --pair JPM,GS --json

# Auto-discover profitable pairs from S&P 500 (default)
.venv/bin/python -m backend.stats_arb.cli --auto-discover

# Auto-discover from NASDAQ-100 (faster)
.venv/bin/python -m backend.stats_arb.cli --auto-discover nasdaq100

# Auto-discover from custom universe with strict filters + Markdown report
.venv/bin/python -m backend.stats_arb.cli --auto-discover NVDA,AMD,KO,PEP,JPM,GS,MSFT --min-sharpe 1.5 --require-mean-reverting --output-md pairs_report.md
```

Analysis results include: correlation stability, Engle-Granger & Johansen cointegration tests, hedge ratio estimates (OLS/rolling/Kalman), spread stationarity (ADF), half-life mean reversion, Hurst exponent, Ornstein-Uhlenbeck process params, regime detection (Hurst/VIX/CUSUM/Chow/Bai-Perron), purged walk-forward validation, backtest metrics (Sharpe, Sortino, Calmar, drawdown, win rate), and pair ranking with multiple comparison correction (Bonferroni/Benjamini-Hochberg). Plots are saved to `stats_arb_plots/`.

The `--auto-discover` command chains all of this together: it screens an entire universe of stocks (S&P 500, NASDAQ-100, Dow 30, or custom list) for cointegrated pairs, runs the full pipeline on the top candidates, scores them by the composite ranking, then filters by profitability (configurable min Sharpe, return, max drawdown, regime, and structural break constraints). Results are printed as a ranked table and optionally saved as JSON or a Markdown report.

### Automated Pairs Discovery Script

For a streamlined experience, use the dedicated `find_pairs` script — it runs a **3-phase pipeline**:

1. **Phase 1 — Correlation pre-filter**: Downloads all tickers in one call, computes pairwise Pearson correlation on returns, keeps only pairs above `--min-corr` (default 0.5). This avoids the O(n²) yfinance calls that naive approaches require.
2. **Phase 2 — EG Cointegration**: Tests only the correlated pairs for Engle-Granger cointegration (in parallel via cached parquet).
3. **Phase 3 — Full pipeline**: Runs the complete pairs analysis (spread modeling, regime detection, walk-forward, backtest, ranking) on the top N candidates, filters by profitability, saves a Markdown report.

```bash
# Windows (PowerShell)
.venv\Scripts\python -m backend.find_pairs
.venv\Scripts\python -m backend.find_pairs nasdaq100
.venv\Scripts\python -m backend.find_pairs dow30
.venv\Scripts\python -m backend.find_pairs NVDA,AMD,KO,PEP,JPM,GS,MSFT

# macOS / Linux
.venv/bin/python -m backend.find_pairs
.venv/bin/python -m backend.find_pairs nasdaq100
.venv/bin/python -m backend.find_pairs dow30
.venv/bin/python -m backend.find_pairs NVDA,AMD,KO,PEP,JPM,GS,MSFT
```

The report is saved as `pairs_report_<timestamp>.md` by default. Customize with options:

```bash
# Custom output file + strict filters
.venv/bin/python -m backend.find_pairs nasdaq100 -o nasdaq_pairs.md --min-sharpe 1.5 --min-return 10 --require-mean-reverting

# Lower correlation threshold to find more candidate pairs
.venv/bin/python -m backend.find_pairs sp500 --min-corr 0.3 --top-candidates 50
```

| Argument | Default | Description |
|---|---|---|
| `universe` | `sp500` | `sp500`, `nasdaq100`, `dow30`, or comma-separated tickers |
| `-o` / `--output` | `pairs_report_<timestamp>.md` | Output Markdown file path |
| `--min-corr` | `0.5` | Minimum Pearson correlation (Phase 1 filter) |
| `--min-sharpe` | `0.0` | Minimum Sharpe ratio filter (0 = any non-negative) |
| `--min-return` | `-1000.0` | Minimum total return % filter |
| `--max-drawdown` | `-100.0` | Maximum drawdown % filter |
| `--top-candidates` | `50` | Number of cointegrated pairs to fully analyze |
| `--require-mean-reverting` | — | Only keep pairs in mean-reverting regime |
| `--require-no-breaks` | — | Exclude pairs with structural breaks |
| `--start` | `2015-01-01` | Start date for analysis |
| `--capital` | `100000` | Initial capital for backtests |
| `--json` | — | Also print JSON to stdout |

**Note:** True cointegrated pairs with profitable backtests are rare. The pipeline is intentionally conservative — it uses a strict EG test (p < 0.05), requires minimum correlation, walk-forward validates the relationship, and filters by profitability. Expect most universes to yield 0–10 qualifying pairs. If you get zero results, try lowering `--min-corr`, `--min-sharpe`, or increasing `--top-candidates`.

## Strategies

| Strategy | Type | Description |
|---|---|---|
| **SmaCrossover** | Rule-based | Buy when short SMA crosses above long SMA, sell on cross below |
| **Simple Strat 1** | Rule-based | Mean reversion with DCA — buys on drops from open, sells on green days with stop loss |
| **ML Strategy** | ML-based | Loads a trained model (RF/GBT/XGB/LGB/SGD/MLP/stacking) and generates signals from 38 technical indicator features with confidence-based position sizing, optional exit management, online learning (SGD partial_fit), and inference-time context feature merging |
| **CorrCointStrategy** | Pairs | Correlation + cointegration mean-reversion pairs strategy — enters when spread z-score exceeds threshold, exits on reversion, with cooldown, correlation gate, and configurable stop-loss |
| **AutoCointStrategy** | Pairs | Auto-discovers the best cointegrated pair from a comma-separated symbol list at init time, then trades z-score mean reversion with stop-loss and time-based exit |
| **TSMOM Strategy** | Trend-following | Time-Series Momentum — trades persistent trends using multi-lookback momentum signals (21d/63d/126d/252d) with trend filters (price>MA200, MA50>MA200, breakout), volatility-adjusted sizing, long-only or long/short modes |

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [x] Phase 2: Custom Backtest Engine — bar-by-bar simulation, P&L tracking, equity curves, metrics
- [x] Phase 3: Backend API + WebSocket — REST endpoints, WebSocket streaming, strategy registry
- [x] Phase 4: React Frontend Dashboard — 7 pages, 6 widgets, WebSocket streaming, buy/sell markers, theme toggle, 133 frontend tests
- [x] Phase 5: Live Trading Engine — Alpaca live execution, market-open detection, WebSocket streaming, Live UI page
- [x] Phase 6: ML Training Pipeline — yfinance data download, 38 technical indicator features, RF/GBT training, grid search, baseline comparison, confidence-based position sizing
- [x] Phase 7: Advanced ML — SGD/MLP model types, stacking ensemble, multi-horizon ensemble, regime-aware modulation, cross-symbol context features, versioned model storage, on-demand retraining API, ML Lab UI, online learning (partial_fit), auto-optimizer with forward selection
- [x] Phase 8: Statistical Arbitrage Framework — Correlation analysis, cointegration (EG + Johansen), hedge ratio estimation (OLS/rolling/Kalman), spread modeling (half-life, Hurst, ADF, OU), regime detection (Hurst/VIX/CUSUM/Chow/Bai-Perron), purged walk-forward validation, backtest engine, pair ranking with multiple comparison correction, ML extensions, CLI + API + web UI (Correlation + Pairs pages)
- [x] Phase 9: C++ Acceleration — 19 pybind11 kernels for Hurst, ADF, EG cointegration, Kalman filter, Bai-Perron, CUSUM, RSI, MACD, Bollinger Bands, ATR, MFI, ADX with pure-Python fallbacks
- [x] Phase 10: TSMOM Research Framework — Time-Series Momentum with 13 research phases (data pipeline, momentum features, trend filters, signal generation, position sizing, portfolio construction, walk-forward validation, backtest engine, benchmarking, regime analysis, visualizations, parameter research, ML extension), TSMOM Strategy for live/backtest use, REST API
