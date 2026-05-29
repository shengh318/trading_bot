# TraderBot — Agent Instructions

## Commands
| Action | PowerShell | macOS/Linux |
|--------|-----------|-------------|
| Activate venv | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Run backend | `.venv\Scripts\uvicorn backend.api.main:app --reload` | `.venv/bin/uvicorn backend.api.main:app --reload` |
| Run frontend | `cd frontend; npm run dev` | `cd frontend && npm run dev` |
| Install backend | `.venv\Scripts\pip install -r backend\requirements.txt` | `.venv/bin/pip install -r backend/requirements.txt` |
| Install frontend | `cd frontend; npm install` | `cd frontend && npm install` |
| Typecheck | `cd frontend; npx tsc --noEmit` | `cd frontend && npx tsc --noEmit` |
| Lint | `cd frontend; npx eslint src/` | `cd frontend && npx eslint src/` |
| Run all backend tests | `.venv\Scripts\python -m pytest backend\tests\ -v` | `.venv/bin/python -m pytest backend/tests/ -v` |
| Run all frontend tests | `cd frontend; npx vitest run` | `cd frontend && npx vitest run` |
| Run single backend test | `.venv\Scripts\python -m pytest backend\tests\test_file.py::test_name -v` | `.venv/bin/python -m pytest backend/tests/test_file.py::test_name -v` |
| Auto-optimize + champion | `.venv\Scripts\python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20` | `.venv/bin/python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20` |
| ML train (basic) | `.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20` | `.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20` |
| Stats arb CLI (pair) | `.venv\Scripts\python -m backend.stats_arb.cli --pair KO,PEP` | `.venv/bin/python -m backend.stats_arb.cli --pair KO,PEP` |
| Stats arb CLI (heatmap) | `.venv\Scripts\python -m backend.stats_arb.cli --heatmap NVDA,AMD,INTC,AAPL,MSFT,GOOGL` | `.venv/bin/python -m backend.stats_arb.cli --heatmap NVDA,AMD,INTC,AAPL,MSFT,GOOGL` |
| Stats arb CLI (discover) | `.venv\Scripts\python -m backend.stats_arb.cli --auto-discover` | `.venv/bin/python -m backend.stats_arb.cli --auto-discover` |
| Find pairs (auto) | `.venv\Scripts\python -m backend.find_pairs` | `.venv/bin/python -m backend.find_pairs` |
| Find pairs (nasdaq100) | `.venv\Scripts\python -m backend.find_pairs nasdaq100` | `.venv/bin/python -m backend.find_pairs nasdaq100` |
| Find pairs (custom + save) | `.venv\Scripts\python -m backend.find_pairs NVDA,AMD,KO,PEP -o pairs.md --min-sharpe 1.5` | `.venv/bin/python -m backend.find_pairs NVDA,AMD,KO,PEP -o pairs.md --min-sharpe 1.5` |

## Project Structure

```
backend/
├── api/                     FastAPI layer (7 routers)
│   ├── main.py              App entry, lifespan, CORS, 7 router includes
│   ├── routes.py            /api/strategies, /api/backtest, /api/portfolio
│   ├── websocket.py         /ws/backtest — streaming backtest + replay
│   ├── live_routes.py       /api/live/status, /api/live/stop
│   ├── live_websocket.py    /ws/live — streaming live trading
│   ├── alpaca_routes.py     /api/alpaca/account, /api/alpaca/positions, /api/alpaca/orders, /api/alpaca/portfolio
│   ├── ml_routes.py         /api/ml/models, /api/ml/retrain, /api/correlation/data
│   ├── pairs_routes.py      /api/pairs/analyze, /api/pairs/rank, /api/pairs/heatmap
│   ├── models.py            Pydantic models: AccountSummary, BacktestRunRequest, MlModelInfo, PairData, etc.
│   └── deps.py              Shared Database singleton dependency
├── strategies/              Strategy base class + implementations
│   ├── base.py              Strategy (abstract: init, next, on_trade), Portfolio, Signal enum
│   ├── registry.py          list_strategies(), get_strategy(name) — auto-discovers via STRATEGY_REGISTRY
│   ├── sma_crossover.py     SmaCrossover — buy when short SMA > long SMA, sell on cross below
│   ├── simple_strat_1.py    SimpleStrat1 — mean reversion DCA, buys on drops from open, sells green days
│   └── ml_strategy.py       MLStrategy — loads trained .joblib model, computes 38 features, buy/sell/hold
├── ml/                      ML training pipeline
│   ├── features.py          compute_features(df) — 38 indicators (SMA, EMA, RSI, MACD, BB, ATR, etc.)
│   ├── model.py             save_model/load_model/list_models/delete_model — versioned .joblib + metadata
│   ├── train.py             CLI training: 6 model types, grid search, stacking, multi-horizon, regime, walk-forward, Kelly, context symbols, triple barrier, pruning, regularization, embargo
│   ├── stacking.py          StackingEnsemble, MetaLabeledModel, MultiHorizonEnsemble, RegimeAwareModel
│   ├── auto_optimize.py     10-step forward selection + champion training
│   ├── logs/                active_pids.json, train_*.log files
│   └── models/              *.joblib + *_metadata.joblib versioned model files
├── backtest/                Simulation engine + metrics
│   ├── engine.py            BacktestEngine (single symbol) + MultiSymbolBacktestEngine, both with .run() and .stream()
│   └── metrics.py           calculate_metrics() — Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor
├── engine/                  Live trading engine
│   └── live.py              LiveEngine — Alpaca paper trading, market-open detection, bar polling, WebSocket callbacks
├── data/                    Data layer
│   ├── loader.py            DataLoader — yfinance fallback/cache, Alpaca primary, dividend fetching
│   ├── store.py             Database — SQLite CRUD for backtest_runs, snapshots, positions, orders
│   └── cache/               Parquet files for bars and dividends
├── stats_arb/               Statistical Arbitrage Research Framework (pairs trading)
│   ├── config.py            Defaults: windows, significance, z-entry/exit, fees, walk-forward params
│   ├── data.py              DataManager — yfinance download + caching for pairs
│   ├── correlation.py       CorrelationAnalyzer — rolling windows, stability score, regime changes, correlation collapse
│   ├── cointegration.py     CointegrationTester (Engle-Granger) + JohansenTester
│   ├── hedge_ratio.py       HedgeRatioEstimator — OLS, rolling OLS, KalmanFilterHedge
│   ├── spread.py            SpreadAnalyzer — half-life, Hurst, ADF, OU process, autocorr, variance ratio
│   ├── regime.py            RegimeDetector — Hurst, volatility, VIX, CUSUM, Chow, Bai-Perron structural breaks
│   ├── walk_forward.py      WalkForwardValidator, PurgedWalkForwardValidator, WalkForwardBacktest
│   ├── strategy.py          TradingStrategy — spread z-score mean reversion with dynamic sizing
│   ├── backtest.py          BacktestEngine — P&L, equity curve, Sharpe, Sortino, drawdown, turnover
│   ├── ranking.py           PairRanker — Bonferroni + Benjamini-Hochberg multiple comparison correction
│   ├── ml_models.py         SpreadPredictor — RF/GBT/XGB regression for spread prediction
│   ├── pipeline.py          PairAnalyzer — orchestrates all phases: data→correlation→coint→spread→regime→WF→strategy→backtest→(ML)
│   ├── cli.py               CLI entry point: --pair, --pairs-file, --heatmap, --rank, --ml, --purged, --json
│   ├── discover.py          Auto-discovery: screens all pairs via EG cointegration, full pipeline on top candidates, ranks, filters by profitability, outputs table + JSON + Markdown
│   └── visualization.py     Visualizer — spread/zscore/cumulative/heatmap plots via matplotlib+seaborn
├── find_pairs.py             Automated pairs discovery wrapper (correlation + cointegration + markdown report)
├── tests/                   277 pytest tests (26 files)
│   ├── test_backtest_engine.py        (12)  — single + multi-symbol, streaming, dividend
│   ├── test_backtest_metrics.py       (11)  — Sharpe, drawdown, win rate calculations
│   ├── test_api_routes.py             (17)  — strategies, backtest CRUD, portfolio
│   ├── test_websocket.py              (4)   — WebSocket backtest streaming
│   ├── test_store.py                  (14)  — SQLite CRUD operations
│   ├── test_loader.py                 (5)   — data fetching + caching
│   ├── test_config.py                 (4)   — config validation
│   ├── test_api.py                    (4)   — health check
│   ├── test_bugs_*.py                 (8 files, 120) — regression tests for bugs 1-22
│   ├── test_bug_backtest_sell_portion.py (3)  — stale qty on SELL (UnboundLocalError)
│   ├── test_bug_backtest_zero_price.py   (3)  — ZeroDivisionError on zero price
│   ├── test_bug_metrics_zero_cash.py     (2)  — inf total_return_pct at zero cash
│   ├── test_bug_ml_auc.py                (3)  — AUC always 0.0 from boolean → scaler
│   ├── test_bug_spread_zscore.py         (3)  — inf/nan z-scores on constant spread
│   ├── test_correlation.py            (35)  — correlation API endpoint
│   ├── test_stats_arb.py              (18)  — cointegration, ranking, strategy, walk-forward
│   └── conftest.py                    — pytest fixtures
├── config.py                 Settings: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, DB_PATH, validate_config()
└── requirements.txt          fastapi, uvicorn, alpaca-py, pandas, numpy, scikit-learn, xgboost, lightgbm, yfinance, statsmodels, matplotlib, seaborn, scipy, pytest, httpx, websockets, pyarrow
```

```
frontend/
└── src/
    ├── App.tsx                     Root — 7 tabs (Dashboard, Live, Backtest, Strategies, ML Lab, Correlation, Pairs)
    ├── main.tsx                    React entry point
    ├── setupTests.ts               Vitest setup: mock lightweight-charts + matchMedia
    ├── api/
    │   ├── client.ts               ApiClient class (24 methods) + BacktestSocket + LiveSocket WebSocket wrappers
    │   ├── client.test.ts          (16 tests) — all REST methods + WebSocket connect/send/close/error
    │   └── __tests__/
    │       └── client.extra.test.ts (30 tests) — WS double-connect leaks, HTTP errors, JSON parse errors, close-callback races
    ├── pages/
    │   ├── Dashboard.tsx           Account summary, positions, orders, portfolio chart (all from Alpaca)
    │   ├── Live.tsx                Live strategy runner — strategy selector, symbol toggles, equity chart, console log
    │   ├── Backtest.tsx            Backtest runner — strategy selector, param editing, date/cash inputs, WS streaming, trade markers, past runs
    │   ├── Strategies.tsx          Strategy browser — cards with name, description, params
    │   ├── MlLab.tsx               ML model management — list models with metric cards, retrain form with full flags, delete
    │   ├── Correlation.tsx         Correlation analysis — rolling correlation chart, cumulative returns, statistics table
    │   ├── Pairs.tsx               Pairs trading — analyze/rank/heatmap sections, spread chart, z-score, regime, backtest metrics
    │   └── __tests__/
    │       ├── Backtest.bugs.test.tsx      (4 tests)
    │       ├── Dashboard.bugs.test.tsx     (3 tests)
    │       ├── Strategies.test.tsx         (5 tests)
    │       └── MlLab.bugs.test.tsx         (3 tests)
    ├── components/
    │   ├── AccountSummary.tsx      Cash, portfolio value, buying power, P&L (horizontal flex with color coding)
    │   ├── PositionsTable.tsx      Holdings table: symbol, qty, avg price, current, P&L, value
    │   ├── OrderHistory.tsx        Orders table: ID, symbol, side, qty, filled, price, status, type, created
    │   ├── PortfolioChart.tsx      Lightweight-charts: area or baseline series with buy/sell trade markers
    │   ├── StrategySelector.tsx    Dropdown + dynamic param inputs rendered from StrategyInfo
    │   └── Clock.tsx               Live clock (HH:MM:SS AM/PM + date), updates every 1s
    └── theme/
        └── ThemeContext.tsx        ThemeProvider — dark/light, system preference detection, full ThemeColors interface
    Tests: 11 files, 79 `it()` calls total
```

## Key Architecture

### API Layer (FastAPI)
- 7 routers in `backend/api/main.py:35-41`: router, ws_router, live_router, live_ws_router, alpaca_router, ml_router, pairs_router
- All return Pydantic models defined in `backend/api/models.py`
- DB dependency via `backend/api/deps.py` (singleton `Database`)

### Strategies (`backend/strategies/`)
- `Strategy` (abstract base in `base.py:7`) — requires `init(data: pd.DataFrame)`, `next(i, data, portfolio) -> Signal`, optional `on_trade()`
- `Portfolio` — holds `cash`, `positions: dict[str, float]`, `avg_entry: dict[str, float]`
- `Signal` enum — `BUY`, `SELL`, `EXIT`, `HOLD`, `NONE`
- `registry.py` — imports all strategies into `STRATEGY_REGISTRY` dict, `get_strategy(name)` creates instances
- Add new strategy: create file, subclass `Strategy`, implement `init` + `next`, import in `registry.py`, add to `STRATEGY_REGISTRY`

### Backtest (`backend/backtest/engine.py`)
- `BacktestEngine` — single symbol, `run()` -> `BacktestResult`, `stream()` -> Generator[dict]
- `MultiSymbolBacktestEngine` — union timestamps, per-symbol strategy instances, shared portfolio
- Both handle dividends via `_apply_dividend()`
- `stream()` yields `{"snapshot": dict, "trade": dict|None, "signal": str, "dividend": dict|None}`
- Websocket in `websocket.py` packs these into JSON frames: `{"type": "bar", "data": {...}}` + `{"type": "complete", "metrics": {...}}`

### Live Engine (`backend/engine/live.py`)
- `LiveEngine` — connects to Alpaca paper trading, polls bars, executes MarketOrder
- Market-open detection (checks if market is open, waits otherwise)
- Per-symbol DataBuffer (max 500 bars), strategy instances
- `start(callbacks)` — callbacks: `on_bar`, `on_status`, `on_error`
- `_run_loop` — poll loop every ~60s (bar interval), checks open, fetches bars, runs strategies, places orders

### ML Pipeline (`backend/ml/`)
- **Features** (`features.py:24`): `compute_features(df)` → 38 columns: SMA(5/10/20/50/200), EMA(12/26), RSI(14), MACD, BB upper/lower/width/%, ATR(14), volume SMA(5/21), volume delta, ROC(1/5/21), log returns (1/5/21), volatility (5/21), rolling max/min z-score, close/corr with SPY
- **Model** (`model.py`): `save_model()` — versioned .joblib + metadata; `load_model(name, version)`; `list_models()`; `delete_model()`
- **Train** (`train.py`): 1573 lines — CLI with argparse, downloads yfinance, computes features, trains 6 model types, grid search, walk-forward, stacking, multi-horizon, regime-aware, Kelly, triple barrier, meta-labeling, pruning, regularization, embargo, baseline comparison, backtest
- **Stacking** (`stacking.py`): `StackingEnsemble`, `MetaLabeledModel`, `MultiHorizonEnsemble`, `RegimeAwareModel`
- **Auto-optimize** (`auto_optimize.py`): 10-step forward selection (baseline → grid → WF → all types → stacking → multi-horizon → regime → context → Kelly → triple-barrier), then Phase 2 champion training with `--beat-baselines`

### Statistical Arbitrage (`backend/stats_arb/`)
- **Pipeline order** (in `pipeline.py`): Data → Correlation → Cointegration (EG + Johansen) → Hedge Ratio (OLS/rolling/Kalman) → Spread (half-life, Hurst, ADF, OU) → Regime (Hurst/VIX/CUSUM/Chow/Bai-Perron) → Walk-Forward (purged) → Strategy (z-score mean reversion) → Backtest → Optional ML
- **PairRanker** (`ranking.py`): composite score = weighted combination of coint p-value, half-life, spread Sharpe, hedge ratio stability, regime stability; multiple comparison correction via Bonferroni or Benjamini-Hochberg
- **CLI** (`cli.py`): `--pair A,B`, `--pairs-file file.json --rank --top-n 10`, `--heatmap TICKERS`, `--ml`, `--purged`, `--json`, `--no-plots`, `--no-parallel`, `--capital 100000`
- **Auto-Discovery** (`discover.py`): `--auto-discover [sp500|nasdaq100|dow30|TICKERS]` — screens all pairs via EG cointegration, runs full pipeline on top candidates, ranks, filters by profitability (Sharpe ≥ 1.0), outputs table + optional JSON/Markdown
- **API**: `POST /api/pairs/analyze` (with `PairAnalysisRequest`), `POST /api/pairs/rank`, `POST /api/pairs/heatmap`

### Frontend (`frontend/src/`)
- **No router** — tab-based navigation via `useState<Tab>` in `App.tsx`
- **API client** (`api/client.ts`): single `ApiClient` class with 24 methods, all return typed responses
  - WebSocket: `BacktestSocket` — `connect(strategy, symbol, start, end, cash)`, events: `onBar`, `onComplete`, `onError`
  - WebSocket: `LiveSocket` — `connect(strategy, symbols, timeframe)`, events: `onBar`, `onStatus`, `onError`
- **Styling**: all inline styles with theme colors from `ThemeContext`
- **Testing**: vitest + jsdom + @testing-library/react; `setupTests.ts` mocks `lightweight-charts` and `window.matchMedia`

## Tests Summary

### Backend: 277 tests across 26 files
| File | Tests | What it covers |
|------|-------|----------------|
| `test_api_routes.py` | 17 | Strategies, backtest CRUD, portfolio, error cases |
| `test_backtest_engine.py` | 12 | Single/multi-symbol run + stream, dividends |
| `test_backtest_metrics.py` | 11 | Sharpe, drawdown, win rate, profit factor |
| `test_websocket.py` | 4 | WS connection, streaming, cleanup |
| `test_store.py` | 14 | DB CRUD for runs, snapshots, positions, orders |
| `test_loader.py` | 5 | Data fetching, caching, missing data handling |
| `test_config.py` | 4 | Config validation, missing keys |
| `test_api.py` | 4 | Health endpoint |
| `test_correlation.py` | 35 | _safe helper, API endpoint, model validation |
| `test_stats_arb.py` | 18 | Cointegration, ranking corrections, strategy, WF leakage |
| `test_bugs_*.py` (8 files) | 120 | Regression tests for bugs 1-22 across all modules |
| `test_bug_backtest_sell_portion.py` | 3 | Stale qty on SELL (UnboundLocalError) |
| `test_bug_backtest_zero_price.py` | 3 | ZeroDivisionError on zero price |
| `test_bug_metrics_zero_cash.py` | 2 | inf total_return_pct at zero cash |
| `test_bug_ml_auc.py` | 3 | AUC always 0.0 from boolean → scaler |
| `test_bug_spread_zscore.py` | 3 | inf/nan z-scores on constant spread |
| `test_correlation.py` | 35 | Correlation API endpoint |
| `test_stats_arb.py` | 18 | Cointegration, ranking, strategy, walk-forward |

### Frontend: 79 `it()` calls across 11 test files
| File | Tests | What it covers |
|------|-------|----------------|
| `App.test.tsx` | 4 | Rendering, tabs, tab switching |
| `client.test.ts` | 16 | All 10 REST methods + 6 WebSocket methods |
| `client.extra.test.ts` | 30 | WS double-connect leaks, HTTP errors, JSON parse errors, close-callback races |
| `StrategySelector.test.tsx` | 5 | Rendering, selection, param changes |
| `PositionsTable.test.tsx` | 3 | Empty state, rows, P&L coloring |
| `OrderHistory.test.tsx` | 3 | Empty state, rows, buy/sell coloring |
| `AccountSummary.test.tsx` | 3 | Null, fields, negative P&L |
| `Backtest.bugs.test.tsx` | 4 | Default cash, callback deps, param validation |
| `Dashboard.bugs.test.tsx` | 3 | Alpaca error, loading, fallback |
| `Strategies.test.tsx` | 5 | Rendering, selection, param changes, API error handling |
| `MlLab.bugs.test.tsx` | 3 | Polling name mismatch, walk_forward type, interval cleanup |

## Data Flow Patterns

1. **Backtest**: Frontend `BacktestSocket.connect()` → WS to `/ws/backtest` → `BacktestEngine.stream()` yields dicts → WS relays as JSON → frontend charts bars + trade markers
2. **Live Trading**: Frontend `LiveSocket.connect()` → WS to `/ws/live` → `LiveEngine.start()` with callbacks → polls Alpaca bars → runs strategies → executes orders → streams bars/trades via WS
3. **ML Retrain**: Frontend MlLab POST `/api/ml/retrain` → spawns `backend.ml.train` subprocess → saves `.joblib` files → frontend polls `GET /api/ml/models` for completion
4. **Pairs Analysis**: Frontend POST `/api/pairs/analyze` → `PairAnalyzer.analyze()` runs full pipeline → returns `PairAnalysisResponse` with all metrics + series data → frontend renders spread/z-score charts + metric cards

## Configuration
- `.env` file: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER`, `DB_PATH`
- No Alpaca keys = app still works (yfinance fallback for data, synthetic for live)
- `.env.example` template provided

## Dependency Versions (key)
- Python 3.12+, FastAPI 0.115, scikit-learn 1.6, xgboost 3.2, lightgbm 4.6, statsmodels 0.14, yfinance 1.4
- Node 18+, React 18, lightweight-charts 4.2, TypeScript ~5.6, Vite 6, Vitest 2.1

## Conventions
- Type hints required on all Python functions
- Frontend uses strict TypeScript (`strict: true` in tsconfig.json)
- All API routes return Pydantic models (no plain dicts/raw responses)
- Frontend has no CSS framework — all styling is inline with theme colors
- No client-side routing — tab-based navigation via `useState`
- No committing unless explicitly asked
- `noUnusedLocals: true` and `noUnusedParameters: true` in TypeScript
