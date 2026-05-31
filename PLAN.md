# TraderBot — Full Architecture Plan

## Tech Stack
| Component | Choice |
|---|---|---|
| Backend / Strategies | Python 3.12+ |
| Backtesting | Custom mini-engine (pandas-based) |
| API Framework | FastAPI |
| Database | SQLite (dev) |
| Frontend | React + TypeScript + Vite |
| Charting | TradingView Lightweight Charts |
| Broker API | Alpaca (paper + live) |
| Real-time | WebSocket (backtest + live streaming) |
| ML Pipeline | scikit-learn, XGBoost, LightGBM (6 model types) |
| Pairs Trading | statsmodels, custom research framework |

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                        │
│  7 tabs: Dashboard | Live | Backtest | Strategies | ML Lab |     │
│          Correlation | Pairs                                      │
│  ┌──────────────┐ ┌────────────────────────────────────┐         │
│  │ Account Bar  │ │ Portfolio Equity Curve (lightweight-│         │
│  │ Cash, Value  │ │ charts — area/baseline series)     │         │
│  └──────────────┘ └────────────────────────────────────┘         │
│  ┌──────────────┐ ┌────────────────────────────────────┐         │
│  │ Positions    │ │ Order History / Strategy Console   │         │
│  └──────────────┘ └────────────────────────────────────┘         │
│                                                                    │
│  Per-page panels: Spread/Z-score charts (Pairs), rolling          │
│  correlation (Correlation), model cards (ML Lab), buy/sell        │
│  markers on chart (Backtest/Live)                                  │
└──────────────────────────────────────────────────────────────────┘
           ▲  HTTP REST (24 API methods)     ▲ WebSocket (/ws/backtest, /ws/live)
           │                                 │
┌──────────┴─────────────────────────────────┴──────────────────────────────┐
│                          FastAPI Backend                                   │
│  7 routers: routes | ws_router | live_router | live_ws_router |           │
│             alpaca_router | ml_router | pairs_router                       │
│                                                                             │
│  ┌────────────────────┐ ┌──────────────────┐ ┌────────────────────────┐   │
│  │ Backtest Engine    │ │ Live Engine      │ │ Stats Arb Pipeline    │   │
│  │ single/multi-sym   │ │ Alpaca paper     │ │ pairs analysis (17    │   │
│  │ dividend handling  │ │ market-open det. │ │ modules: coint,       │   │
│  │ streaming via WS   │ │ 60s poll loop    │ │ regime, WF, backtest) │   │
│  └────────────────────┘ └──────────────────┘ └────────────────────────┘   │
│  ┌────────────────────┐ ┌──────────────────┐ ┌────────────────────────┐   │
│  │ ML Pipeline        │ │ Strategies       │ │ Alpaca / yfinance /   │   │
│  │ 6 model types,     │ │ 4 strategies:    │ │ SQLite                 │   │
│  │ stacking, grid     │ │ SMA, SimpleStrat,│ │                       │   │
│  │ walk-forward,      │ │ ML, CorrCoint    │ │                       │   │
│  │ auto-optimizer     │ │                  │ │                       │   │
│  └────────────────────┘ └──────────────────┘ └────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
trader/
├── backend/
│   ├── api/
│   │   ├── main.py             FastAPI app entry + 7 routers (routes, ws, live, live_ws, alpaca, ml, pairs)
│   │   ├── routes.py           REST: strategies, backtest CRUD, portfolio
│   │   ├── websocket.py        /ws/backtest — streaming backtest + replay
│   │   ├── live_routes.py      /api/live/status, /api/live/stop
│   │   ├── live_websocket.py   /ws/live — streaming live trading
│   │   ├── alpaca_routes.py    /api/alpaca/* — account, positions, orders, portfolio
│   │   ├── ml_routes.py        /api/ml/* — models, retrain, correlation data
│   │   ├── pairs_routes.py     /api/pairs/* — analyze, rank, heatmap
│   │   ├── models.py           Pydantic models for all API responses
│   │   └── deps.py             Database dependency injection
│   ├── engine/
│   │   ├── __init__.py
│   │   └── live.py             LiveEngine — Alpaca paper trading, market-open detection, WebSocket
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py             Abstract Strategy, Signal enum, Portfolio
│   │   ├── registry.py         STRATEGY_REGISTRY — auto-discovery + instantiation
│   │   ├── sma_crossover.py    SmaCrossover — SMA crossover strategy
│   │   ├── simple_strat_1.py   SimpleStrat1 — mean-reversion DCA
│   │   ├── ml_strategy.py      MLStrategy — loads .joblib model, 38 features
│   │   └── corr_coint_strat.py CorrCointStrategy — pairs mean-reversion
│   ├── ml/                     ML training pipeline
│   │   ├── features.py         38 technical indicator features
│   │   ├── model.py            Versioned .joblib save/load
│   │   ├── train.py            1.5K line CLI trainer (6 model types, grid search, stacking, etc.)
│   │   ├── stacking.py         StackingEnsemble, MetaLabeledModel, MultiHorizonEnsemble, RegimeAwareModel
│   │   ├── auto_optimize.py    10-step forward selection + champion
│   │   ├── logs/               Training logs + active PID tracking
│   │   └── models/             Versioned .joblib model files
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py           BacktestEngine + MultiSymbolBacktestEngine
│   │   └── metrics.py          Sharpe, Sortino, Calmar, drawdown, win rate, profit factor
│   ├── stats_arb/              Statistical arbitrage research framework (17 files)
│   │   ├── config.py, data.py, correlation.py, cointegration.py, hedge_ratio.py
│   │   ├── spread.py, regime.py, walk_forward.py, strategy.py, backtest.py
│   │   ├── ranking.py, ml_models.py, pipeline.py, cli.py, discover.py, visualization.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           Alpaca/yfinance data loader + caching
│   │   └── store.py            SQLite CRUD for runs, snapshots, positions, orders
│   ├── tests/                  50 test files, 742 tests
│   ├── config.py               API keys, DB path, validation
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx             7 tabs: Dashboard, Live, Backtest, Strategies, ML Lab, Correlation, Pairs
│       ├── pages/              Dashboard, Live, Backtest, Strategies, MlLab, Correlation, Pairs
│       ├── components/         AccountSummary, PositionsTable, OrderHistory, PortfolioChart, StrategySelector, Clock
│       ├── api/                ApiClient (24 methods), BacktestSocket, LiveSocket
│       └── theme/              ThemeContext — dark/light, system preference
├── AGENTS.md, README.md, PLAN.md, ML_PLAN.md, ML_AUDIT.md, STAT_ARB_REFACTOR_PLAN.md
```

## Pages (7 tabs)

| Tab | Description |
|---|---|
| **Dashboard** | Alpaca account summary (cash, portfolio value, buying power, P&L), positions table, order history, portfolio equity chart |
| **Live** | Run a strategy live on Alpaca paper — selector, symbol toggles, equity chart, console log |
| **Backtest** | Strategy selector, param editing, date/cash inputs, WS streaming, trade markers, past runs |
| **Strategies** | Browse strategies — cards with name, description, params |
| **ML Lab** | List models with metric cards, retrain form with full flags, delete |
| **Correlation** | Rolling correlation chart, cumulative returns, statistics table |
| **Pairs** | Analyze/rank/heatmap sections, spread/z-score/regime charts, backtest metrics |

## Widgets

| Widget | Description |
|---|---|
| Account Summary Bar | Cash balance, buying power, portfolio value, day P&L |
| Portfolio Equity Curve | Line chart (lightweight-charts) with buy/sell trade markers |
| Positions Table | Symbol, qty, avg entry, current price, P&L |
| Order History | Chronological list of fills with timestamps, prices |
| Strategy Selector | Dropdown + dynamic param inputs from StrategyInfo |
| Clock | Live clock (HH:MM:SS AM/PM + date), updates every 1s |

## Data Flows

### Real-Time Backtest
```
User clicks "Run" on frontend
  → WebSocket connects to /ws/backtest
  → Backend starts BacktestEngine.stream() in background thread
  → Each bar: engine yields {snapshot, trade, signal, dividend}
  → WebSocket relays as JSON {"type": "bar", "data": {...}}
  → Frontend updates equity curve, trade markers, positions, order log
  → On completion: {"type": "complete", "metrics": {...}}
```

### Live Trading
```
Frontend LiveSocket.connect()
  → WebSocket to /ws/live
  → LiveEngine.start() with callbacks
  → Polls Alpaca bars every ~60s
  → Runs per-symbol strategy instances
  → Places MarketOrder via Alpaca API
  → Streams bars/trades via WebSocket
```

### ML Retrain
```
Frontend POST /api/ml/retrain
  → Spawns subprocess: python -m backend.ml.train ...
  → stdout piped to log file (no deadlock)
  → Returns {status: "started", pid: ...}
  → Frontend polls GET /api/ml/models for completion
```

### Pairs Analysis
```
Frontend POST /api/pairs/analyze
  → PairAnalyzer.analyze() full pipeline:
    data → correlation → cointegration → hedge ratio → spread
    → regime → walk-forward → strategy → backtest → (ML)
  → Returns PairAnalysisResponse with all metrics + series
```

## Custom Backtest Engine Design

```python
class Strategy:
    def init(self, data: pd.DataFrame):
        # Called once. Compute indicators here.
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> Signal:
        # Called each bar. Return BUY/SELL/HOLD
        pass

engine = BacktestEngine(data, strategy, initial_cash=10000)
results = engine.run()
# results.metrics: Sharpe, Max DD, Win Rate, Total Return
```

## Build Phases (All Complete ✅)

| Phase | What | Deliverables |
|---|---|---|
| **1** | Foundation | Python project, Alpaca data loader, SQLite schema |
| **2** | Custom Backtest Engine | Strategy base class, bar-by-bar simulation, trade tracking, metrics |
| **3** | Backend API + WebSocket | FastAPI REST routes + WebSocket streaming |
| **4** | React Frontend | 7 tabs, 6 widgets, WebSocket streaming, buy/sell markers, theme toggle, 133 tests |
| **5** | Live Trading Engine | Alpaca paper execution, market-open detection, WebSocket streaming |
| **6** | ML Training Pipeline | yfinance data, 38 features, RF/GBT/XGB/LGB/SGD/MLP, grid search, baselines |
| **7** | Advanced ML | Stacking, multi-horizon, regime-aware, context symbols, auto-optimizer, SGD online learning, MLP deep learning, ML Lab UI |
| **8** | Statistical Arbitrage | Correlation, cointegration (EG + Johansen), hedge ratio (OLS/rolling/Kalman), spread analysis, regime detection (CUSUM/Chow/Bai-Perron), purged walk-forward, pairs backtest, ranking (Bonferroni/Benjamini-Hochberg), auto-discovery, CLI + API + web UI |
