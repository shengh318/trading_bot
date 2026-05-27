# TraderBot — Full Architecture Plan

## Tech Stack
| Component | Choice |
|---|---|
| Backend / Strategies | Python 3.11+ |
| Backtesting | Custom mini-engine (pandas-based) |
| API Framework | FastAPI |
| Database | SQLite (dev) |
| Frontend | React + TypeScript + Vite |
| Charting | TradingView Lightweight Charts |
| Broker API | Alpaca (paper + live) |
| Real-time | WebSocket (backtest streaming) |

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /dashboard (Live Mode)                              │   │
│  │  ┌──────────────┐ ┌──────────────┐                   │   │
│  │  │ Account Bar  │ │ Strategy     │                   │   │
│  │  │ Cash: $xx    │ │ Selector     │                   │   │
│  │  └──────────────┘ └──────────────┘                   │   │
│  │  ┌──────────────────────────────────┐                │   │
│  │  │   Portfolio Equity Curve (Chart) │                │   │
│  │  └──────────────────────────────────┘                │   │
│  │  ┌──────────────┐ ┌────────────────────────┐         │   │
│  │  │ Positions    │ │ Order History           │         │   │
│  │  └──────────────┘ └────────────────────────┘         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /backtest (Backtest View)                           │   │
│  │  Same layout but shows backtest simulation running   │   │
│  │  in real-time with streaming updates via WebSocket   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
           ▲  HTTP (dashboard data)         ▲ WebSocket (backtest events)
           │                                │
┌──────────┴────────────────────────────────┴──────────────────┐
│                    FastAPI Backend                            │
│  ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐   │
│  │ /api/...     │ │ WebSocket      │ │ Trading Engine    │   │
│  │ REST routes  │ │ /ws/backtest   │ │ (background)     │   │
│  └──────────────┘ └────────────────┘ └──────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Backtest Engine                                      │   │
│  │  for bar in data:  ──WebSocket msg──>  update chart  │   │
│  │    strategy.next()                                    │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Alpaca Client / SQLite                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
trader/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # FastAPI endpoints
│   │   ├── models.py           # Pydantic response models
│   │   ├── websocket.py        # WebSocket handlers
│   │   └── main.py             # FastAPI app entry
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── live.py             # Alpaca live trading loop
│   │   ├── portfolio.py        # Position/risk management
│   │   └── paper.py            # Paper trading wrapper
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract strategy class
│   │   ├── sma_crossover.py    # Example strategy
│   │   └── ...
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py           # Core backtest simulation
│   │   ├── metrics.py          # Sharpe, drawdown, win rate
│   │   └── reporter.py         # Pretty results output
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Fetch from Alpaca / cache
│   │   └── store.py            # SQLite schema & operations
│   ├── requirements.txt
│   └── config.py               # API keys, settings
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx    # Portfolio overview
│   │   │   ├── Backtest.tsx     # Run/view backtests
│   │   │   └── Strategies.tsx   # Manage strategies
│   │   ├── components/
│   │   │   ├── PortfolioChart.tsx
│   │   │   ├── PositionsTable.tsx
│   │   │   ├── OrderHistory.tsx
│   │   │   ├── AccountSummary.tsx
│   │   │   └── StrategySelector.tsx
│   │   ├── api/
│   │   │   └── client.ts       # API + WebSocket client
│   │   └── main.tsx
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
└── PLAN.md
```

## Dashboard Widgets

| Widget | Description |
|---|---|
| Account Summary Bar | Cash balance, buying power, portfolio value, day P&L |
| Portfolio Equity Curve | Line chart of total portfolio value over time (lightweight-charts) |
| Positions Table | Symbol, qty, avg entry, current price, P&L |
| Order History | Chronological list of all fills with timestamps, prices |
| Strategy Selector | Dropdown to switch between strategies + parameters |

## Data Flow — Real-Time Backtest Visualization

```
User clicks "Run" on frontend
  → WebSocket connects to /ws/backtest/{session_id}
  → Backend starts engine in background thread
  → On each bar:
       engine processes → yields (timestamp, equity, holdings, trades)
       → WebSocket sends JSON to frontend
       → Frontend updates equity curve (append point),
         positions table (replace), order log (append row)
  → When done: sends final metrics (Sharpe, DD, win rate)
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

## Build Phases

| Phase | What | Deliverables |
|---|---|---|
| **1** | Foundation | Python project, Alpaca data loader, SQLite schema |
| **2** | Custom Backtest Engine | Strategy base class, bar-by-bar simulation, trade tracking, metrics |
| **3** | Backend API + WebSocket | FastAPI REST routes + WebSocket streaming |
| **4** | React Frontend | Two tabs (Dashboard/Backtest), streaming charts, all widgets |
| **5** | Live Trading Engine | Paper/live Alpaca loop, same strategy code, same WebSocket format |
| **6** | Your Strategies | Build and tune actual strategies |
