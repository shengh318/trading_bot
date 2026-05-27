# TraderBot

Simulation tool to test stock trading strategies — no account needed.

## Architecture

```
Frontend (React + Vite)                           ──►  Portfolio chart, holdings,
  ▲  HTTP REST + WebSocket                              order history, strategy setup
  │
Backend (FastAPI + Python)                        ──►  API layer + simulation engine
  │
  ├── Alpaca Broker API (optional)                ──►  Real market data
  ├── Custom Simulation Engine                    ──►  Bar-by-bar simulation
  └── SQLite                                       ──►  Trades, snapshots, run history
```

## Project Structure

```
trader/
├── backend/
│   ├── api/              FastAPI routes, WebSocket, models, deps
│   │   ├── main.py       FastAPI app entry + lifespan
│   │   ├── routes.py     REST endpoints (portfolio, holdings, orders, simulation)
│   │   ├── websocket.py  WebSocket handler (live simulation streaming)
│   │   ├── models.py     Pydantic response models
│   │   └── deps.py       Shared DB dependency
│   ├── strategies/       Strategy base class + implementations
│   │   ├── base.py       Abstract Strategy + Signal + Portfolio
│   │   ├── registry.py   Strategy registry (discovery + instantiation)
│   │   └── sma_crossover.py
│   ├── backtest/         Custom simulation engine + metrics
│   │   ├── engine.py     Bar-by-bar simulation
│   │   └── metrics.py    Sharpe, drawdown, win rate, etc.
│   ├── data/             Alpaca data loader & SQLite store
│   ├── tests/            71 pytest tests covering all modules
│   └── config.py         Settings & env vars
├── frontend/
│   └── src/
│       ├── pages/        Dashboard, Simulation, Strategies views
│       ├── components/   Charts, tables, account bar widgets
│       └── api/          HTTP + WebSocket client
└── PLAN.md               Full architecture & build phases
```

## Setup

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

Copy `.env.example` to `.env` and fill in your Alpaca API keys for real market data.
The app works without it — it will tell you when Alpaca data isn't available.

```bash
cp .env.example .env
```

## Running

```bash
# Terminal 1 — Backend
.venv/bin/uvicorn backend.api.main:app --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open http://localhost:5173 in your browser.

## Testing

```bash
# Backend tests (71 tests)
.venv/bin/python -m pytest backend/tests/ -v --tb=short

# Frontend tests (34 tests)
cd frontend && npm test
```

## What You Can Do

| Page | What it does |
|---|---|
| **Dashboard** | See your account summary, holdings, order history, and equity chart |
| **Simulation** | Pick a strategy + stock + date range, run a simulation with buy/sell markers on the chart, replay past runs, clear history |
| **Strategies** | Browse available strategies and their settings |

### Simulation tips
- Pick a **start date** — it runs through the last full month automatically
- Enter a **starting amount** (like $100)
- Click **Run Simulation** — watch the chart and buy/sell markers animate in real time

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [x] Phase 2: Custom Simulation Engine — bar-by-bar simulation, P&L tracking, equity curves, metrics
- [x] Phase 3: Backend API + WebSocket — REST endpoints for portfolio/holdings/orders/simulation, WebSocket for real-time simulation streaming, strategy registry
- [x] Phase 4: React Frontend Dashboard — 3 pages (Dashboard, Simulation, Strategies), 5 widgets, WebSocket streaming with buy/sell markers, fractional shares, 34 frontend tests
- [ ] Phase 5: Live Trading Engine
- [ ] Phase 6: Strategy building & tuning
