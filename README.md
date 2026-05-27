# TraderBot

Algorithmic day trading bot with a custom backtest engine, live Alpaca trading, and a real-time React dashboard.

## Architecture

```
Frontend (React + Vite)                           ──►  Portfolio chart, positions,
  ▲  HTTP REST + WebSocket                              order history, strategy config
  │
Backend (FastAPI + Python)                        ──►  API layer + trading engine
  │
  ├── Alpaca Broker API                           ──►  Live/paper order execution
  ├── Custom Backtest Engine                      ──►  Bar-by-bar simulation
  └── SQLite                                       ──►  Trades, snapshots, backtest results
```

## Project Structure

```
trader/
├── backend/
│   ├── api/              FastAPI routes, WebSocket, models, deps
│   │   ├── main.py       FastAPI app entry + lifespan
│   │   ├── routes.py     REST endpoints (portfolio, positions, orders, backtest)
│   │   ├── websocket.py  WebSocket handler (live backtest streaming)
│   │   ├── models.py     Pydantic response models
│   │   └── deps.py       Shared DB dependency
│   ├── engine/           Live/paper trading loop
│   ├── strategies/       Strategy base class + implementations
│   │   ├── base.py       Abstract Strategy + Signal + Portfolio
│   │   ├── registry.py   Strategy registry (discovery + instantiation)
│   │   └── sma_crossover.py
│   ├── backtest/         Custom backtest engine + metrics
│   │   ├── engine.py     Bar-by-bar simulation
│   │   └── metrics.py    Sharpe, drawdown, win rate, etc.
│   ├── data/             Alpaca data loader & SQLite store
│   ├── tests/            71 pytest tests covering all modules
│   └── config.py         Settings & env vars
├── frontend/
│   └── src/
│       ├── pages/        Dashboard, Backtest, Strategies views
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
```

### Configuration

Copy `.env.example` to `.env` and fill in your Alpaca API keys:

```bash
cp .env.example .env
```

Get free paper trading keys at [alpaca.markets](https://alpaca.markets).

## Running

```bash
# Backend
.venv/bin/uvicorn backend.api.main:app --reload

# Frontend (separate terminal)
cd frontend && npm run dev
```

## Testing

All tests use `pytest` with isolated databases per test via the `api_db` fixture.

```bash
.venv/bin/python -m pytest backend/tests/ -v --tb=short
```

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [x] Phase 2: Custom Backtest Engine — bar-by-bar simulation, P&L tracking, equity curves, metrics
- [x] Phase 3: Backend API + WebSocket — REST endpoints for portfolio/positions/orders/backtest, WebSocket for real-time backtest streaming, strategy registry
- [x] Phase 4: React Frontend Dashboard — 3 pages (Dashboard, Backtest, Strategies), 5 widgets, WebSocket streaming, 34 frontend tests
- [ ] Phase 5: Live Trading Engine
- [ ] Phase 6: Strategy building & tuning
