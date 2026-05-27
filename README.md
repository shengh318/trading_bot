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
│   ├── api/              FastAPI routes & WebSocket
│   ├── engine/           Live/paper trading loop
│   ├── strategies/       Strategy base class + implementations
│   ├── backtest/         Custom backtest engine + metrics
│   ├── data/             Alpaca data loader & SQLite store
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
# Activate virtual environment
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend && npm install
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

## Build Progress

- [x] Phase 1: Foundation — SQLite schema, Alpaca data loader, config
- [ ] Phase 2: Custom Backtest Engine
- [ ] Phase 3: Backend API + WebSocket
- [ ] Phase 4: React Frontend Dashboard
- [ ] Phase 5: Live Trading Engine
- [ ] Phase 6: Strategy building & tuning
