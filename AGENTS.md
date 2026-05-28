# TraderBot — Agent Instructions

## Commands

### Windows (PowerShell)
- **Activate venv**: `.venv\Scripts\Activate.ps1`
- **Run backend**: `.venv\Scripts\uvicorn backend.api.main:app --reload`
- **Run frontend**: `cd frontend; npm run dev`
- **Install backend deps**: `.venv\Scripts\pip install -r backend\requirements.txt`
- **Install frontend deps**: `cd frontend; npm install`
- **Check types**: `cd frontend; npx tsc --noEmit`
- **Lint**: `cd frontend; npx eslint src/`
- **Train ML model**: `.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20`
- **Train with grid search (auto-find best params)**: `.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search`
- **Train + only save if it beats baselines**: `.venv\Scripts\python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --beat-baselines`

### macOS / Linux
- **Activate venv**: `source .venv/bin/activate`
- **Run backend**: `.venv/bin/uvicorn backend.api.main:app --reload`
- **Run frontend**: `cd frontend && npm run dev`
- **Install backend deps**: `.venv/bin/pip install -r backend/requirements.txt`
- **Install frontend deps**: `cd frontend && npm install`
- **Check types**: `cd frontend && npx tsc --noEmit`
- **Lint**: `cd frontend && npx eslint src/`
- **Train ML model**: `.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20`
- **Train with grid search**: `.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --grid-search`
- **Train + beat baselines**: `.venv/bin/python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20 --beat-baselines`

## Architecture Quick Reference

- **Backend**: FastAPI at `backend/api/main.py`
- **Frontend**: React + Vite at `frontend/src/`
- **Strategies**: Python classes in `backend/strategies/`, extend `base.py`
- **Backtest engine**: `backend/backtest/engine.py`, streams via WebSocket
- **DB**: SQLite (auto-created on first run)
- **Broker**: Alpaca (keys in `.env`)
- **ML training**: `backend/ml/train.py` — CLI script, uses yfinance data, saves models to `backend/ml/models/`
- **ML strategy**: `MLStrategy` — loads pre-trained model from disk, uses 24 technical indicator features

## Conventions

- Type hints required on all Python functions
- Strategies extend `Strategy` base class with `init()` and `next()`
- Frontend uses TypeScript strictly
- All API routes return Pydantic models
- No committing unless explicitly asked
