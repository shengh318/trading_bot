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
| Auto-optimize + champion | `.venv\Scripts\python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20` | `.venv/bin/python -m backend.ml.auto_optimize --symbols NVDA,AMD,VOO,SPY,META --years 20` |

## Quick Reference
- **API**: FastAPI at `backend/api/main.py`
- **Frontend**: React + Vite at `frontend/src/`
- **Strategies**: `backend/strategies/*.py`, extend `base.py` → `Strategy` with `init()` + `next()`
- **Backtest**: `backend/backtest/engine.py`, streams via WebSocket
- **DB**: SQLite (auto-created on first run)
- **Broker**: Alpaca (keys in `.env`)
- **ML train**: `backend/ml/train.py` — CLI, yfinance data, saves to `backend/ml/models/`
- **ML auto-optimize**: `backend/ml/auto_optimize.py` — forward-selection search + champion training
- **ML strategy**: `backend/strategies/ml_strategy.py` — loads pre-trained model, 38 features
- **Tests**: `backend/tests/` (209 pytest tests), `frontend/src/` (41 vitest tests)

## Conventions
- Type hints required on all Python functions
- Frontend uses strict TypeScript
- All API routes return Pydantic models
- No committing unless explicitly asked
