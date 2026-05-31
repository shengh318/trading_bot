import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException

from backend.api.deps import get_db
from backend.api.models import (
    AccountSummary,
    BacktestMetrics,
    BacktestRunRequest,
    BacktestRunResponse,
    OrderResponse,
    PositionResponse,
    StrategyInfo,
    db_row_to_backtest_run_response,
)
# BacktestEngine and DataLoader are lazy-imported inside route handlers
from backend.strategies.registry import get_strategy, list_strategies

router = APIRouter()


@router.get("/api/strategies", response_model=list[StrategyInfo])
def get_strategies():
    return list_strategies()


@router.get("/api/portfolio/summary", response_model=AccountSummary)
def get_portfolio_summary():
    db = get_db()
    curve = db.get_equity_curve(limit=1)
    value = curve[0]["total_equity"] if curve else 0.0
    cash = curve[0]["cash"] if curve else 0.0
    return AccountSummary(
        cash=cash,
        portfolio_value=value,
        buying_power=cash * 2,
        day_pnl=0.0,
    )


@router.get("/api/portfolio/equity-curve")
def get_equity_curve(limit: int = 500):
    db = get_db()
    return db.get_equity_curve(limit=limit)


@router.get("/api/positions", response_model=list[PositionResponse])
def get_positions():
    db = get_db()
    return db.get_positions()


@router.get("/api/orders", response_model=list[OrderResponse])
def get_orders(limit: int = 100):
    db = get_db()
    return db.get_orders(limit=limit)


@router.get("/api/backtest/runs", response_model=list[BacktestRunResponse])
def get_backtest_runs(limit: int = 20):
    db = get_db()
    rows = db.get_backtest_runs(limit=limit)
    return [db_row_to_backtest_run_response(r) for r in rows]


@router.get("/api/backtest/runs/{run_id}", response_model=BacktestRunResponse)
def get_backtest_run(run_id: int):
    db = get_db()
    row = db.get_backtest_run_by_id(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return db_row_to_backtest_run_response(row)


@router.get("/api/backtest/runs/{run_id}/trades")
def get_backtest_trades(run_id: int):
    db = get_db()
    return db.get_backtest_trades(run_id)


@router.get("/api/backtest/runs/{run_id}/equity")
def get_backtest_equity(run_id: int):
    db = get_db()
    return db.get_backtest_snapshots(run_id)


@router.delete("/api/backtest/runs")
def delete_backtest_runs():
    db = get_db()
    db.delete_backtest_runs()
    return {"status": "ok"}

@router.delete("/api/backtest/runs/{run_id}")
def delete_backtest_run(run_id: int):
    db = get_db()
    db.delete_backtest_run(run_id)
    return {"status": "ok"}


@router.post("/api/backtest/run", response_model=BacktestRunResponse)
def run_backtest(req: BacktestRunRequest):
    from backend.backtest.engine import BacktestEngine
    from backend.data.loader import DataLoader
    db = get_db()

    try:
        strategy = get_strategy(req.strategy_name, req.parameters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        loader = DataLoader()
        start = datetime.fromisoformat(req.start_date)
        end = datetime.fromisoformat(req.end_date) if req.end_date else (datetime.now(timezone.utc).replace(day=1) - timedelta(days=1))
        df = loader.load_bars(
            symbol=req.symbol,
            start=start,
            end=end,
        )
        div_df = loader.load_dividends(
            symbol=req.symbol,
            start=start,
            end=end,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load data: {e}")

    engine = BacktestEngine(
        df, strategy, symbol=req.symbol, initial_cash=req.initial_cash, dividends=div_df,
    )
    result = engine.run()

    metrics = result.metrics
    run_id = db.save_backtest_run({
        "strategy_name": req.strategy_name,
        "parameters": json.dumps(req.parameters) if req.parameters else None,
        "symbol": req.symbol,
        "start_date": req.start_date,
        "end_date": req.end_date or end.date().isoformat(),
        "initial_cash": req.initial_cash,
        "final_equity": metrics["final_equity"],
        "total_return": metrics["total_return_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown_pct"],
        "win_rate": metrics["win_rate_pct"],
        "num_trades": metrics["num_trades"],
        "profit_factor": metrics["profit_factor"],
    })

    if not result.trades.empty:
        db.save_backtest_trades(run_id, result.trades.to_dict("records"))
    if not result.equity_curve.empty:
        db.save_backtest_snapshots(run_id, result.equity_curve.to_dict("records"))

    row = db.get_backtest_run_by_id(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Backtest run not found after save")
    return db_row_to_backtest_run_response(row)
