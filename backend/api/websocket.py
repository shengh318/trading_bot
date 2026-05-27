import json
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.deps import get_db
from backend.backtest.engine import BacktestEngine
from backend.backtest.metrics import calculate_metrics
from backend.data.loader import DataLoader
from backend.strategies.registry import get_strategy

ws_router = APIRouter()


@ws_router.websocket("/ws/backtest")
async def backtest_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        action = data.get("action", "run")

        if action == "run":
            await _handle_run(websocket, data)
        elif action == "replay":
            run_id = data.get("run_id")
            if run_id is None:
                await websocket.send_json({"type": "error", "message": "Missing run_id"})
                return
            await _handle_replay(websocket, int(run_id))
        else:
            await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except WebSocketDisconnect:
            pass


async def _handle_run(websocket: WebSocket, data: dict) -> None:
    strategy = get_strategy(data["strategy_name"], data.get("parameters"))

    loader = DataLoader()
    end = datetime.fromisoformat(data["end_date"]) if data.get("end_date") else (datetime.now(timezone.utc).replace(day=1) - timedelta(days=1))
    start = datetime.fromisoformat(data["start_date"])
    df = loader.load_bars(
        symbol=data["symbol"],
        start=start,
        end=end,
    )
    div_df = loader.load_dividends(
        symbol=data["symbol"],
        start=start,
        end=end,
    )

    engine = BacktestEngine(
        df, strategy,
        symbol=data["symbol"],
        initial_cash=float(data.get("initial_cash", 10000)),
        dividends=div_df,
    )

    trades_batch: list[dict] = []
    snapshots_batch: list[dict] = []

    for event in engine.stream():
        msg: dict = {
            "type": "bar",
            "bar_index": event["snapshot"]["bar_index"],
            "timestamp": event["snapshot"]["timestamp"],
            "equity": event["snapshot"]["equity"],
            "cash": event["snapshot"]["cash"],
            "signal": event["signal"],
            "trade": event["trade"],
        }
        if event.get("dividend"):
            msg["dividend"] = event["dividend"]

        await websocket.send_json(msg)

        if event["trade"]:
            trades_batch.append(event["trade"])
        snapshots_batch.append(event["snapshot"])

    db = get_db()
    initial_cash = float(data.get("initial_cash", 10000))

    snapshots_df = pd.DataFrame(snapshots_batch) if snapshots_batch else pd.DataFrame(columns=["equity"])
    trades_df = pd.DataFrame(trades_batch) if trades_batch else pd.DataFrame(columns=["side", "pnl"])
    metrics = calculate_metrics(snapshots_df, trades_df, initial_cash)

    run_id = db.save_backtest_run({
        "strategy_name": data["strategy_name"],
        "parameters": json.dumps(data.get("parameters")),
        "symbol": data["symbol"],
        "start_date": data["start_date"],
        "end_date": data.get("end_date") or end.date().isoformat(),
        "initial_cash": initial_cash,
        "final_equity": metrics["final_equity"],
        "total_return": metrics["total_return_pct"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown_pct"],
        "win_rate": metrics["win_rate_pct"],
        "num_trades": metrics["num_trades"],
        "profit_factor": metrics["profit_factor"],
    })

    if trades_batch:
        db.save_backtest_trades(run_id, trades_batch)
    if snapshots_batch:
        db.save_backtest_snapshots(run_id, snapshots_batch)

    await websocket.send_json({
        "type": "complete",
        "run_id": run_id,
        "metrics": metrics,
    })


async def _handle_replay(websocket: WebSocket, run_id: int) -> None:
    db = get_db()
    snapshots = db.get_backtest_snapshots(run_id)
    trades = db.get_backtest_trades(run_id)

    if not snapshots:
        await websocket.send_json({"type": "error", "message": f"Run {run_id} not found"})
        return

    trade_map = {t["bar_index"]: t for t in trades}

    for snap in snapshots:
        await websocket.send_json({
            "type": "bar",
            "bar_index": snap["bar_index"],
            "timestamp": snap["timestamp"],
            "equity": snap["equity"],
            "cash": snap["cash"],
            "signal": "",
            "trade": trade_map.get(snap["bar_index"]),
        })

    run_row = db.get_backtest_run_by_id(run_id)
    metrics = None
    if run_row and run_row.get("final_equity") is not None:
        row = run_row
        metrics = {
            "total_return_pct": row.get("total_return", 0) or 0,
            "final_equity": row.get("final_equity", 0) or 0,
            "sharpe_ratio": row.get("sharpe_ratio", 0) or 0,
            "max_drawdown_pct": row.get("max_drawdown", 0) or 0,
            "win_rate_pct": row.get("win_rate", 0) or 0,
            "num_trades": row.get("num_trades", 0) or 0,
            "profit_factor": row.get("profit_factor", 0) or 0,
        }

    await websocket.send_json({
        "type": "complete",
        "run_id": run_id,
        "metrics": metrics,
    })
