import json
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.deps import get_db
from backend.backtest.engine import BacktestEngine
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
    end = datetime.fromisoformat(data["end_date"]) if data.get("end_date") else datetime.now(timezone.utc)
    df = loader.load_bars(
        symbol=data["symbol"],
        start=datetime.fromisoformat(data["start_date"]),
        end=end,
    )

    engine = BacktestEngine(
        df, strategy,
        symbol=data["symbol"],
        initial_cash=float(data.get("initial_cash", 10000)),
    )

    trades_batch: list[dict] = []
    snapshots_batch: list[dict] = []

    for event in engine.stream():
        msg = {
            "type": "bar",
            "bar_index": event["snapshot"]["bar_index"],
            "timestamp": event["snapshot"]["timestamp"],
            "equity": event["snapshot"]["equity"],
            "cash": event["snapshot"]["cash"],
            "signal": event["signal"],
            "trade": event["trade"],
        }
        await websocket.send_json(msg)

        if event["trade"]:
            trades_batch.append(event["trade"])
        snapshots_batch.append(event["snapshot"])

    db = get_db()
    run_id = db.save_backtest_run({
        "strategy_name": data["strategy_name"],
        "parameters": json.dumps(data.get("parameters")),
        "symbol": data["symbol"],
        "start_date": data["start_date"],
        "end_date": data.get("end_date") or end.date().isoformat(),
        "initial_cash": float(data.get("initial_cash", 10000)),
    })

    if trades_batch:
        db.save_backtest_trades(run_id, trades_batch)
    if snapshots_batch:
        db.save_backtest_snapshots(run_id, snapshots_batch)

    runs = db.get_backtest_runs(limit=1)
    match = [r for r in runs if r["id"] == run_id]

    await websocket.send_json({
        "type": "complete",
        "run_id": run_id,
        "metrics": match[0] if match else None,
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

    await websocket.send_json({"type": "complete", "run_id": run_id})
