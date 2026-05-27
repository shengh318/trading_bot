import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from backend.engine.live import LiveEngine, set_live_engine, get_live_engine
from backend.strategies.base import Signal

live_ws_router = APIRouter()


def _parse_timeframe(tf_str: str) -> TimeFrame:
    import re
    tf_str = tf_str.strip()
    m = re.match(r"^(\d+)\s*(Min|Hour|Day|Week|Month)$", tf_str, re.IGNORECASE)
    if not m:
        return TimeFrame.Day
    amount = int(m.group(1))
    unit = m.group(2).lower()
    unit_map = {
        "min": TimeFrameUnit.Minute,
        "hour": TimeFrameUnit.Hour,
        "day": TimeFrameUnit.Day,
        "week": TimeFrameUnit.Week,
        "month": TimeFrameUnit.Month,
    }
    return TimeFrame(amount, unit_map[unit])


@live_ws_router.websocket("/ws/live")
async def live_websocket(websocket: WebSocket):
    await websocket.accept()
    engine: LiveEngine | None = None

    try:
        data = await websocket.receive_json()
        action = data.get("action")

        if action != "start":
            await websocket.send_json({"type": "error", "message": "First message must be 'start'"})
            await websocket.close()
            return

        existing = get_live_engine()
        if existing and existing.running:
            existing.stop()

        symbols: list[str] = data.get("symbols") or [data["symbol"]]
        timeframe = data.get("timeframe", "1Day")

        async def on_initial_equity(equity: float, cash: float, timestamp: str) -> None:
            try:
                await websocket.send_json({
                    "type": "bar",
                    "bar_index": 0,
                    "timestamp": timestamp,
                    "equity": equity,
                    "cash": cash,
                    "trades": [],
                })
            except Exception:
                pass

        async def on_trades(trades: list[dict], timestamp: str) -> None:
            try:
                equity = engine.portfolio.cash + sum(
                    engine.portfolio.positions.get(sym, 0) * float(
                        engine.data_buffers[sym].iloc[-1]["close"]
                    ) if sym in engine.data_buffers and not engine.data_buffers[sym].empty else 0
                    for sym in engine.symbols
                )
                await websocket.send_json({
                    "type": "bar",
                    "bar_index": engine._bar_count,
                    "timestamp": timestamp,
                    "equity": round(equity, 2),
                    "cash": round(engine.portfolio.cash, 2),
                    "trades": trades,
                })
            except Exception:
                pass

        async def on_status(status: str, message: str) -> None:
            try:
                await websocket.send_json({"type": "status", "status": status, "message": message})
            except Exception:
                pass

        async def on_error(message: str) -> None:
            try:
                await websocket.send_json({"type": "error", "message": message})
            except Exception:
                pass

        engine = LiveEngine(
            strategy_name=data["strategy_name"],
            parameters=data.get("parameters"),
            symbols=symbols,
            timeframe_str=timeframe,
        )
        set_live_engine(engine)
        await engine.start({
            "on_initial_equity": on_initial_equity,
            "on_trades": on_trades,
            "on_status": on_status,
            "on_error": on_error,
        })

        while engine.running:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=1)
                if msg.get("action") == "stop":
                    engine.stop()
                    break
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if engine and engine.running:
            engine.stop()
        try:
            await websocket.close()
        except Exception:
            pass
