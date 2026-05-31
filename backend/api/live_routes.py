from fastapi import APIRouter, HTTPException

# LiveEngine is lazy-imported inside route handlers for fast startup

live_router = APIRouter()


@live_router.get("/api/live/status")
def live_status():
    from backend.engine.live import get_live_engine
    engine = get_live_engine()
    if engine is None or not engine.running:
        return {
            "running": False,
            "strategy_name": None,
            "symbols": [],
            "timeframe": None,
            "cash": 0.0,
            "equity": 0.0,
            "bar_count": 0,
        }
    equity = engine.portfolio.cash + sum(
        engine.portfolio.positions.get(sym, 0) * float(
            engine.data_buffers[sym].iloc[-1]["close"]
        ) if sym in engine.data_buffers and not engine.data_buffers[sym].empty else 0
        for sym in engine.symbols
    )
    return {
        "running": engine.running,
        "strategy_name": engine.strategy_name,
        "symbols": engine.symbols,
        "timeframe": engine.timeframe_str,
        "cash": round(engine.portfolio.cash, 2),
        "equity": round(equity, 2),
        "bar_count": engine._bar_count,
    }


@live_router.post("/api/live/stop")
def live_stop():
    from backend.engine.live import get_live_engine
    engine = get_live_engine()
    if engine and engine.running:
        engine.stop()
    return {"status": "stopped"}
