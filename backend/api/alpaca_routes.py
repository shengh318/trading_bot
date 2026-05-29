from fastapi import APIRouter, HTTPException
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

from backend.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

alpaca_router = APIRouter()


def _get_client() -> TradingClient:
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Alpaca API keys not configured")
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


@alpaca_router.get("/api/alpaca/account")
def alpaca_account():
    client = _get_client()
    acct = client.get_account()
    equity = float(acct.equity)
    day_pnl = equity - float(acct.last_equity) if acct.last_equity else 0.0
    return {
        "cash": round(float(acct.cash), 2),
        "portfolio_value": round(equity, 2),
        "buying_power": round(float(acct.buying_power), 2),
        "day_pnl": round(day_pnl, 2),
    }


@alpaca_router.get("/api/alpaca/positions")
def alpaca_positions():
    client = _get_client()
    positions = client.get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": round(float(p.market_value), 2),
            "avg_entry_price": round(float(p.avg_entry_price), 2),
            "current_price": round(float(p.current_price), 2),
            "unrealized_pl": round(float(p.unrealized_pl), 2),
            "unrealized_plpc": round(float(p.unrealized_plpc) * 100, 2),
        }
        for p in positions
    ]


@alpaca_router.get("/api/alpaca/orders")
def alpaca_orders(limit: int = 25):
    client = _get_client()
    req = GetOrdersRequest(limit=limit, status="all")
    orders = client.get_orders(req)
    return [
        {
            "id": str(o.id),
            "symbol": o.symbol,
            "side": o.side.value if hasattr(o.side, "value") else str(o.side),
            "type": o.type.value if hasattr(o.type, "value") else str(o.type),
            "qty": float(o.qty) if o.qty else 0,
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
            "filled_avg_price": round(float(o.filled_avg_price), 2) if o.filled_avg_price else None,
            "status": o.status.value if hasattr(o.status, "value") else str(o.status),
            "created_at": o.created_at.isoformat() if o.created_at else "",
            "updated_at": o.updated_at.isoformat() if o.updated_at else "",
        }
        for o in orders
    ]


@alpaca_router.get("/api/alpaca/portfolio-history")
def alpaca_portfolio_history(period: str = "1M", timeframe: str = "1D"):
    client = _get_client()
    params = {"period": period, "timeframe": timeframe}
    try:
        history = client.get("/account/portfolio/history", params)
    except Exception:
        return []
    timestamps = history.get("timestamp", []) if isinstance(history, dict) else []
    equities = history.get("equity", []) if isinstance(history, dict) else []
    return [
        {
            "timestamp": ts,
            "equity": round(equity, 2),
        }
        for ts, equity in zip(timestamps, equities)
    ]
