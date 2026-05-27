from typing import Any, Optional

from pydantic import BaseModel


class AccountSummary(BaseModel):
    cash: float = 0.0
    portfolio_value: float = 0.0
    buying_power: float = 0.0
    day_pnl: float = 0.0


class PositionResponse(BaseModel):
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    market_value: float


class OrderResponse(BaseModel):
    id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    filled_avg_price: Optional[float] = None
    status: str
    type: str
    created_at: str
    updated_at: str


class StrategyParamInfo(BaseModel):
    name: str
    type: str
    default: Any


class StrategyInfo(BaseModel):
    name: str
    description: str
    params: list[StrategyParamInfo]


class BacktestRunRequest(BaseModel):
    strategy_name: str
    symbol: str
    start_date: str
    end_date: Optional[str] = None
    initial_cash: float = 10000.0
    parameters: Optional[dict[str, Any]] = None


class BacktestMetrics(BaseModel):
    total_return_pct: float
    final_equity: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    profit_factor: float


class BacktestRunResponse(BaseModel):
    id: int
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_cash: float
    metrics: Optional[BacktestMetrics] = None
    created_at: str


def db_row_to_backtest_run_response(row: dict) -> BacktestRunResponse:
    metrics = None
    if row.get("final_equity") is not None:
        metrics = BacktestMetrics(
            total_return_pct=row.get("total_return", 0) or 0,
            final_equity=row.get("final_equity", 0) or 0,
            sharpe_ratio=row.get("sharpe_ratio", 0) or 0,
            max_drawdown_pct=row.get("max_drawdown", 0) or 0,
            win_rate_pct=row.get("win_rate", 0) or 0,
            num_trades=row.get("num_trades", 0) or 0,
            profit_factor=row.get("profit_factor", 0) or 0,
        )
    return BacktestRunResponse(
        id=row["id"],
        strategy_name=row["strategy_name"],
        symbol=row["symbol"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        initial_cash=row["initial_cash"],
        metrics=metrics,
        created_at=row["created_at"],
    )
