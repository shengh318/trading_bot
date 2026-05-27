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
    end_date: str
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
