from typing import Any, Optional

from pydantic import BaseModel


class MlModelInfo(BaseModel):
    name: str
    version: int
    model_type: str
    train_date: str
    train_symbols: list[str]
    context_symbols: list[str] = []
    validation_metrics: dict
    beat_baselines: bool
    versions: list[int]


class MlRetrainRequest(BaseModel):
    symbols: str = "NVDA,AMD,VOO,SPY,META"
    years: int = 20
    name: str = "multi_symbol_model"
    model_types: str = "rf,gbt"
    beat_baselines: bool = False
    grid_search: bool = False
    walk_forward: int = 0
    stacking: bool = False
    meta_labeling: bool = False
    regularize: bool = False
    prune: float = 0.0
    kelly: bool = False
    auto_threshold: bool = False
    labeling: str = "next_bar"
    forecast_horizon: int = 1
    context_symbols: str | None = None
    multi_horizon: str | None = None
    regime_aware: bool = False
    embargo: int = 5
    cutoff_date: str | None = None
    val_split: float = 0.8
    triple_barrier_pct: float = 0.02
    triple_barrier_max_bars: int = 10
    n_estimators: int = 200
    max_depth: int = 10
    learning_rate: float = 0.1
    confidence_threshold: float = 0.55
    base_buy_size: float = 1000.0
    model_dir: str | None = None


class MlRetrainResponse(BaseModel):
    status: str
    pid: int | None = None
    message: str = ""


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


def _resolve_profit_factor(row: dict) -> float:
    pf = row.get("profit_factor")
    if pf is not None:
        return pf
    num_trades = row.get("num_trades", 0) or 0
    win_rate = row.get("win_rate", 0) or 0
    if num_trades > 0 and win_rate == 100.0:
        return float("inf")
    return 0.0


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
            profit_factor=_resolve_profit_factor(row),
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
