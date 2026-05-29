from typing import Any, Optional
from datetime import date

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


class CorrelationPoint(BaseModel):
    time: str
    value: float


class CorrelationDataResponse(BaseModel):
    symbol_a: str
    symbol_b: str
    correlations: dict[str, list[CorrelationPoint]]
    cumulative_returns: dict[str, list[CorrelationPoint]]
    statistics: dict[str, float]


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


# ── Pairs Trading Models ─────────────────────────────────────────────────


class PairAnalysisRequest(BaseModel):
    ticker_a: str
    ticker_b: str
    start: str = "2015-01-01"
    end: str | None = None
    significance: float = 0.05
    run_johansen: bool = True


class CorrelationMetrics(BaseModel):
    window: int
    current: float
    mean: float
    std: float
    drift: float = 0.0
    regime_changes: int = 0
    threshold_crossings: dict[str, int] = {}
    correlation_collapse: bool = False
    overall_slope: float = 0.0
    max_correlation_drawdown: float = 0.0
    stability_score: float = 0.0


class CointegrationResultModel(BaseModel):
    method: str
    p_value: float
    test_statistic: float
    critical_values: dict[str, float]
    hedge_ratio: float
    hedge_ratio_intercept: float = 0.0
    hedge_ratio_method: str = "ols"
    is_cointegrated: bool


class JohansenResultModel(BaseModel):
    is_cointegrated: bool
    trace_statistic: float
    trace_critical_values: dict[str, float]
    eigenvalue_statistic: float
    eigenvalue_critical_values: dict[str, float]


class SpreadResultModel(BaseModel):
    mean: float
    std: float
    current_zscore: float
    half_life: float
    hurst_exponent: float
    adf_statistic: float = 0.0
    adf_pvalue: float = 1.0
    is_stationary: bool = False
    mean_reversion_speed: float = 0.0
    expected_time_to_mean: float = 0.0
    persistence: float = 0.0
    spread_autocorr_5: float = 0.0
    variance_ratio: float = 1.0
    spread_series: list[dict[str, float]]
    zscore_series: list[dict[str, float]]


class RegimeResultModel(BaseModel):
    current_regime: str = "unknown"
    trading_allowed: bool = True
    signal_suppressed: bool = False
    structural_break: bool = False
    correlation_breakdown: bool = False
    spread_variance_expansion: bool = False
    vix_level: float = 0.0
    regime_summary: dict[str, float] = {}
    cusum_break_detected: bool = False
    cusum_break_indices: list[int] = []
    chow_break_detected: bool = False
    chow_break_dates: list[str] = []
    bai_perron_breaks: list[int] = []
    num_structural_breaks: int = 0


class WalkForwardMetrics(BaseModel):
    avg_train_p_value: float = 1.0
    avg_oos_p_value: float = 1.0
    avg_half_life: float
    cointegration_percentage: float
    avg_spread_sharpe: float
    hedge_ratio_stability: float
    avg_spread_drawdown: float = 0.0
    oos_stationarity_pct: float = 0.0
    regime_stability_pct: float = 0.0
    num_folds: int


class WFBacktestMetricsModel(BaseModel):
    total_return_pct: float = 0.0
    annualised_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    num_trades: int = 0
    avg_holding_period: float = 0.0
    turnover: float = 0.0
    num_folds: int = 0


class BacktestMetricsModel(BaseModel):
    total_return_pct: float
    annualised_return_pct: float = 0.0
    sharpe_ratio: float
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    avg_holding_period: float
    turnover: float
    final_equity: float
    profit_factor: float = 0.0
    exposure_pct: float = 0.0
    beta_to_market: float = 0.0
    equity_curve: list[dict[str, float]]
    drawdown_series: list[dict[str, float]] = []


class PairData(BaseModel):
    ticker_a: str
    ticker_b: str
    correlations: list[CorrelationMetrics]
    cointegration: CointegrationResultModel
    johansen: JohansenResultModel | None = None
    spread: SpreadResultModel
    walk_forward: WalkForwardMetrics
    backtest: BacktestMetricsModel
    wf_backtest: WFBacktestMetricsModel = WFBacktestMetricsModel()
    regime: RegimeResultModel = RegimeResultModel()
    score: float = 0.0


class PairAnalysisResponse(BaseModel):
    status: str = "ok"
    pair: PairData


class PairRankRequest(BaseModel):
    pairs: list[list[str]]
    start: str = "2015-01-01"
    end: str | None = None
    significance: float = 0.05
    top_n: int = 10


class PairRankResponse(BaseModel):
    status: str = "ok"
    ranked_pairs: list[PairData]


class HeatmapRequest(BaseModel):
    tickers: list[str]
    start: str = "2015-01-01"
    end: str | None = None
    significance: float = 0.05


class HeatmapResponse(BaseModel):
    status: str = "ok"
    tickers: list[str]
    matrix: dict[str, dict[str, float]]


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
