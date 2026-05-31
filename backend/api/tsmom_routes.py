"""TSMOM API — Time-Series Momentum analysis, backtesting, and research."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# TSMOM classes are lazy-imported inside route handlers for fast startup

logger = logging.getLogger("tsmom_api")

router = APIRouter()


class TSMOMAnalysisRequest(BaseModel):
    symbols: str = "SPY,QQQ,VOO,NVDA,AMD,META"
    start: str = "2015-01-01"
    end: str = ""
    initial_capital: float = 100000.0
    long_only: bool = True
    top_n: int = 3
    rebalance_freq: str = "ME"
    momentum_lookbacks: list[int] | None = None
    trend_filters: list[str] | None = None
    transaction_cost_pct: float = 0.001
    target_volatility: float = 0.15


class TSMOMResearchResponse(BaseModel):
    status: str
    metrics: dict
    benchmark: dict | None = None
    regime_analysis: dict | None = None
    walk_forward: dict | None = None
    ml_performance: dict | None = None
    message: str = ""


class TSMOMParamSearchRequest(BaseModel):
    symbols: str = "SPY,QQQ,VOO,NVDA,AMD,META"
    start: str = "2015-01-01"
    end: str = ""
    initial_capital: float = 100000.0


class TSMOMParamSearchResponse(BaseModel):
    status: str
    results: list[dict]
    best_config: dict | None = None
    message: str = ""


@router.post("/api/tsmom/analyze", response_model=TSMOMResearchResponse)
def analyze_tsmom(req: TSMOMAnalysisRequest):
    from backend.strategies.tsmom_strategy import TSMOMConfig, TSMOMResearchFramework
    symbols = [s.strip().upper() for s in req.symbols.split(",") if s.strip()]
    end_date = req.end or datetime.now().strftime("%Y-%m-%d")

    config = TSMOMConfig(
        symbols=symbols,
        start=req.start,
        end=end_date,
        initial_capital=req.initial_capital,
        long_only=req.long_only,
        top_n=req.top_n,
        rebalance_freq=req.rebalance_freq,
        momentum_lookbacks=req.momentum_lookbacks or [21, 63, 126, 252],
        trend_filters=req.trend_filters or ["price_above_ma200"],
        transaction_cost_pct=req.transaction_cost_pct,
        target_volatility=req.target_volatility,
    )

    framework = TSMOMResearchFramework(config)
    try:
        result = framework.run_full_analysis(symbols, req.start, end_date)
    except Exception as e:
        logger.error(f"TSMOM analysis failed: {e}")
        return TSMOMResearchResponse(
            status="error",
            metrics={},
            message=str(e),
        )

    return TSMOMResearchResponse(
        status="ok",
        metrics=result.get("metrics", {}),
        benchmark=result.get("benchmark"),
        regime_analysis=result.get("regime_analysis"),
        walk_forward=result.get("walk_forward"),
        ml_performance=result.get("ml_performance"),
    )


@router.post("/api/tsmom/param-search", response_model=TSMOMParamSearchResponse)
def param_search_tsmom(req: TSMOMParamSearchRequest):
    from backend.strategies.tsmom_strategy import TSMOMConfig, TSMOMDataLoader, TSMOMParameterResearch
    symbols = [s.strip().upper() for s in req.symbols.split(",") if s.strip()]
    end_date = req.end or datetime.now().strftime("%Y-%m-%d")

    config = TSMOMConfig(
        symbols=symbols,
        start=req.start,
        end=end_date,
        initial_capital=req.initial_capital,
    )

    data_loader = TSMOMDataLoader(config)
    prices = data_loader.load_prices(symbols, req.start, end_date)

    if not prices:
        return TSMOMParamSearchResponse(
            status="error",
            results=[],
            message="No data loaded for the given symbols",
        )

    researcher = TSMOMParameterResearch(config)
    try:
        results = researcher.run_grid(prices)
        df = researcher.to_dataframe(results)
        best_idx = df["sharpe_ratio"].idxmax() if not df.empty and "sharpe_ratio" in df.columns else None
        best = df.loc[best_idx].to_dict() if best_idx is not None else None
    except Exception as e:
        logger.error(f"TSMOM param search failed: {e}")
        return TSMOMParamSearchResponse(
            status="error",
            results=[],
            message=str(e),
        )

    return TSMOMParamSearchResponse(
        status="ok",
        results=df.to_dict("records") if not df.empty else [],
        best_config=best,
    )


@router.get("/api/tsmom/health")
def tsmom_health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "description": "Time-Series Momentum Research Framework",
    }
