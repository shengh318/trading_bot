"""
Statistical Arbitrage Research Framework.

A production-grade modular framework for pairs trading and spread
mean-reversion analysis. Covers the full pipeline from data acquisition
through relationship discovery, spread modeling, regime detection,
walk-forward validation, backtesting, pair ranking, and ML extensions.
"""

from __future__ import annotations

from .config import (
    CORRELATION_WINDOWS,
    DEFAULT_SIGNIFICANCE,
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_STOP_LOSS,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_BORROW_FEE,
    DEFAULT_SLIPPAGE,
    DEFAULT_TRAIN_PCT,
    DEFAULT_VAL_PCT,
    DEFAULT_TEST_PCT,
    DEFAULT_WALK_FORWARD_TRAIN,
    DEFAULT_WALK_FORWARD_TEST,
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_MAX_HOLDING_DAYS,
)
from .data import DataManager
from .correlation import CorrelationAnalyzer, RollingCorrelationResult
from .cointegration import CointegrationResult, CointegrationTester, JohansenResult, JohansenTester
from .hedge_ratio import estimate_ols, HedgeRatioEstimator, KalmanFilterHedge
from .spread import SpreadAnalyzer, SpreadResult
from .regime import RegimeDetector, RegimeResult
from .walk_forward import (
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardValidator,
    PurgedWalkForwardValidator,
    WFBacktestFold,
    WFBacktestResult,
    WalkForwardBacktest,
)
from .strategy import TradingStrategy, TradeRecord
from .backtest import BacktestEngine, BacktestResult
from .ranking import PairRanker, RankedPair
from .ml_models import SpreadPredictor
from .pipeline import PairAnalysisResult, PairAnalyzer
from .visualization import Visualizer

__all__ = [
    "CORRELATION_WINDOWS",
    "DEFAULT_SIGNIFICANCE",
    "DEFAULT_Z_ENTRY",
    "DEFAULT_Z_EXIT",
    "DEFAULT_STOP_LOSS",
    "DEFAULT_TRANSACTION_COST",
    "DEFAULT_BORROW_FEE",
    "DEFAULT_SLIPPAGE",
    "DEFAULT_TRAIN_PCT",
    "DEFAULT_VAL_PCT",
    "DEFAULT_TEST_PCT",
    "DEFAULT_WALK_FORWARD_TRAIN",
    "DEFAULT_WALK_FORWARD_TEST",
    "DEFAULT_EMBARGO_DAYS",
    "DEFAULT_MAX_HOLDING_DAYS",
    "DataManager",
    "CorrelationAnalyzer",
    "RollingCorrelationResult",
    "CointegrationResult",
    "CointegrationTester",
    "JohansenResult",
    "JohansenTester",
    "estimate_ols",
    "HedgeRatioEstimator",
    "KalmanFilterHedge",
    "SpreadAnalyzer",
    "SpreadResult",
    "RegimeDetector",
    "RegimeResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "WalkForwardValidator",
    "PurgedWalkForwardValidator",
    "WFBacktestFold",
    "WFBacktestResult",
    "WalkForwardBacktest",
    "TradingStrategy",
    "TradeRecord",
    "BacktestEngine",
    "BacktestResult",
    "PairRanker",
    "RankedPair",
    "SpreadPredictor",
    "PairAnalysisResult",
    "PairAnalyzer",
    "Visualizer",
]
