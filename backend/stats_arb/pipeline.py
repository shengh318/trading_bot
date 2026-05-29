"""
Phase 10 — Pipeline Orchestrator.

Runs the full statistical arbitrage pipeline for a single pair:
    Data → Correlation → Cointegration → Spread → Regime →
    Walk-Forward → Strategy → Backtest → (Optional ML)

Coordinates all phases in dependency order, ensuring no lookahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_SIGNIFICANCE,
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_STOP_LOSS,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_SLIPPAGE,
    DEFAULT_BORROW_FEE,
    DEFAULT_MAX_HOLDING_DAYS,
    DEFAULT_TRAIN_PCT,
    DEFAULT_VAL_PCT,
    DEFAULT_TEST_PCT,
    DEFAULT_WALK_FORWARD_TRAIN,
    DEFAULT_WALK_FORWARD_TEST,
    DEFAULT_EMBARGO_DAYS,
)
from .data import DataManager
from .correlation import CorrelationAnalyzer
from .cointegration import CointegrationTester, JohansenTester
from .hedge_ratio import HedgeRatioEstimator, KalmanFilterHedge
from .spread import SpreadAnalyzer
from .regime import RegimeDetector
from .walk_forward import (
    WalkForwardValidator,
    PurgedWalkForwardValidator,
    WalkForwardBacktest,
)
from .strategy import TradingStrategy
from .backtest import BacktestEngine
from .ml_models import SpreadPredictor, ModelType, TaskType

logger = logging.getLogger("stats_arb.pipeline")


@dataclass
class PairAnalysisResult:
    """Complete analysis result for a single pair.

    Attributes
    ----------
    ticker_a : str
        First ticker.
    ticker_b : str
        Second ticker.
    correlation : dict
        Rolling correlation results.
    cointegration : Any
        Engle-Granger cointegration result.
    johansen : Any, optional
        Johansen test result.
    hedge_ratio : Any
        Hedge ratio estimation result.
    spread : Any
        Spread analysis result.
    regime : Any
        Regime detection result.
    walk_forward : Any
        Walk-forward validation result.
    in_sample_backtest : Any
        Full-sample in-sample backtest (REFERENCE ONLY — contains lookahead).
    wf_backtest : Any, optional
        Walk-forward out-of-sample backtest (no lookahead).
    ml_results : dict
        ML prediction results (task → MLResult).
    score : float
        Composite ranking score.
    start_date : str
        Analysis start date.
    end_date : str
        Analysis end date.
    """

    ticker_a: str
    ticker_b: str
    correlation: dict
    cointegration: Any
    johansen: Optional[Any]
    hedge_ratio: Any
    spread: Any
    regime: Any
    walk_forward: Any
    in_sample_backtest: Any
    ml_results: dict[str, Any]
    score: float
    start_date: str
    end_date: str
    wf_backtest: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ticker_a": self.ticker_a,
            "ticker_b": self.ticker_b,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "score": round(self.score, 4),
            "cointegration": self.cointegration.to_dict(),
            "hedge_ratio": self.hedge_ratio.to_dict(),
            "spread": self.spread.to_dict(),
            "regime": self.regime.to_dict(),
        }
        if self.johansen:
            d["johansen"] = self.johansen.to_dict()
        if self.walk_forward:
            d["walk_forward"] = self.walk_forward.to_dict()
        if self.in_sample_backtest:
            d["in_sample_backtest"] = self.in_sample_backtest.to_dict()
        if self.wf_backtest:
            d["wf_backtest"] = self.wf_backtest.to_dict()
        if self.ml_results:
            d["ml"] = {
                k: v.to_dict() for k, v in self.ml_results.items()
            }
        return d


class PairAnalyzer:
    """Full pipeline orchestrator for a single ticker pair.

    Runs all phases in strict dependency order:
        1. DataManager.fetch()
        2. CorrelationAnalyzer.run()
        3. CointegrationTester.run() + JohansenTester.run()
        4. HedgeRatioEstimator.ols()
        5. SpreadAnalyzer.run()
        6. RegimeDetector.detect()
        7. WalkForwardValidator.run() or PurgedWalkForwardValidator.run()
        8. TradingStrategy.execute()
        9. BacktestEngine.run()
        10. SpreadPredictor.train() (if requested)

    Parameters
    ----------
    ticker_a : str
        First ticker symbol.
    ticker_b : str
        Second ticker symbol.
    start : str or datetime
        Start date.
    end : str or datetime, optional
        End date (defaults to today).
    significance : float
        Cointegration significance threshold.
    use_purged_walk_forward : bool
        Use purged walk-forward (True) or simple (False).
    run_johansen : bool
        Run Johansen test.
    run_kalman : bool
        Run Kalman filter hedge ratio.
    run_ml : bool
        Run ML models.
    hedge_ratio_method : str
        'ols', 'rolling', or 'kalman'.
    prices : pd.DataFrame, optional
        Pre-fetched price data (skips data fetch).
    capital : float
        Initial capital for backtest.
    """

    def __init__(
        self,
        ticker_a: str,
        ticker_b: str,
        start: str | datetime,
        end: str | datetime | None = None,
        significance: float = DEFAULT_SIGNIFICANCE,
        use_purged_walk_forward: bool = True,
        run_johansen: bool = True,
        run_kalman: bool = False,
        run_ml: bool = False,
        hedge_ratio_method: str = "ols",
        prices: Optional[pd.DataFrame] = None,
        capital: float = 100_000.0,
    ) -> None:
        self.ticker_a = ticker_a.upper()
        self.ticker_b = ticker_b.upper()
        self.start = start
        self.end = end or datetime.today().strftime("%Y-%m-%d")
        self.significance = significance
        self.use_purged_walk_forward = use_purged_walk_forward
        self.run_johansen = run_johansen
        self.run_kalman = run_kalman
        self.run_ml = run_ml
        self.hedge_ratio_method = hedge_ratio_method
        self._prices = prices
        self.capital = capital

    def analyze(self) -> PairAnalysisResult:
        prices = self._get_prices()

        if len(prices) < 504:
            logger.warning(
                f"Only {len(prices)} days of data — "
                f"walk-forward results may be unreliable"
            )

        corr = self._run_correlation(prices)
        coint_result = self._run_cointegration(prices)
        johansen_result = self._run_johansen(prices)
        hr_result = self._run_hedge_ratio(prices)
        spread_result = self._run_spread_analysis(coint_result.spread, hr_result)
        regime_result = self._run_regime_detection(spread_result.spread)
        wf_result = self._run_walk_forward(prices)
        bt_result = self._run_in_sample_backtest(prices, hr_result)
        wf_bt_result = self._run_walk_forward_backtest(prices)
        ml_results = self._run_ml(spread_result, corr, regime_result)

        start_date = str(prices.index[0].date())
        end_date = str(prices.index[-1].date())

        for w in sorted(corr.keys(), reverse=True):
            if corr[w].current > 0.99:
                logger.info(
                    f"Correlation {corr[w].current:.4f} > 0.99 — suppressing "
                    f"oversensitive structural break signals"
                )
                regime_result.structural_break = False
                regime_result.cusum_break_detected = False
                regime_result.chow_break_detected = False
                regime_result.num_structural_breaks = 0
                break

        return PairAnalysisResult(
            ticker_a=self.ticker_a,
            ticker_b=self.ticker_b,
            correlation=corr,
            cointegration=coint_result,
            johansen=johansen_result,
            hedge_ratio=hr_result,
            spread=spread_result,
            regime=regime_result,
            walk_forward=wf_result,
            in_sample_backtest=bt_result,
            wf_backtest=wf_bt_result,
            ml_results=ml_results,
            score=0.0,
            start_date=start_date,
            end_date=end_date,
        )

    def _get_prices(self) -> pd.DataFrame:
        if self._prices is not None:
            return self._prices
        dm = DataManager(
            [self.ticker_a, self.ticker_b],
            self.start,
            self.end,
        )
        return dm.fetch()

    @staticmethod
    def _run_correlation(prices: pd.DataFrame) -> dict:
        ca = CorrelationAnalyzer(prices)
        return ca.run()

    def _run_cointegration(self, prices: pd.DataFrame) -> Any:
        ct = CointegrationTester(prices, self.significance)
        return ct.run()

    def _run_johansen(self, prices: pd.DataFrame) -> Optional[Any]:
        if not self.run_johansen:
            return None
        try:
            jt = JohansenTester(prices, self.significance)
            return jt.run()
        except Exception as e:
            logger.warning(f"Johansen test failed: {e}")
            return None

    def _run_hedge_ratio(self, prices: pd.DataFrame) -> Any:
        he = HedgeRatioEstimator(prices)
        if self.hedge_ratio_method == "kalman" or self.run_kalman:
            kf = KalmanFilterHedge()
            return kf.fit_dataframe(prices)
        if self.hedge_ratio_method == "rolling":
            return he.rolling_ols(window=63)
        return he.ols()

    @staticmethod
    def _run_spread_analysis(spread: pd.Series, hr_result: Any) -> Any:
        sa = SpreadAnalyzer(spread)
        return sa.run()

    @staticmethod
    def _run_regime_detection(spread: pd.Series) -> Any:
        rd = RegimeDetector(spread)
        return rd.detect()

    def _run_walk_forward(self, prices: pd.DataFrame) -> Any:
        if self.use_purged_walk_forward:
            wf = PurgedWalkForwardValidator(
                prices,
                significance=self.significance,
            )
        else:
            wf = WalkForwardValidator(
                prices,
                significance=self.significance,
            )
        return wf.run()

    def _run_in_sample_backtest(self, prices: pd.DataFrame, hr_result: Any) -> Any:
        logger.info(
            "Running full-sample in-sample backtest (REFERENCE ONLY — "
            "contains lookahead bias. For OOS results, use walk-forward backtest.)"
        )
        beta = (
            float(hr_result.beta)
            if isinstance(hr_result.beta, (int, float))
            else float(hr_result.beta[-1])
        )
        strategy = TradingStrategy(
            prices,
            hedge_ratio=beta,
            initial_capital=self.capital,
        )
        equity, trades = strategy.execute()
        engine = BacktestEngine(equity, trades, self.capital)
        return engine.run()

    def _run_walk_forward_backtest(self, prices: pd.DataFrame) -> Any:
        try:
            wfbt = WalkForwardBacktest(
                prices,
                initial_capital=self.capital,
            )
            return wfbt.run()
        except Exception as e:
            logger.warning(f"Walk-forward backtest failed: {e}")
            return None

    def _run_ml(
        self,
        spread_result: Any,
        corr_results: dict,
        regime_result: Any,
    ) -> dict[str, Any]:
        if not self.run_ml:
            return {}

        spread = spread_result.spread
        zscore = spread_result.zscore_series

        hurst_series = getattr(regime_result, "rolling_hurst", None)
        regime_series = getattr(regime_result, "regime_series", None)

        corr_series = None
        if corr_results and 60 in corr_results:
            corr_series = corr_results[60].series

        results: dict[str, Any] = {}
        tasks = [
            (TaskType.REVERSION_PROB, "reversion_prob"),
            (TaskType.REVERSION_MAGNITUDE, "reversion_magnitude"),
            (TaskType.TIME_TO_MEAN, "time_to_mean"),
            (TaskType.BREAKOUT_PROB, "breakout_prob"),
        ]

        for task, key in tasks:
            try:
                predictor = SpreadPredictor(
                    model_type=ModelType.RANDOM_FOREST,
                    task=task,
                )
                ml_result = predictor.train(
                    spread, zscore, hurst_series, corr_series, regime_series,
                )
                results[key] = ml_result
            except Exception as e:
                logger.warning(f"ML {task.value} failed: {e}")

        return results
