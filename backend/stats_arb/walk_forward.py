"""
Phase 5 — Walk-Forward Validation.

Purged walk-forward validation with embargo windows. This is the most
important phase for avoiding lookahead bias and data leakage.

Key principles:
    1. Hedge ratio estimated ONLY on training data — never recomputed
       on validation or test.
    2. Cointegration tested ONLY on training data.
    3. Validation/test remain completely unseen until evaluation.
    4. Embargo windows prevent leakage between adjacent folds.
    5. Purged CV drops gap periods between train/test splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .config import (
    DEFAULT_SIGNIFICANCE,
    DEFAULT_WALK_FORWARD_TRAIN,
    DEFAULT_WALK_FORWARD_TEST,
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_MIN_TRAIN_SAMPLES,
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_STOP_LOSS,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_SLIPPAGE,
    DEFAULT_BORROW_FEE,
    DEFAULT_MAX_HOLDING_DAYS,
    WALK_FORWARD_ADF_MAXLAG,
)
from .hedge_ratio import estimate_ols
from .strategy import TradingStrategy, TradeRecord

logger = logging.getLogger("stats_arb.walk_forward")


@dataclass
class WalkForwardFold:
    """Result from a single walk-forward fold.

    Attributes
    ----------
    fold : int
        Fold index.
    train_start : pd.Timestamp
        Training window start.
    train_end : pd.Timestamp
        Training window end.
    test_start : pd.Timestamp
        Test window start.
    test_end : pd.Timestamp
        Test window end.
    hedge_ratio : float
        Hedge ratio estimated on training data ONLY.
    hedge_ratio_drift : float
        Relative change in hedge ratio vs previous fold.
    train_p_value : float
        Cointegration p-value on training data (Engle-Granger).
    oos_p_value : float
        ADF p-value on out-of-sample test spread (frozen HR).
    half_life : float
        Half-life estimated on test spread.
    spread_sharpe : float
        Annualised Sharpe ratio of test spread.
    spread_drawdown : float
        Maximum drawdown of test spread.
    is_cointegrated : bool
        Whether training p-value < significance.
    test_stationary : bool
        Whether ADF test on test spread is stationary.
    regime_stable : bool
        Whether the spread regime remained stable during test.
    """

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    hedge_ratio: float
    hedge_ratio_drift: float
    train_p_value: float
    oos_p_value: float
    half_life: float
    spread_sharpe: float
    spread_drawdown: float
    is_cointegrated: bool
    test_stationary: bool
    regime_stable: bool


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Attributes
    ----------
    folds : list[WalkForwardFold]
        Individual fold results.
    avg_train_p_value : float
        Mean training EG cointegration p-value across folds.
    avg_oos_p_value : float
        Mean out-of-sample ADF p-value across folds.
    avg_half_life : float
        Mean half-life across folds.
    cointegration_percentage : float
        Percentage of folds where cointegration held.
    avg_spread_sharpe : float
        Mean out-of-sample spread Sharpe.
    hedge_ratio_stability : float
        Standard deviation of hedge ratios (lower = more stable).
    avg_spread_drawdown : float
        Mean maximum drawdown of test spread.
    oos_stationarity_pct : float
        Percentage of test windows where spread was stationary.
    regime_stability_pct : float
        Percentage of test windows where regime was stable.
    """

    folds: list[WalkForwardFold]
    avg_train_p_value: float
    avg_oos_p_value: float
    avg_half_life: float
    cointegration_percentage: float
    avg_spread_sharpe: float
    hedge_ratio_stability: float
    avg_spread_drawdown: float
    oos_stationarity_pct: float
    regime_stability_pct: float

    def to_dict(self) -> dict[str, Any]:
        if not self.folds:
            return {"num_folds": 0}
        return {
            "num_folds": len(self.folds),
            "avg_train_p_value": round(self.avg_train_p_value, 6),
            "avg_oos_p_value": round(self.avg_oos_p_value, 6),
            "avg_half_life": round(self.avg_half_life, 2),
            "cointegration_percentage": round(self.cointegration_percentage, 2),
            "avg_spread_sharpe": round(self.avg_spread_sharpe, 4),
            "hedge_ratio_stability": round(self.hedge_ratio_stability, 6),
            "avg_spread_drawdown": round(self.avg_spread_drawdown, 2),
            "oos_stationarity_pct": round(self.oos_stationarity_pct, 2),
            "regime_stability_pct": round(self.regime_stability_pct, 2),
        }


class WalkForwardValidator:
    """Simple rolling walk-forward analysis without purging or embargo.

    Useful as a baseline comparison against the purged version.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        train_days: int = DEFAULT_WALK_FORWARD_TRAIN,
        test_days: int = DEFAULT_WALK_FORWARD_TEST,
        significance: float = DEFAULT_SIGNIFICANCE,
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.train_days = train_days
        self.test_days = test_days
        self.significance = significance

    def run(self) -> WalkForwardResult:
        data = self.prices
        total = len(data)
        step = self.test_days
        folds: list[WalkForwardFold] = []
        hedge_ratios: list[float] = []

        if total < self.train_days + self.test_days:
            logger.warning(
                f"Walk-forward needs ≥{self.train_days + self.test_days} days, got {total}"
            )
            return self._empty_result()

        fold_idx = 0
        for start in range(0, total - self.train_days - self.test_days + 1, step):
            train = data.iloc[start : start + self.train_days]
            test = data.iloc[
                start + self.train_days : start + self.train_days + self.test_days
            ]

            if len(train) < self.train_days or len(test) < self.test_days:
                continue

            result = self._evaluate_fold(train, test, hedge_ratios, fold_idx)
            folds.append(result)
            fold_idx += 1

        if not folds:
            return self._empty_result()

        return self._aggregate(folds, hedge_ratios)

    def _evaluate_fold(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        hedge_ratios: list[float],
        fold_idx: int,
    ) -> WalkForwardFold:
        a_train = train[self.cols[0]].values
        b_train = train[self.cols[1]].values

        hr = estimate_ols(a_train, b_train)
        prev_hr = hedge_ratios[-1] if hedge_ratios else hr
        hr_drift = abs(hr - prev_hr) / max(abs(prev_hr), 1e-10)
        hedge_ratios.append(hr)

        a_test = test[self.cols[0]].values
        b_test = test[self.cols[1]].values
        spread_test = a_test - hr * b_test

        # EG on training data
        _, train_p_val, _ = coint(a_train, b_train, maxlag=1, autolag="AIC")

        # ADF on OOS spread with frozen hedge ratio (NO re-estimation on test)
        adf_stat, oos_p_val = adfuller(
            spread_test, maxlag=WALK_FORWARD_ADF_MAXLAG, autolag="AIC"
        )[:2]

        hl = self._estimate_half_life(spread_test)
        spread_sharpe = self._compute_spread_sharpe(spread_test)
        spread_dd = self._compute_max_drawdown(spread_test)

        test_stationary = self._is_stationary(spread_test)
        regime_stable = abs(hl) < 120 if np.isfinite(hl) else False

        return WalkForwardFold(
            fold=fold_idx,
            train_start=train.index[0],
            train_end=train.index[-1],
            test_start=test.index[0],
            test_end=test.index[-1],
            hedge_ratio=hr,
            hedge_ratio_drift=hr_drift,
            train_p_value=float(train_p_val),
            oos_p_value=float(oos_p_val),
            half_life=hl,
            spread_sharpe=spread_sharpe,
            spread_drawdown=spread_dd,
            is_cointegrated=bool(train_p_val < self.significance),
            test_stationary=test_stationary,
            regime_stable=regime_stable,
        )

    @staticmethod
    def _estimate_half_life(spread: np.ndarray) -> float:
        import math
        if len(spread) < 10:
            return float("inf")
        s = pd.Series(spread)
        lagged = s.shift(1).dropna().values
        delta = s.diff().dropna().values
        if len(lagged) < 5:
            return float("inf")
        lagged_const = np.column_stack([np.ones_like(lagged), lagged])
        try:
            theta, _, _, _ = np.linalg.lstsq(lagged_const, delta, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")
        theta_val = theta[1]
        if theta_val >= 0:
            return float("inf")
        hl = -math.log(2) / theta_val
        if not np.isfinite(hl) or hl > 1e6:
            return float("inf")
        return float(hl)

    @staticmethod
    def _compute_spread_sharpe(spread: np.ndarray) -> float:
        import math
        if spread.std() == 0:
            return 0.0
        return float(spread.mean() / spread.std() * math.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(spread: np.ndarray) -> float:
        s = pd.Series(spread)
        cummax = s.cummax()
        dd = ((s - cummax) / cummax * 100)
        return float(dd.min()) if not dd.empty else 0.0

    @staticmethod
    def _is_stationary(series: np.ndarray) -> bool:
        from statsmodels.tsa.stattools import adfuller
        try:
            p = float(adfuller(series, maxlag=1, autolag="AIC")[1])
            return bool(p < 0.05)
        except Exception:
            return False

    @staticmethod
    def _empty_result() -> WalkForwardResult:
        return WalkForwardResult(
            folds=[],
            avg_train_p_value=1.0,
            avg_oos_p_value=1.0,
            avg_half_life=float("inf"),
            cointegration_percentage=0.0,
            avg_spread_sharpe=0.0,
            hedge_ratio_stability=0.0,
            avg_spread_drawdown=0.0,
            oos_stationarity_pct=0.0,
            regime_stability_pct=0.0,
        )

    def _aggregate(
        self, folds: list[WalkForwardFold], hedge_ratios: list[float]
    ) -> WalkForwardResult:
        train_p_vals = [f.train_p_value for f in folds]
        oos_p_vals = [f.oos_p_value for f in folds]
        half_lives = [f.half_life for f in folds if np.isfinite(f.half_life)]
        sharps = [f.spread_sharpe for f in folds]
        dds = [f.spread_drawdown for f in folds]
        stationarity = [f.test_stationary for f in folds]
        stability = [f.regime_stable for f in folds]

        return WalkForwardResult(
            folds=folds,
            avg_train_p_value=float(np.mean(train_p_vals)) if train_p_vals else 1.0,
            avg_oos_p_value=float(np.mean(oos_p_vals)) if oos_p_vals else 1.0,
            avg_half_life=float(np.mean(half_lives)) if half_lives else float("inf"),
            cointegration_percentage=float(np.mean([f.is_cointegrated for f in folds])) * 100,
            avg_spread_sharpe=float(np.mean(sharps)) if sharps else 0.0,
            hedge_ratio_stability=float(np.std(hedge_ratios)) if hedge_ratios else 0.0,
            avg_spread_drawdown=float(np.mean(dds)) if dds else 0.0,
            oos_stationarity_pct=float(np.mean(stationarity)) * 100 if stationarity else 0.0,
            regime_stability_pct=float(np.mean(stability)) * 100 if stability else 0.0,
        )


class PurgedWalkForwardValidator:
    """Purged walk-forward validation with embargo windows.

    This is the GOLD STANDARD for time series cross-validation in
    quantitative finance.

    Key differences from naive walk-forward:
        1. PURGING: A gap is left between train and test to prevent
           leakage from overlapping observations.
        2. EMBARGO: After each test period, the next N days are excluded
           from the next training set to prevent information leakage.
        3. Strict separation: hedge ratio and cointegration are NEVER
           tested on data that overlaps with training.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with columns [ticker_a, ticker_b].
    train_days : int
        Number of days in each training window.
    test_days : int
        Number of days in each test window.
    embargo_days : int
        Number of days to embargo after each test period.
    min_train_samples : int
        Minimum training samples required.
    significance : float
        Cointegration significance threshold.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        train_days: int = DEFAULT_WALK_FORWARD_TRAIN,
        test_days: int = DEFAULT_WALK_FORWARD_TEST,
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
        min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES,
        significance: float = DEFAULT_SIGNIFICANCE,
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.min_train_samples = min_train_samples
        self.significance = significance

    def run(self) -> WalkForwardResult:
        data = self.prices
        total = len(data)
        folds: list[WalkForwardFold] = []
        hedge_ratios: list[float] = []

        if total < self.train_days + self.test_days + self.embargo_days:
            logger.warning(
                f"Purged WFV needs ≥{self.train_days + self.test_days + self.embargo_days} "
                f"days, got {total}"
            )
            return WalkForwardValidator._empty_result()

        test_start_pos = self.train_days + self.embargo_days
        step = self.test_days + self.embargo_days
        fold_idx = 0

        for start in range(0, total - test_start_pos - self.test_days + 1, step):
            train_end = start + self.train_days
            test_start = train_end + self.embargo_days
            test_end = test_start + self.test_days

            if test_end > total:
                continue

            train = data.iloc[start:train_end]
            test = data.iloc[test_start:test_end]

            if len(train) < self.min_train_samples or len(test) < self.test_days // 2:
                continue

            wfv = WalkForwardValidator.__new__(WalkForwardValidator)
            wfv.prices = self.prices
            wfv.cols = self.cols
            wfv.significance = self.significance

            result = wfv._evaluate_fold(train, test, hedge_ratios, fold_idx)
            folds.append(result)
            fold_idx += 1

        if not folds:
            return WalkForwardValidator._empty_result()

        wfv = WalkForwardValidator.__new__(WalkForwardValidator)
        return wfv._aggregate(folds, hedge_ratios)


# ── Walk-Forward Backtest ──────────────────────────────────────────────


@dataclass
class WFBacktestFold:
    """Single fold result from walk-forward backtest."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    hedge_ratio: float
    equity_curve: pd.Series
    trades: list[TradeRecord]
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int


@dataclass
class WFBacktestResult:
    """Aggregated walk-forward backtest results."""

    folds: list[WFBacktestFold]
    combined_equity: pd.Series
    combined_trades: list[TradeRecord]
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    avg_holding_period: float
    turnover: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_folds": len(self.folds),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualised_return_pct": round(self.annualised_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "num_trades": self.num_trades,
            "avg_holding_period": round(self.avg_holding_period, 2),
            "turnover": round(self.turnover, 4),
        }


class WalkForwardBacktest:
    """Runs actual spread trading strategy on each walk-forward fold.

    For each fold:
        1. Estimate hedge ratio on training data ONLY
        2. Run TradingStrategy on test data using fold's hedge ratio
        3. Record fold-level equity curve and trades

    No lookahead — each fold knows only its own training data.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        train_days: int = DEFAULT_WALK_FORWARD_TRAIN,
        test_days: int = DEFAULT_WALK_FORWARD_TEST,
        embargo_days: int = DEFAULT_EMBARGO_DAYS,
        z_entry: float = DEFAULT_Z_ENTRY,
        z_exit: float = DEFAULT_Z_EXIT,
        stop_loss: float = DEFAULT_STOP_LOSS,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        slippage: float = DEFAULT_SLIPPAGE,
        borrow_fee: float = DEFAULT_BORROW_FEE,
        max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss = stop_loss
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.borrow_fee = borrow_fee
        self.max_holding_days = max_holding_days
        self.initial_capital = initial_capital

    def run(self) -> WFBacktestResult:
        data = self.prices
        total = len(data)
        folds: list[WFBacktestFold] = []
        combined_trades: list[TradeRecord] = []

        if total < self.train_days + self.embargo_days + self.test_days:
            logger.warning(f"WF backtest needs ≥{self.train_days + self.embargo_days + self.test_days} days, got {total}")
            empty_eq = pd.Series([self.initial_capital], index=data.index[:1])
            return WFBacktestResult(
                folds=[], combined_equity=empty_eq, combined_trades=[],
                total_return_pct=0.0, annualised_return_pct=0.0, sharpe_ratio=0.0,
                max_drawdown_pct=0.0, win_rate_pct=0.0, num_trades=0,
                avg_holding_period=0.0, turnover=0.0,
            )

        test_start_pos = self.train_days + self.embargo_days
        step = self.test_days + self.embargo_days
        fold_idx = 0

        for start in range(0, total - test_start_pos - self.test_days + 1, step):
            train_end = start + self.train_days
            test_start = train_end + self.embargo_days
            test_end = test_start + self.test_days

            if test_end > total:
                continue

            train = data.iloc[start:train_end]
            test = data.iloc[test_start:test_end]

            if len(train) < 126 or len(test) < self.test_days // 2:
                continue

            fold_result = self._run_fold(train, test, fold_idx)
            folds.append(fold_result)
            combined_trades.extend(fold_result.trades)
            fold_idx += 1

        if not folds:
            empty_eq = pd.Series([self.initial_capital], index=data.index[:1])
            return WFBacktestResult(
                folds=[], combined_equity=empty_eq, combined_trades=[],
                total_return_pct=0.0, annualised_return_pct=0.0, sharpe_ratio=0.0,
                max_drawdown_pct=0.0, win_rate_pct=0.0, num_trades=0,
                avg_holding_period=0.0, turnover=0.0,
            )

        combined_eq = self._build_combined_equity(folds)
        return self._aggregate(folds, combined_eq, combined_trades)

    def _run_fold(
        self, train: pd.DataFrame, test: pd.DataFrame, fold_idx: int,
    ) -> WFBacktestFold:
        a_train = train[self.cols[0]].values
        b_train = train[self.cols[1]].values
        hr = estimate_ols(a_train, b_train)

        strategy = TradingStrategy(
            test,
            hedge_ratio=hr,
            z_entry=self.z_entry,
            z_exit=self.z_exit,
            stop_loss=self.stop_loss,
            transaction_cost=self.transaction_cost,
            slippage=self.slippage,
            borrow_fee=self.borrow_fee,
            max_holding_days=self.max_holding_days,
            initial_capital=self.initial_capital,
        )
        equity, trades = strategy.execute()

        total_ret = (float(equity.iloc[-1]) / self.initial_capital - 1.0) * 100.0
        daily_ret = equity.pct_change().dropna()
        sharpe = float(daily_ret.mean() / daily_ret.std() * 252**0.5) if daily_ret.std() > 0 else 0.0
        cummax = equity.cummax()
        dd = ((equity - cummax) / cummax * 100)
        max_dd = float(dd.min()) if not dd.empty else 0.0

        return WFBacktestFold(
            fold=fold_idx,
            train_start=train.index[0],
            train_end=train.index[-1],
            test_start=test.index[0],
            test_end=test.index[-1],
            hedge_ratio=hr,
            equity_curve=equity,
            trades=trades,
            total_return_pct=total_ret,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            num_trades=len(trades),
        )

    @staticmethod
    def _build_combined_equity(folds: list[WFBacktestFold]) -> pd.Series:
        segments: list[pd.Series] = []
        for f in folds:
            eq = f.equity_curve
            if segments:
                eq = eq * (segments[-1].iloc[-1] / eq.iloc[0])
            segments.append(eq)
        return pd.concat(segments)

    def _aggregate(
        self, folds: list[WFBacktestFold], combined_eq: pd.Series,
        combined_trades: list[TradeRecord],
    ) -> WFBacktestResult:
        total_ret = (float(combined_eq.iloc[-1]) / self.initial_capital - 1.0) * 100.0
        years = len(combined_eq) / 252.0
        annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100 if years > 0 and (1 + total_ret / 100) > 0 else 0.0

        daily_ret = combined_eq.pct_change().dropna()
        sharpe = float(daily_ret.mean() / daily_ret.std() * 252**0.5) if daily_ret.std() > 0 else 0.0
        cummax = combined_eq.cummax()
        dd = ((combined_eq - cummax) / cummax * 100)
        max_dd = float(dd.min()) if not dd.empty else 0.0

        num_trades = len(combined_trades)
        wins = [t for t in combined_trades if getattr(t, "pnl", 0) > 0]
        win_rate = (len(wins) / num_trades * 100) if num_trades > 0 else 0.0

        holding_periods = [getattr(t, "holding_period", 0) for t in combined_trades]
        avg_hold = float(np.mean(holding_periods)) if holding_periods else 0.0

        total_traded = sum(
            abs(getattr(t, "shares_a", 0)) * getattr(t, "entry_price_a", 0)
            + abs(getattr(t, "shares_b", 0)) * getattr(t, "entry_price_b", 0)
            for t in combined_trades
        )
        turnover = float(total_traded / self.initial_capital / max(num_trades, 1))

        return WFBacktestResult(
            folds=folds,
            combined_equity=combined_eq,
            combined_trades=combined_trades,
            total_return_pct=total_ret,
            annualised_return_pct=annual_ret,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
            num_trades=num_trades,
            avg_holding_period=avg_hold,
            turnover=turnover,
        )
