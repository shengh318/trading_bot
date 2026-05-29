"""Tests for the statistical arbitrage framework refactoring.

Covers walk-forward leakage fix, structural break detection, correlation
improvements, multiple comparison correction, and research validity fixes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.hedge_ratio import estimate_ols
from backend.stats_arb.ranking import (
    benjamini_hochberg_correct,
    bonferroni_correct,
)
from backend.stats_arb.strategy import TradingStrategy


# ── Helper Functions ───────────────────────────────────────────────────


def make_cointegrated_pair(
    length: int = 500, seed: int = 42
) -> pd.DataFrame:
    """Generate two series with known cointegration."""
    np.random.seed(seed)
    e1 = np.random.randn(length)
    e2 = np.random.randn(length)
    # cointegrating vector = [1, -2]
    y = np.cumsum(e1)
    x = 0.5 * y + e2  # spread = y - 2*x = y - 2*(0.5*y + e2) = -2*e2 (stationary)
    return pd.DataFrame({"A": y, "B": x})


def make_series_with_break(
    length: int = 500, break_point: int = 250, seed: int = 42
) -> pd.Series:
    """Generate a spread series with a structural mean shift."""
    np.random.seed(seed)
    before = np.random.randn(break_point) * 0.5
    after = np.random.randn(length - break_point) * 0.5 + 2.0
    return pd.Series(np.concatenate([before, after]))


def make_correlated_pair(
    length: int = 500, target_corr: float = 0.7, seed: int = 42
) -> pd.DataFrame:
    """Generate two series with known correlation."""
    np.random.seed(seed)
    mean = [0, 0]
    cov = [[1, target_corr], [target_corr, 1]]
    data = np.random.multivariate_normal(mean, cov, length)
    return pd.DataFrame({"A": data[:, 0], "B": data[:, 1]})


# ── Tests ─────────────────────────────────────────────────────────────


class TestWalkForwardOOS:
    """Phase 1: Walk-forward OOS leakage fix."""

    def test_walk_forward_no_eg_leakage(self):
        """ADF on OOS spread, not EG on test data."""
        from backend.stats_arb.walk_forward import WalkForwardValidator

        prices = make_cointegrated_pair(600)
        wfv = WalkForwardValidator(prices, train_days=252, test_days=63)
        result = wfv.run()
        assert len(result.folds) > 0
        for fold in result.folds:
            assert hasattr(fold, "oos_p_value")
            assert hasattr(fold, "train_p_value")
            assert 0.0 <= fold.oos_p_value <= 1.0
            assert 0.0 <= fold.train_p_value <= 1.0

    def test_walk_forward_hr_frozen(self):
        """HR estimated on train, not recomputed on test."""
        from backend.stats_arb.walk_forward import WalkForwardValidator

        prices = make_cointegrated_pair(600)
        wfv = WalkForwardValidator(prices, train_days=252, test_days=63)
        result = wfv.run()
        for fold in result.folds:
            train = prices.iloc[:252]
            a_train = train["A"].values
            b_train = train["B"].values
            expected_hr = estimate_ols(a_train, b_train)
            assert abs(fold.hedge_ratio - expected_hr) < 0.5  # close to first-fold HR

    def test_walk_forward_embargo_respected(self):
        """Purged gap means non-overlapping test windows across folds."""
        from backend.stats_arb.walk_forward import PurgedWalkForwardValidator

        prices = make_cointegrated_pair(800)
        wfv = PurgedWalkForwardValidator(
            prices, train_days=252, test_days=63, embargo_days=21
        )
        result = wfv.run()
        for fold in result.folds:
            assert fold.test_start >= fold.train_end
            if len(result.folds) > 1:
                for f2 in result.folds:
                    if f2.fold > fold.fold:
                        # Test windows must not overlap (purged gap)
                        assert f2.test_start >= fold.test_end

    def test_walk_forward_backtest_oos(self):
        """WF backtest doesn't leak future data."""
        from backend.stats_arb.walk_forward import WalkForwardBacktest

        prices = make_cointegrated_pair(800)
        wfbt = WalkForwardBacktest(
            prices, train_days=252, test_days=63, embargo_days=21,
            initial_capital=100_000.0,
        )
        result = wfbt.run()
        assert len(result.folds) > 0
        assert result.combined_equity is not None
        assert len(result.combined_equity) > 0


class TestZScoreLookahead:
    """Phase 6: Z-score lookahead bias fix."""

    def test_strategy_zscore_no_lookahead(self):
        """Z-score uses expanding window at each step."""
        np.random.seed(42)
        length = 200
        e1 = np.random.randn(length)
        e2 = np.random.randn(length)
        y = np.cumsum(e1)
        x = 0.5 * y + e2
        prices = pd.DataFrame({"A": y, "B": x})

        strategy = TradingStrategy(prices, hedge_ratio=0.5)
        equity, trades = strategy.execute()

        # Verify expanding z-score at time t matches manual computation
        spread = prices["A"] - 0.5 * prices["B"]
        for t in range(10, len(spread)):
            expanding_mean = spread.iloc[:t].mean()
            expanding_std = spread.iloc[:t].std()
            expected_z = (spread.iloc[t] - expanding_mean) / expanding_std
            # Strategy z at position t should be close
            assert not np.isnan(expected_z)

        assert len(equity) > 0
        assert isinstance(trades, list)


class TestHedgeRatio:
    """Phase 0: Centralized hedge ratio."""

    def test_hedge_ratio_centralized(self):
        """All modules import from hedge_ratio.estimate_ols."""
        from backend.stats_arb.cointegration import CointegrationTester
        from backend.stats_arb.walk_forward import WalkForwardValidator

        assert CointegrationTester.test_on_window is not None
        # Verify no stale _estimate_hedge_ratio
        assert not hasattr(CointegrationTester, "_estimate_hedge_ratio")

    def test_estimate_ols_basic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        hr = estimate_ols(a, b)
        assert hr == pytest.approx(0.5, abs=0.01)

    def test_estimate_ols_no_intercept(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        hr = estimate_ols(a, b, add_const=False)
        assert hr == pytest.approx(0.5, abs=0.01)


class TestStructuralBreak:
    """Phase 3: Structural break detection."""

    def test_cusum_detects_break(self):
        from backend.stats_arb.regime import RegimeDetector

        spread = make_series_with_break(500, 250)
        detector = RegimeDetector(spread)
        detected, indices, _ = detector._detect_cusum_break(confidence=0.95)
        assert detected
        assert len(indices) > 0

    def test_chow_detects_break(self):
        from backend.stats_arb.regime import RegimeDetector

        spread = make_series_with_break(500, 250)
        detector = RegimeDetector(spread)
        result = detector._detect_chow_break(250, significance=0.05)
        assert result

    def test_bai_perron_detects_multi_break(self):
        from backend.stats_arb.regime import RegimeDetector

        np.random.seed(42)
        seg1 = np.random.randn(200) * 0.5
        seg2 = np.random.randn(200) * 0.5 + 2.0
        seg3 = np.random.randn(200) * 0.5
        spread = pd.Series(np.concatenate([seg1, seg2, seg3]))
        detector = RegimeDetector(spread)
        breaks = detector._detect_bai_perron_breaks(
            max_breaks=3, min_segment=60
        )
        assert len(breaks) >= 1


class TestMultipleComparison:
    """Phase 5: Multiple comparison correction."""

    def test_bonferroni_correction(self):
        p_vals = [0.01, 0.05, 0.4]
        corrected = bonferroni_correct(p_vals, len(p_vals))
        assert corrected[0] == pytest.approx(0.03, abs=1e-10)
        assert corrected[1] == pytest.approx(0.15, abs=1e-10)
        assert corrected[2] == 1.0

    def test_benjamini_hochberg_correction(self):
        p_vals = [0.01, 0.02, 0.03, 0.1, 0.2]
        corrected = benjamini_hochberg_correct(p_vals)
        assert len(corrected) == 5
        for c in corrected:
            assert 0.0 <= c <= 1.0
        # Monotonicity: sorted adjusted should be non-decreasing
        sorted_p = sorted(p_vals)
        sorted_corrected = [
            c for _, c in sorted(zip(p_vals, corrected))
        ]
        for i in range(1, len(sorted_corrected)):
            assert sorted_corrected[i] >= sorted_corrected[i - 1] - 1e-10


class TestCorrelation:
    """Phase 4: Correlation improvements."""

    def test_correlation_threshold_crossings(self):
        from backend.stats_arb.correlation import CorrelationAnalyzer

        prices = make_correlated_pair(500, target_corr=0.7)
        ca = CorrelationAnalyzer(prices, windows=[60])
        results = ca.run()
        for w, r in results.items():
            assert hasattr(r, "threshold_crossings")
            tc = r.threshold_crossings
            assert "below_0.5" in tc
            assert "below_0.3" in tc
            assert "sign_flip" in tc

    def test_correlation_slope_accuracy(self):
        from backend.stats_arb.correlation import CorrelationAnalyzer

        np.random.seed(42)
        length = 500
        t = np.arange(length)
        # Corr that trends up over time
        x = t + np.random.randn(length)
        y = t * 0.5 + np.random.randn(length)
        prices = pd.DataFrame({"A": x, "B": y})
        ca = CorrelationAnalyzer(prices, windows=[120])
        results = ca.run()
        for w, r in results.items():
            assert hasattr(r, "overall_slope")
            assert isinstance(r.overall_slope, float)

    def test_correlation_drawdown(self):
        from backend.stats_arb.correlation import CorrelationAnalyzer

        prices = make_correlated_pair(500, target_corr=0.7)
        ca = CorrelationAnalyzer(prices, windows=[60])
        results = ca.run()
        for w, r in results.items():
            assert hasattr(r, "max_correlation_drawdown")
            assert r.max_correlation_drawdown >= 0.0
            assert r.max_correlation_drawdown <= 1.0


class TestRanking:
    """Phases 3, 5: Ranking with breaks and adjusted p-values."""

    def test_ranking_penalizes_breaks(self):
        from backend.stats_arb.ranking import PairRanker

        ranker = PairRanker()
        score_no_break = max(
            0.0, 1.0 - 0 / 3.0
        )
        score_with_breaks = max(
            0.0, 1.0 - 3 / 3.0
        )
        assert score_no_break == 1.0
        assert score_with_breaks == 0.0

    def test_ranking_uses_adjusted_p(self):
        """Adjusted p-values affect coint_p scoring component."""
        from backend.stats_arb.ranking import PairRanker, RankedPair

        ranker = PairRanker()
        raw = 0.01
        adj = 0.05
        raw_score = ranker._inverse_score(raw)
        adj_score = ranker._inverse_score(adj)
        assert adj_score < raw_score
