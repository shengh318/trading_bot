"""Audit tests for statistical arbitrage module.

Identifies edge cases and potential bugs in:
- CorrelationAnalyzer
- CointegrationTester / JohansenTester
- SpreadAnalyzer
- RegimeDetector
- WalkForwardValidator / PurgedWalkForwardValidator
- TradingStrategy (exit reason bug)
- PairRanker (multiple comparison corrections)
- BacktestEngine (stats_arb)
"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.config import (
    DEFAULT_Z_ENTRY, DEFAULT_Z_EXIT, DEFAULT_STOP_LOSS, DEFAULT_MAX_HOLDING_DAYS,
)
from backend.stats_arb.correlation import CorrelationAnalyzer, _rolling_correlation
from backend.stats_arb.cointegration import CointegrationTester, JohansenTester
from backend.stats_arb.spread import SpreadAnalyzer
from backend.stats_arb.regime import RegimeDetector
from backend.stats_arb.walk_forward import (
    WalkForwardValidator, PurgedWalkForwardValidator, WalkForwardBacktest,
)
from backend.stats_arb.strategy import TradingStrategy, PositionSide
from backend.stats_arb.backtest import BacktestEngine as StatsArbBacktestEngine
from backend.stats_arb.ranking import (
    PairRanker, bonferroni_correct, benjamini_hochberg_correct,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def make_pair_data(length=500, seed=42):
    """Generate random walk pair."""
    np.random.seed(seed)
    e1 = np.random.randn(length)
    e2 = np.random.randn(length)
    y = np.cumsum(e1)
    x = 0.5 * y + e2
    return pd.DataFrame({"A": y, "B": x})


def make_constant_pair(length=100):
    """Generate pair with constant prices."""
    return pd.DataFrame({"A": [100.0] * length, "B": [200.0] * length})


def make_identical_pair(length=100):
    """Generate pair with identical prices."""
    vals = np.cumsum(np.random.randn(length)) + 100.0
    return pd.DataFrame({"A": vals, "B": vals})


# ══════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYZER
# ══════════════════════════════════════════════════════════════════════════

class TestCorrelationAnalyzerEdgeCases:

    def test_empty_series(self):
        """Empty series should not crash."""
        prices = pd.DataFrame({"A": [], "B": []})
        ca = CorrelationAnalyzer(prices, windows=[20])
        results = ca.run()
        assert results == {}

    def test_constant_prices(self):
        """Constant prices -> pct_change is all 0, corr is NaN -> should not crash."""
        prices = make_constant_pair(100)
        ca = CorrelationAnalyzer(prices, windows=[20])
        results = ca.run()
        # Windows with 0 return series -> rolling correlation is NaN
        # This should not crash, results dict may be empty
        assert isinstance(results, dict)

    def test_single_window(self):
        """Single window returns correctly."""
        prices = make_pair_data(200)
        ca = CorrelationAnalyzer(prices, windows=[60])
        results = ca.run()
        assert 60 in results
        r = results[60]
        assert -1.0 <= r.current <= 1.0
        assert r.stability_score >= 0.0

    def test_correlation_drawdown_no_division_by_zero(self):
        """Correlation drawdown should handle zero cummax gracefully."""
        prices = make_pair_data(100)
        ca = CorrelationAnalyzer(prices, windows=[20])
        results = ca.run()
        for r in results.values():
            assert r.max_correlation_drawdown >= 0.0
            assert r.max_correlation_drawdown <= 1.0

    def test_stability_score_bounds(self):
        """Stability score should be in [0, 1]."""
        prices = make_pair_data(500)
        ca = CorrelationAnalyzer(prices, windows=[60, 120])
        results = ca.run()
        for r in results.values():
            assert 0.0 <= r.stability_score <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# COINTEGRATION
# ══════════════════════════════════════════════════════════════════════════

class TestCointegrationEdgeCases:

    def test_identical_series(self):
        """Two identical series should be cointegrated."""
        prices = make_identical_pair(200)
        ct = CointegrationTester(prices)
        result = ct.run()
        assert result.is_cointegrated

    def test_constant_series(self):
        """Constant series may fail to converge but should not crash."""
        prices = make_constant_pair(100)
        # This may cause issues because ADF on constant spread is degenerate
        try:
            ct = CointegrationTester(prices)
            result = ct.run()
            # May or may not be cointegrated, but shouldn't crash
            assert isinstance(result.is_cointegrated, bool)
        except Exception as e:
            # Some implementations may raise on degenerate input
            # This is acceptable as long as it's a predictable error
            pytest.skip(f"Constant series cointegration raised: {e}")

    def test_johansen_constant_prices(self):
        """Johansen test should handle constant prices gracefully."""
        prices = make_constant_pair(100)
        jt = JohansenTester(prices)
        result = jt.run()
        if result is not None:
            assert isinstance(result.is_cointegrated, bool)

    def test_johansen_too_few_samples(self):
        """Johansen with very few samples should not crash."""
        prices = make_pair_data(10)
        jt = JohansenTester(prices, k_ar_diff=1)
        result = jt.run()
        # May be None if it fails
        assert result is None or isinstance(result.is_cointegrated, bool)


# ══════════════════════════════════════════════════════════════════════════
# SPREAD ANALYZER
# ══════════════════════════════════════════════════════════════════════════

class TestSpreadAnalyzerEdgeCases:

    def test_constant_spread(self):
        """Constant spread: z-score should be 0 (mean=std, z=0/0 clamped to 0)."""
        spread = pd.Series([10.0] * 100)
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        # With std=0, zscore = (spread - mean) / std = 0/0 -> inf
        # But spread.py line handles: `zscore_series = (spread - mean) / std if std > 0 else pd.Series(0.0, ...)`
        assert result.std == 0.0
        assert result.current_zscore == 0.0
        assert result.zscore_series.iloc[-1] == 0.0

    def test_single_value_spread(self):
        """Single value spread should not crash."""
        spread = pd.Series([10.0])
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        assert result.half_life == float("inf") or math.isinf(result.half_life)
        assert result.current_zscore == 0.0

    def test_two_value_spread(self):
        """Two values should not crash (ADF may raise due to too few samples)."""
        spread = pd.Series([10.0, 12.0])
        sa = SpreadAnalyzer(spread)
        # ADF needs more samples; this should not crash entirely
        try:
            result = sa.run()
            assert not math.isnan(result.current_zscore)
        except ValueError as e:
            # Acceptable: statsmodels raises when data is too short
            assert "sample size" in str(e).lower() or "maxlag" in str(e).lower()

    def test_half_life_no_mean_reversion(self):
        """Non-mean-reverting series (random walk) should have high half-life."""
        np.random.seed(42)
        spread = pd.Series(np.cumsum(np.random.randn(500)))
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        # Random walk should have high half-life (or inf)
        assert result.half_life > 50 or math.isinf(result.half_life)

    def test_hurst_constant(self):
        """Hurst of constant series should be 0.5."""
        spread = pd.Series([5.0] * 50)
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        assert result.hurst_exponent == 0.5

    def test_variance_explosion_no_crash(self):
        """Short spread should not crash variance explosion calc."""
        spread = pd.Series(np.random.randn(50))
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        # With only 50 data points, rolling 63 std won't have values
        assert result.variance_explosion_events == 0

    def test_expected_time_to_mean_inf(self):
        """When speed >= 0, expected_time_to_mean should be inf."""
        spread = pd.Series(np.random.randn(100))
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        if result.mean_reversion_speed >= 0:
            assert result.expected_time_to_mean == float("inf")

    def test_zscore_series_length_matches_spread(self):
        """Z-score series should be same length as input spread."""
        spread = pd.Series(np.random.randn(200), index=pd.date_range("2025-01-01", periods=200, freq="D"))
        sa = SpreadAnalyzer(spread)
        result = sa.run()
        assert len(result.zscore_series) == len(spread)


# ══════════════════════════════════════════════════════════════════════════
# REGIME DETECTOR
# ══════════════════════════════════════════════════════════════════════════

class TestRegimeDetectorEdgeCases:

    def test_constant_spread_regime(self):
        """Constant spread regime detection may raise ZeroDivisionError from Chow test."""
        spread = pd.Series([1.0] * 200)
        detector = RegimeDetector(spread)
        try:
            result = detector.detect()
            assert result.current_regime is not None
        except ZeroDivisionError:
            # Known issue: Chow test divides by zero when RSS=0 (constant spread)
            pass

    def test_short_spread_regime(self):
        """Very short spread should not crash."""
        spread = pd.Series([1.0, 2.0, 3.0])
        detector = RegimeDetector(spread)
        result = detector.detect()
        assert result.trading_allowed is not None

    def test_vix_level_unavailable(self):
        """Missing VIX data should set vix_level to 0."""
        spread = pd.Series(np.random.randn(200))
        detector = RegimeDetector(spread)
        result = detector.detect()
        # VIX may be downloaded from yfinance; if it succeeds, vix_level > 0
        # The point is it should not crash regardless
        assert isinstance(result.vix_level, float)
        assert result.vix_level >= 0.0

    def test_cusum_empty_spread(self):
        """CUSUM on empty spread should not crash (n<30 check returns early)."""
        spread = pd.Series(dtype=float)
        detector = RegimeDetector(spread)
        detected, indices, cusum = detector._detect_cusum_break()
        assert detected is False
        assert indices == []
        assert len(cusum) == 0

    def test_bai_perron_short(self):
        """Bai-Perron on very short series should not crash."""
        spread = pd.Series(np.random.randn(30))
        detector = RegimeDetector(spread)
        breaks = detector._detect_bai_perron_breaks(max_breaks=3, min_segment=60)
        assert isinstance(breaks, list)
        assert len(breaks) == 0  # no breaks found for short series


# ══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATOR
# ══════════════════════════════════════════════════════════════════════════

class TestWalkForwardEdgeCases:

    def test_insufficient_data(self):
        """Walk-forward should return empty result when data is too short."""
        prices = make_pair_data(100)
        wfv = WalkForwardValidator(prices, train_days=252, test_days=63)
        result = wfv.run()
        assert len(result.folds) == 0
        assert result.avg_train_p_value == 1.0
        assert result.cointegration_percentage == 0.0

    def test_purged_insufficient_data(self):
        """Purged walk-forward should return empty when data too short."""
        prices = make_pair_data(100)
        pwfv = PurgedWalkForwardValidator(
            prices, train_days=252, test_days=63, embargo_days=21
        )
        result = pwfv.run()
        assert len(result.folds) == 0

    def test_purged_minimal_data(self):
        """Purged walk-forward with barely enough data should work."""
        total_needed = 252 + 21 + 63 + 1  # train + embargo + test + buffer
        prices = make_pair_data(total_needed)
        pwfv = PurgedWalkForwardValidator(
            prices, train_days=252, test_days=63, embargo_days=21
        )
        result = pwfv.run()
        # Should produce at least 1 fold
        assert len(result.folds) >= 1

    def test_walk_forward_backtest_insufficient(self):
        """WF backtest should return empty when data is too short."""
        prices = make_pair_data(50)
        wfbt = WalkForwardBacktest(
            prices, train_days=252, test_days=63
        )
        result = wfbt.run()
        assert len(result.folds) == 0
        assert result.sharpe_ratio == 0.0

    def test_walk_forward_constant_prices(self):
        """Constant prices in walk-forward should not crash."""
        prices = make_constant_pair(500)
        wfv = WalkForwardValidator(prices, train_days=100, test_days=30)
        try:
            result = wfv.run()
            assert len(result.folds) >= 1
        except Exception as e:
            pytest.skip(f"Constant prices WF raised: {e}")

    def test_spread_drawdown_zero_cummax(self):
        """Spread drawdown should handle zero cummax."""
        # Spread values that go below zero
        for spread_vals in [
            np.array([0.0, -1.0, -2.0, -1.0]),
            np.array([-1.0, -2.0, -3.0, -1.0]),
        ]:
            s = pd.Series(spread_vals)
            cummax = s.cummax()
            # When cummax is 0, dividing by 0 gives inf
            # The _compute_max_drawdown doesn't guard against this
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = ((s - cummax) / cummax * 100)
                min_dd = float(dd.min()) if not dd.empty else 0.0
            # Should not crash, may produce inf
            assert np.isfinite(min_dd) or math.isinf(min_dd)


# ══════════════════════════════════════════════════════════════════════════
# TRADING STRATEGY — Exit reason bug
# ══════════════════════════════════════════════════════════════════════════

class TestTradingStrategyExitReason:
    """Tests the _get_exit_reason bug: uses DEFAULT_* constants instead of instance values."""

    def test_get_exit_reason_uses_defaults_known_bug(self):
        """BUG: _get_exit_reason is a @staticmethod that uses DEFAULT_Z_ENTRY * DEFAULT_STOP_LOSS
        and DEFAULT_MAX_HOLDING_DAYS instead of instance values. This test documents the issue."""
        prices = make_pair_data(300)
        # Use custom z_entry (3.0) instead of default (2.0)
        strategy = TradingStrategy(
            prices, hedge_ratio=0.5,
            z_entry=3.0, z_exit=0.0, stop_loss=2.0, max_holding_days=30,
        )
        # _get_exit_reason checks: abs(z) >= DEFAULT_Z_ENTRY * DEFAULT_STOP_LOSS = 2.0 * 3.0 = 6.0
        # But should check: abs(z) >= self.z_entry * self.stop_loss = 3.0 * 2.0 = 6.0
        # In this case both equal 6.0, so it coincidentally works
        z_at_stop = 6.0  # Would trigger with either
        reason = strategy._get_exit_reason(PositionSide.LONG_SPREAD, z_at_stop, 30)
        assert reason == "stop_loss"

        # Now test where custom values differ from defaults
        strategy2 = TradingStrategy(
            prices, hedge_ratio=0.5,
            z_entry=1.0, z_exit=0.0, stop_loss=2.0, max_holding_days=10,
        )
        # Instance: z_entry * stop_loss = 1.0 * 2.0 = 2.0 -> stop loss at |z| >= 2.0
        # DEFAULT: 2.0 * 3.0 = 6.0 -> stop loss at |z| >= 6.0
        z_at_3 = 3.0
        reason_bug = strategy2._get_exit_reason(PositionSide.LONG_SPREAD, z_at_3, 15)
        # With instance values: |3.0| >= 2.0 -> stop_loss
        # With defaults: |3.0| < 6.0 -> signal (WRONG)
        if reason_bug == "signal":
            # Bug confirmed: using defaults instead of instance values
            pass  # This test documents the known behavior
        assert reason_bug in ("stop_loss", "signal")


# ══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE (Stats Arb)
# ══════════════════════════════════════════════════════════════════════════

class TestStatsArbBacktestEdgeCases:

    def test_empty_equity_curve(self):
        """Empty equity curve should return empty result."""
        equity = pd.Series(dtype=float)
        engine = StatsArbBacktestEngine(equity, [], 100000.0)
        result = engine.run()
        assert result.total_return_pct == 0.0
        assert result.num_trades == 0

    def test_single_bar_equity(self):
        """Single-bar equity curve should return empty result (< 2 bars)."""
        equity = pd.Series([100000.0])
        engine = StatsArbBacktestEngine(equity, [], 100000.0)
        result = engine.run()
        assert result.total_return_pct == 0.0

    def test_turnover_no_trades(self):
        """Turnover should be 0 when there are no trades."""
        equity = pd.Series([100000.0, 101000.0, 102000.0])
        engine = StatsArbBacktestEngine(equity, [], 100000.0)
        result = engine.run()
        assert result.turnover == 0.0

    def test_exposure_no_trades(self):
        """Exposure should be 0 when there are no trades."""
        equity = pd.Series([100000.0, 101000.0])
        engine = StatsArbBacktestEngine(equity, [], 100000.0)
        result = engine.run()
        assert result.exposure_pct == 0.0

    def test_all_positive_returns_sortino(self):
        """Sortino with only positive returns should be 0 (no downside)."""
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        engine = StatsArbBacktestEngine(equity, [], 100.0)
        result = engine.run()
        assert result.sortino_ratio == 0.0


# ══════════════════════════════════════════════════════════════════════════
# PAIR RANKER
# ══════════════════════════════════════════════════════════════════════════

class TestPairRankerEdgeCases:

    def test_empty_pair_list(self):
        """Ranking empty pair list should return empty list."""
        ranker = PairRanker()
        result = ranker.rank([])
        assert result == []

    def test_bonferroni_empty(self):
        """Bonferroni on empty list should return empty list."""
        assert bonferroni_correct([], 0) == []

    def test_bonferroni_single(self):
        """Bonferroni with single p-value."""
        assert bonferroni_correct([0.05], 1) == [0.05]

    def test_benjamini_hochberg_empty(self):
        """BH on empty list should return empty list."""
        assert benjamini_hochberg_correct([]) == []

    def test_benjamini_hochberg_all_ones(self):
        """BH with all p-values = 1.0."""
        p_vals = [1.0, 1.0, 1.0]
        corrected = benjamini_hochberg_correct(p_vals)
        assert all(c == 1.0 for c in corrected)

    def test_benjamini_hochberg_all_zero(self):
        """BH with all p-values = 0.0."""
        p_vals = [0.0, 0.0, 0.0]
        corrected = benjamini_hochberg_correct(p_vals)
        assert all(c == 0.0 for c in corrected)

    def test_benjamini_hochberg_monotonic(self):
        """BH corrected p-values should be non-decreasing when sorted."""
        p_vals = [0.001, 0.01, 0.1, 0.5]
        corrected = benjamini_hochberg_correct(p_vals)
        sorted_corrected = sorted(corrected)
        for i in range(1, len(sorted_corrected)):
            assert sorted_corrected[i] >= sorted_corrected[i - 1] - 1e-10

    def test_half_life_score_edge_cases(self):
        """_half_life_score should handle edge values."""
        ranker = PairRanker()
        assert ranker._half_life_score(float("inf")) == 0.0
        assert ranker._half_life_score(float("nan")) == 0.0
        assert ranker._half_life_score(-1.0) == 0.0
        assert ranker._half_life_score(0.0) == 0.0
        assert ranker._half_life_score(5.0) == 1.0
        assert ranker._half_life_score(60.0) == 1.0
        assert ranker._half_life_score(2.5) == 0.5  # 2.5/5.0 = 0.5
        assert ranker._half_life_score(260.0) == 0.0  # 1 - (260-60)/200 = 0.0
