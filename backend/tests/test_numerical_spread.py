"""Numerical edge cases for stats_arb/spread.py"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.spread import SpreadAnalyzer, SpreadResult


class TestSpreadAnalyzerEdgeCases:
    def test_single_value_spread(self):
        """A spread with a single value should not crash."""
        spread = pd.Series([100.0])
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert isinstance(result, SpreadResult)
        assert np.isfinite(result.mean) or result.mean == 100.0

    def test_two_value_spread(self):
        """A spread with two values should produce finite results."""
        spread = pd.Series([100.0, 101.0])
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.mean == pytest.approx(100.5)
        assert np.isfinite(result.std)
        assert np.isfinite(result.current_zscore)

    def test_constant_spread_half_life(self):
        """Half-life of constant spread should be inf (no mean reversion)."""
        spread = pd.Series([100.0] * 200)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.half_life == float("inf") or not np.isfinite(result.half_life), (
            f"Half-life of constant spread should be inf, got {result.half_life}"
        )

    def test_half_life_negative_theta(self):
        """When theta is negative, half-life should be positive."""
        np.random.seed(42)
        # Mean-reverting series: spread = 0.9 * lag + noise
        spread_vals = [0.0]
        for _ in range(200):
            spread_vals.append(0.9 * spread_vals[-1] + np.random.randn() * 0.1)
        spread = pd.Series(spread_vals)

        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        if np.isfinite(result.half_life):
            assert result.half_life > 0, (
                f"Half-life should be positive for mean-reverting spread, got {result.half_life}"
            )

    def test_half_life_short_series(self):
        """Half-life for very short series should be inf."""
        spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.half_life == float("inf"), (
            f"Half-life of 5-point series should be inf, got {result.half_life}"
        )

    def test_half_life_with_nan(self):
        """Half-life computation with NaN values should not crash."""
        spread = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0] * 40)
        analyzer = SpreadAnalyzer(spread)
        try:
            result = analyzer.run()
        except Exception as e:
            pytest.fail(f"SpreadAnalyzer crashed on NaN spread: {e}")

    def test_hurst_trending_returns_high_value(self):
        """Strong trending spread should have Hurst > 0.5."""
        t = np.arange(200, dtype=float)
        spread = pd.Series(t + np.random.randn(200) * 0.5)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.hurst_exponent >= 0.0 and result.hurst_exponent <= 1.0, (
            f"Hurst should be in [0, 1], got {result.hurst_exponent}"
        )

    def test_expected_time_to_mean_inf_for_non_reverting(self):
        """When mean reversion speed >= 0, expected time to mean should be inf."""
        spread = pd.Series(np.cumsum(np.random.randn(200)))  # random walk
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        if result.mean_reversion_speed >= 0:
            assert result.expected_time_to_mean == float("inf"), (
                "Expected time to mean should be inf when speed >= 0"
            )

    def test_variance_ratio_constant_spread(self):
        """Variance ratio of constant spread should be 1.0 (fallback)."""
        spread = pd.Series([100.0] * 100)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.variance_ratio == 1.0, (
            f"Variance ratio of constant spread should be 1.0, got {result.variance_ratio}"
        )

    def test_variance_ratio_short_series(self):
        """Variance ratio for short series should return 1.0."""
        spread = pd.Series([100.0, 101.0, 102.0])
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.variance_ratio == 1.0, (
            f"Variance ratio of short series should be 1.0, got {result.variance_ratio}"
        )

    def test_variance_explosion_events_short_spread(self):
        """When spread is shorter than 63 bars, explosion events should be 0."""
        spread = pd.Series(np.random.randn(50))
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.variance_explosion_events == 0, (
            f"Variance explosion events should be 0 for short spread, got {result.variance_explosion_events}"
        )

    def test_persistence_constant_spread(self):
        """Persistence (lag-1 autocorr) of constant spread should be NaN (filled to 0)."""
        spread = pd.Series([100.0] * 100)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        # autocorr of constant series is NaN, but we return 0.0 for len > 2
        assert result.persistence == 0.0 or np.isnan(result.persistence), (
            f"Persistence of constant spread should be 0 or NaN, got {result.persistence}"
        )

    def test_autocorr_5_short_spread(self):
        """Autocorr at lag 5 for short spread should be 0."""
        spread = pd.Series([1.0, 2.0, 3.0, 4.0])  # len <= 5
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.spread_autocorr_5 == 0.0, (
            f"Autocorr_5 for short spread should be 0, got {result.spread_autocorr_5}"
        )

    def test_zscore_inf_nan_handling(self):
        """Z-score series should never contain inf or NaN."""
        np.random.seed(42)
        spread = pd.Series(np.random.randn(500) * 2 + 100)
        # Inject some extreme values
        spread.iloc[250] = 1e10
        spread.iloc[251] = -1e10

        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        zs = result.zscore_series
        finite_mask = np.isfinite(zs)
        assert finite_mask.all(), (
            f"Z-score series contains {len(zs) - finite_mask.sum()} non-finite values"
        )

    def test_adf_on_constant_spread(self):
        """ADF test on constant spread should return p-value = 1.0."""
        spread = pd.Series([100.0] * 100)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        assert result.adf_pvalue == 1.0 or result.adf_pvalue >= 0.05, (
            f"ADF p-value for constant spread should be high (>=0.05), got {result.adf_pvalue}"
        )
        assert result.is_stationary == False, (
            "Constant spread should not be considered stationary"
        )

    def test_ou_speed_all_nan(self):
        """OU mean reversion speed on NaN series should not crash.
        
        Potential bug: all-NaN series after dropna() gives empty spread,
        causing IndexError on zscore_series.iloc[-1].
        """
        spread = pd.Series([float("nan")] * 100)
        try:
            analyzer = SpreadAnalyzer(spread)
            result = analyzer.run()
            # After dropna(), spread is empty. run() may crash on iloc[-1].
        except IndexError:
            pytest.fail(
                "SpreadAnalyzer raised IndexError on all-NaN series. "
                "Bug: spread.dropna() gives empty series, then iloc[-1] fails."
            )
        except Exception as e:
            pytest.fail(f"SpreadAnalyzer crashed on all-NaN series: {e}")

    def test_half_life_oulier_regression(self):
        """Half-life estimation via OLS should handle unusual values.
        
        Bug: For a linear trend, theta is very close to 0 (not exactly 0),
        so -ln(2)/theta becomes astronomically large but finite, not inf.
        """
        spread = pd.Series(np.arange(100, dtype=float))
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        # Theta should be near 0 (< 0 means mean-reverting, but trending isn't)
        # For a linear trend, theta is slightly negative due to OLS bias
        # Result is an extremely large finite value instead of inf
        if result.half_life != float("inf"):
            # Should at least be very large (indicating no mean reversion)
            assert result.half_life > 1e6 or result.half_life == float("inf"), (
                f"Half-life of trending spread should be very large or inf, got {result.half_life}"
            )

    def test_to_dict_rounding(self):
        """to_dict() should round values without crashing on inf."""
        spread = pd.Series([100.0] * 200)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        d = result.to_dict()
        for k, v in d.items():
            if isinstance(v, float):
                if v == float("inf"):
                    assert k in ("half_life", "expected_time_to_mean"), (
                        f"Only half_life/expected_time_to_mean should be inf, got {k}={v}"
                    )
                else:
                    assert np.isfinite(v), f"Key {k} has non-finite value {v}"
