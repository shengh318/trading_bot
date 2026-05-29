"""Bug: SpreadAnalyzer produces inf/nan z-scores when spread has zero variance.

Line 123 of spread.py: `(spread - mean) / std` — if all spread values are identical,
std is 0, producing infinite z-scores that propagate to downstream calculations.
"""

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.spread import SpreadAnalyzer


class TestSpreadZeroVariance:
    def test_constant_spread_does_not_produce_inf_zscore(self):
        """When spread has zero variance, z-score should be 0, not inf."""
        spread = pd.Series([100.0] * 200)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        # current_zscore should not be inf or nan
        assert np.isfinite(result.current_zscore), (
            f"current_zscore should be finite for constant spread, got {result.current_zscore}"
        )
        assert result.current_zscore == pytest.approx(0.0, abs=1e-6), (
            f"current_zscore should be 0 for constant spread, got {result.current_zscore}"
        )

    def test_constant_spread_zscore_series_has_no_inf(self):
        """All z-score values should be finite for constant spread."""
        spread = pd.Series([50.0] * 300)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        zscore_series = result.zscore_series
        assert zscore_series is not None
        assert len(zscore_series) > 0
        assert all(np.isfinite(zscore_series)), (
            "zscore_series contains inf/nan values for constant spread"
        )
        assert all(abs(z) < 1e-6 for z in zscore_series), (
            "All z-scores should be ~0 for constant spread"
        )

    def test_zero_variance_std_in_spread_series(self):
        """Half-life, Hurst, and ADF should handle zero-variance spread."""
        spread = pd.Series(np.ones(500) * 42.0)
        analyzer = SpreadAnalyzer(spread)
        result = analyzer.run()

        # These should all be finite values
        assert np.isfinite(result.half_life) or result.half_life == float("inf"), (
            f"half_life should be finite or inf, got {result.half_life}"
        )
        assert np.isfinite(result.hurst_exponent), (
            f"hurst_exponent should be finite, got {result.hurst_exponent}"
        )
        assert np.isfinite(result.adf_statistic), (
            f"adf_statistic should be finite, got {result.adf_statistic}"
        )
        assert np.isfinite(result.adf_pvalue), (
            f"adf_pvalue should be finite, got {result.adf_pvalue}"
        )
