"""Numerical edge cases for stats_arb/correlation.py"""

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.correlation import CorrelationAnalyzer, _rolling_correlation, RollingCorrelationResult


class TestRollingCorrelation:
    def test_identical_series(self):
        """Perfectly correlated series should give correlation = 1.0."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": np.linspace(100, 200, 100),
            "B": np.linspace(100, 200, 100),
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20])
        results = analyzer.run()
        for w, r in results.items():
            assert r.current == pytest.approx(1.0, abs=0.01), (
                f"Correlation of identical series should be ~1.0, got {r.current}"
            )

    def test_inverse_series(self):
        """Perfectly inverse returns should give correlation = -1.0.
        
        Note: linear inverse prices (200→100 vs 100→200) do NOT produce
        inverse returns due to percentage math. We use exponential
        series with opposite daily returns instead.
        """
        np.random.seed(42)
        n = 100
        noise = np.random.randn(n) * 0.01
        returns_a = 0.0005 + noise
        returns_b = -0.0005 - noise  # exact inverse of returns_a
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        prices = pd.DataFrame({
            "A": 100 * (1 + returns_a).cumprod(),
            "B": 100 * (1 + returns_b).cumprod(),
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20])
        results = analyzer.run()
        for w, r in results.items():
            assert r.current == pytest.approx(-1.0, abs=0.01), (
                f"Correlation of inverse returns should be ~-1.0, got {r.current}"
            )

    def test_constant_series_a(self):
        """Constant series A has 0 variance, correlation should be undefined (NaN crash)."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": [100.0] * 100,
            "B": np.linspace(100, 200, 100),
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20])
        try:
            results = analyzer.run()
        except Exception as e:
            pytest.fail(f"CorrelationAnalyzer crashed on constant series: {e}")

    def test_single_window(self):
        """Single data point per window should not crash."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        prices = pd.DataFrame({
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [50.0, 51.0, 52.0, 53.0, 54.0],
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20])  # window longer than data
        try:
            results = analyzer.run()
        except Exception as e:
            pytest.fail(f"CorrelationAnalyzer crashed with window > data: {e}")

    def test_correlation_drawdown_zero_division(self):
        """Correlation drawdown should handle division by zero (cummax=0)."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": np.linspace(100, 200, 100),
            "B": np.linspace(100, 200, 100),
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20])
        results = analyzer.run()
        for w, r in results.items():
            assert np.isfinite(r.max_correlation_drawdown), (
                f"Correlation drawdown should be finite, got {r.max_correlation_drawdown}"
            )
            assert 0.0 <= r.max_correlation_drawdown <= 1.0, (
                f"Correlation drawdown should be in [0, 1], got {r.max_correlation_drawdown}"
            )

    def test_stability_score_bounds(self):
        """Stability score should be in [0, 1]."""
        dates = pd.date_range("2025-01-01", periods=200, freq="D")
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.randn(200)) + 100,
            "B": np.cumsum(np.random.randn(200)) + 50,
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20, 63])
        results = analyzer.run()
        for w, r in results.items():
            assert 0.0 <= r.stability_score <= 1.0, (
                f"Stability score for window {w} should be in [0, 1], got {r.stability_score}"
            )

    def test_threshold_crossings_count(self):
        """Threshold crossings should be non-negative integers."""
        dates = pd.date_range("2025-01-01", periods=200, freq="D")
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.randn(200)) + 100,
            "B": np.cumsum(np.random.randn(200)) + 50,
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[63])
        results = analyzer.run()
        for r in results.values():
            for k, v in r.threshold_crossings.items():
                assert isinstance(v, int) and v >= 0, (
                    f"Threshold crossing {k} should be non-negative int, got {v}"
                )

    def test_correlation_collapse(self):
        """Correlation collapse flag should be boolean."""
        dates = pd.date_range("2025-01-01", periods=200, freq="D")
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.randn(200)) + 100,
            "B": np.cumsum(np.random.randn(200)) + 50,
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[63])
        results = analyzer.run()
        for r in results.values():
            assert isinstance(r.correlation_collapse, bool), (
                f"correlation_collapse should be bool, got {type(r.correlation_collapse)}"
            )

    def test_overall_slope_with_flat_correlation(self):
        """Stable correlation should have near-zero slope."""
        dates = pd.date_range("2025-01-01", periods=200, freq="D")
        # Two identical series -> constant 1.0 correlation
        prices = pd.DataFrame({
            "A": np.linspace(100, 200, 200),
            "B": np.linspace(100, 200, 200),
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[20])
        results = analyzer.run()
        for r in results.values():
            assert abs(r.overall_slope) < 0.1, (
                f"Slope of stable correlation should be near 0, got {r.overall_slope}"
            )

    def test_rolling_slope_in_rolling_correlation(self):
        """Rolling slope (63d window) should not produce inf/nan."""
        dates = pd.date_range("2025-01-01", periods=300, freq="D")
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.randn(300)) + 100,
            "B": np.cumsum(np.random.randn(300)) + 50,
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[63])
        results = analyzer.run()
        for r in results.values():
            rs = r.rolling_slope_series
            if len(rs) > 0:
                assert np.all(np.isfinite(rs)), (
                    "Rolling slope series should have all finite values"
                )

    def test_empty_prices_returns_empty(self):
        """Empty price data should not crash."""
        prices = pd.DataFrame({"A": [], "B": []})
        analyzer = CorrelationAnalyzer(prices, windows=[20])
        try:
            results = analyzer.run()
        except Exception as e:
            pytest.fail(f"CorrelationAnalyzer crashed on empty data: {e}")

    def test_to_dict_rounding(self):
        """to_dict should round values without crashing."""
        dates = pd.date_range("2025-01-01", periods=200, freq="D")
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.randn(200)) + 100,
            "B": np.cumsum(np.random.randn(200)) + 50,
        }, index=dates)
        analyzer = CorrelationAnalyzer(prices, windows=[63])
        results = analyzer.run()
        for r in results.values():
            d = r.to_dict()
            for k, v in d.items():
                if isinstance(v, float):
                    assert np.isfinite(v), f"to_dict key {k} has non-finite value {v}"


class TestRollingCorrelationFunction:
    def test_rolling_correlation_short_data(self):
        """Short data returns empty series."""
        a = pd.Series([100.0, 101.0])
        b = pd.Series([50.0, 51.0])
        result = _rolling_correlation(a, b, window=20)
        assert len(result) == 0

    def test_rolling_correlation_constant_returns(self):
        """Constant returns (all 0) should produce empty or NaN correlation."""
        a = pd.Series([100.0] * 50)
        b = pd.Series([50.0] * 50)
        result = _rolling_correlation(a, b, window=20)
        # pct_change of constant series is 0, corr of 0s is NaN -> filtered out
        assert len(result) == 0 or result.isnull().all(), (
            "Constant returns should produce no valid correlation"
        )

    def test_rolling_correlation_nan_handling(self):
        """Rolling corr with NaN should not crash."""
        a = pd.Series(np.random.randn(100))
        b = pd.Series(np.random.randn(100))
        b.iloc[50] = float("nan")
        try:
            result = _rolling_correlation(a, b, window=20)
        except Exception as e:
            pytest.fail(f"_rolling_correlation crashed with NaN: {e}")
