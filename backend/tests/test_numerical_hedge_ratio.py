"""Numerical edge cases for stats_arb/hedge_ratio.py"""

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.hedge_ratio import (
    estimate_ols,
    HedgeRatioEstimator,
    KalmanFilterHedge,
)


class TestEstimateOLS:
    def test_perfect_correlation(self):
        """Perfectly correlated series should give beta=2.0."""
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        beta = estimate_ols(a, b, add_const=True)
        assert beta == pytest.approx(2.0, abs=0.01), (
            f"Perfect correlation: beta should be ~2.0, got {beta}"
        )

    def test_no_correlation(self):
        """Uncorrelated series should give beta near 0.
        
        Note: With constant B, OLS can give any beta since (B - mean(B)) = 0.
        The intercept captures the full mean of A, so beta is degenerate.
        """
        a = np.array([10.0, 20.0, 10.0, 20.0, 10.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.random.seed(42)
        a = np.random.randn(100) + 100
        b = np.random.randn(100) + 50  # uncorrelated
        beta = estimate_ols(a, b, add_const=True)
        # Uncorrelated: beta should be near 0
        assert abs(beta) < 2.0, (
            f"Uncorrelated series: beta should be near 0, got {beta}"
        )

    def test_negative_relationship(self):
        """Inverse relationship should give negative beta."""
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        beta = estimate_ols(a, b, add_const=True)
        assert beta < 0, (
            f"Inverse relationship: beta should be negative, got {beta}"
        )

    def test_no_intercept(self):
        """Without intercept, beta should go through origin.
        
        Note: When data goes through origin (a = 2*b), both with and without
        intercept give the same slope. Use data with a non-zero intercept.
        """
        a = np.array([15.0, 25.0, 35.0, 45.0])  # = 2*b + 5
        b = np.array([5.0, 10.0, 15.0, 20.0])
        beta_no_const = estimate_ols(a, b, add_const=False)
        beta_with_const = estimate_ols(a, b, add_const=True)
        # Without intercept, beta is forced through origin, so it differs
        assert abs(beta_no_const - beta_with_const) > 0.01, (
            f"Without intercept, beta ({beta_no_const}) should differ from with-intercept "
            f"version ({beta_with_const}) for data with non-zero intercept"
        )

    def test_single_element(self):
        """Single element arrays should not crash."""
        a = np.array([10.0])
        b = np.array([5.0])
        try:
            beta = estimate_ols(a, b)
            assert np.isfinite(beta), f"Beta should be finite, got {beta}"
        except Exception as e:
            pytest.fail(f"estimate_ols crashed on single element: {e}")

    def test_all_zeros(self):
        """All zeros should not crash."""
        a = np.zeros(10)
        b = np.zeros(10)
        try:
            beta = estimate_ols(a, b)
            assert np.isfinite(beta) or beta == 0.0, (
                f"Beta should be 0 or finite for all zeros, got {beta}"
            )
        except Exception as e:
            pytest.fail(f"estimate_ols crashed on all zeros: {e}")

    def test_nan_in_input(self):
        """NaN input values should not crash (though result is undefined)."""
        a = np.array([10.0, float("nan"), 30.0, 40.0])
        b = np.array([5.0, 10.0, float("nan"), 20.0])
        try:
            beta = estimate_ols(a, b)
        except Exception as e:
            pytest.fail(f"estimate_ols crashed on NaN values: {e}")

    def test_inf_in_input(self):
        """Inf input values should not crash."""
        a = np.array([10.0, float("inf"), 30.0, 40.0])
        b = np.array([5.0, 10.0, 15.0, float("inf")])
        try:
            beta = estimate_ols(a, b)
        except Exception as e:
            pytest.fail(f"estimate_ols crashed on inf values: {e}")

    def test_robust_huber(self):
        """Huber robust regression should produce a beta."""
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 1000.0])  # outlier
        b = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 500.0])
        beta = estimate_ols(a, b, robust=True)
        assert np.isfinite(beta), f"Huber beta should be finite, got {beta}"


class TestHedgeRatioEstimator:
    def test_rolling_ols_identical_prices(self):
        """Rolling OLS on identical price series should give beta=1."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": np.linspace(100, 200, 100),
            "B": np.linspace(100, 200, 100),
        }, index=dates)
        estimator = HedgeRatioEstimator(prices)
        result = estimator.rolling_ols(window=30)
        assert np.allclose(result.beta, 1.0, atol=0.01), (
            f"Rolling OLS on identical series should give beta≈1, got {result.beta[:5]}"
        )

    def test_rolling_ols_constant_prices(self):
        """Rolling OLS on constant prices: beta should be degenerate (A = intercept + 0*B).
        
        Note: With constant A=100 and B=50, many solutions satisfy the OLS,
        e.g., A = 0 + 2*50. The result depends on numerical precision.
        """
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": [100.0] * 100,
            "B": [50.0] * 100,
        }, index=dates)
        estimator = HedgeRatioEstimator(prices)
        result = estimator.rolling_ols(window=30)
        betas = np.array(result.beta)
        finite_betas = betas[np.isfinite(betas)]
        # With both series constant, OLS is degenerate. Beta could be anything
        # that satisfies A ≈ intercept + beta * B. Since A=100, B=50:
        # Valid solutions include beta=2, intercept=0, or beta=0, intercept=100
        # Just verify no crash and all betas are the same (stable degenerate solution)
        if len(finite_betas) > 1:
            assert np.allclose(finite_betas, finite_betas[0], atol=1e-10), (
                f"Degenerate OLS on constant data should produce stable beta, got varying values"
            )

    def test_rolling_ols_short_data(self):
        """Rolling OLS with window larger than data should return empty."""
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "A": np.random.randn(10) + 100,
            "B": np.random.randn(10) + 50,
        }, index=dates)
        estimator = HedgeRatioEstimator(prices)
        result = estimator.rolling_ols(window=63)
        assert len(result.beta_series) == 0, (
            f"Rolling OLS with window > data should be empty, got {len(result.beta_series)}"
        )


class TestKalmanFilterHedge:
    def test_kalman_identical_series(self):
        """Kalman filter on identical series: beta should → 1."""
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n))
        y = x.copy()
        kf = KalmanFilterHedge(lambda_=0.99)
        betas = kf.fit(y, x)
        # Last values should be near 1
        assert np.mean(betas[-50:]) == pytest.approx(1.0, abs=0.1), (
            f"Kalman beta on identical series should be ~1, got {np.mean(betas[-50:])}"
        )

    def test_kalman_short_series(self):
        """Short series (< 10) should return NaN."""
        kf = KalmanFilterHedge()
        betas = kf.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert np.all(np.isnan(betas)), (
            f"Short series should return NaN, got {betas}"
        )

    def test_kalman_linear_relationship(self):
        """Kalman filter should track a linear relationship y = 2x + 1."""
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n))
        y = 2.0 * x + 1.0 + np.random.randn(n) * 0.1
        kf = KalmanFilterHedge(lambda_=0.99)
        betas = kf.fit(y, x)
        assert np.mean(betas[-50:]) == pytest.approx(2.0, abs=0.2), (
            f"Kalman beta should track ~2.0, got {np.mean(betas[-50:])}"
        )

    def test_kalman_constant_input(self):
        """Kalman filter with constant input should not crash."""
        x = np.ones(100) * 50.0
        y = np.ones(100) * 100.0
        kf = KalmanFilterHedge()
        try:
            betas = kf.fit(y, x)
        except Exception as e:
            pytest.fail(f"Kalman filter crashed on constant input: {e}")

    def test_kalman_all_nan_output(self):
        """Kalman filter with NaN in output should not crash."""
        x = np.random.randn(100)
        y = np.array([float("nan")] * 100)
        kf = KalmanFilterHedge()
        try:
            betas = kf.fit(y, x)
        except Exception as e:
            pytest.fail(f"Kalman filter crashed on NaN output: {e}")

    def test_kalman_fit_dataframe(self):
        """Kalman fit_dataframe should return proper HedgeRatioResult."""
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "A": np.cumsum(np.random.randn(100)) + 100,
            "B": np.cumsum(np.random.randn(100)) + 50,
        }, index=dates)
        kf = KalmanFilterHedge()
        result = kf.fit_dataframe(prices)
        assert result.method == "kalman_rls"
        assert len(result.beta_series) == 100
        assert result.intercept == 0.0

    def test_hedge_ratio_result_to_dict(self):
        """HedgeRatioResult.to_dict should round values without crashing."""
        from backend.stats_arb.hedge_ratio import HedgeRatioResult
        result = HedgeRatioResult(beta=1.5, intercept=0.5, method="ols")
        d = result.to_dict()
        assert d["method"] == "ols"
        assert isinstance(d["beta"], float)
        assert isinstance(d["intercept"], float)

    def test_hedge_ratio_result_to_dict_array_beta(self):
        """to_dict should handle array beta by taking last value."""
        from backend.stats_arb.hedge_ratio import HedgeRatioResult
        result = HedgeRatioResult(
            beta=np.array([1.0, 1.5, 2.0]),
            intercept=0.5, method="rolling_ols",
        )
        d = result.to_dict()
        assert d["beta"] == pytest.approx(2.0, abs=0.01), (
            f"to_dict with array beta should use last value, got {d['beta']}"
        )
