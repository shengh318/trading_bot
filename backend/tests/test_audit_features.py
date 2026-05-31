"""Audit tests for ML features and API edge cases.

Identifies edge cases in:
- compute_features (RSI, ADX, MFI with constant prices)
- clean_features / safe_divide
- _safe helper in ml_routes
- Correlation API endpoint edge cases
"""

import math
import numpy as np
import pandas as pd
import pytest

from backend.ml.features import (
    compute_features, clean_features, safe_divide, safe_pct_change,
    get_feature_columns, _hurst_exponent, merge_context_features,
)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION EDGE CASES
# ══════════════════════════════════════════════════════════════════════════

def make_ohlcv(close_prices=None, length=100, start="2025-01-01"):
    """Create standard OHLCV DataFrame."""
    if close_prices is None:
        np.random.seed(42)
        close_prices = 100.0 + np.cumsum(np.random.randn(length))
    dates = pd.date_range(start, periods=len(close_prices), freq="D")
    return pd.DataFrame({
        "open": close_prices,
        "high": [c * 1.01 for c in close_prices],
        "low": [c * 0.99 for c in close_prices],
        "close": close_prices,
        "volume": [10000] * len(close_prices),
    }, index=dates)


class TestComputeFeaturesEdgeCases:

    def test_constant_prices_rsi(self):
        """RSI should be 50 for constant prices (no gains/losses)."""
        data = make_ohlcv(close_prices=[100.0] * 100)
        result = compute_features(data)
        rsi_last = result["rsi"].iloc[-1]
        assert rsi_last == 50.0, f"Expected 50 (neutral RSI), got {rsi_last}"

    def test_constant_prices_macd(self):
        """MACD should be 0 for constant prices."""
        data = make_ohlcv(close_prices=[100.0] * 100)
        result = compute_features(data)
        macd_last = result["macd"].iloc[-1]
        assert macd_last == 0.0

    def test_constant_prices_bb_width(self):
        """BB width should be 0 for constant prices."""
        data = make_ohlcv(close_prices=[100.0] * 100)
        result = compute_features(data)
        bb_width = result["bb_width"].iloc[-1]
        assert bb_width == 0.0

    def test_constant_prices_adx(self):
        """ADX should handle constant prices without crashing."""
        data = make_ohlcv(close_prices=[100.0] * 100)
        result = compute_features(data)
        assert not result["adx"].isna().all()
        # ADX should be a finite number
        assert np.isfinite(result["adx"].iloc[-1])

    def test_constant_prices_mfi(self):
        """MFI should handle constant prices."""
        data = make_ohlcv(close_prices=[100.0] * 100)
        result = compute_features(data)
        assert "mfi" in result.columns
        # MFI may be 0 or NaN for constant prices, but shouldn't crash
        assert np.isfinite(result["mfi"].iloc[-1])

    def test_very_short_dataframe(self):
        """Very short DataFrame should still produce features (with many NaNs)."""
        data = make_ohlcv(close_prices=[100.0, 101.0, 102.0])
        result = compute_features(data)
        assert len(result) == 3
        # Most features should exist (may be 0 after clean_features)
        assert "rsi" in result.columns
        assert "macd" in result.columns

    def test_single_bar_dataframe(self):
        """Single bar should not crash."""
        data = make_ohlcv(close_prices=[100.0])
        result = compute_features(data)
        assert len(result) == 1

    def test_non_datetime_index(self):
        """Non-datetime index should set day_of_week to 0 (Monday)."""
        data = make_ohlcv(close_prices=[100.0, 101.0, 102.0])
        data.index = pd.RangeIndex(len(data))
        result = compute_features(data)
        assert result["day_of_week"].iloc[0] == 0

    def test_all_features_present(self):
        """All expected features should be present after compute_features."""
        data = make_ohlcv(length=100)
        result = compute_features(data)
        expected_features = [
            "ret_1", "ret_5", "ret_21", "vol_5", "vol_21",
            "sma_20", "sma_50", "close_sma_20", "close_sma_50",
            "vol_ma_20", "vol_ratio", "rsi", "macd", "macd_signal", "macd_hist",
            "atr", "highest_20", "lowest_20", "price_position",
            "day_of_week", "vol_ratio_5_21",
            "mom_10", "mom_20", "mom_60",
            "bb_width", "bb_pct_b", "adx", "plus_di", "minus_di",
            "obv_ratio", "mfi",
            "ret_1_lag1", "ret_1_lag2", "ret_1_lag3", "rsi_lag1", "vol_21_lag1",
            "hurst", "choppiness",
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_no_nan_in_output(self):
        """After clean_features, there should be no NaN values."""
        data = make_ohlcv(length=100)
        result = compute_features(data)
        assert not result.isna().any().any(), "Found NaN values after clean_features"

    def test_no_inf_in_output(self):
        """After clean_features, there should be no inf values."""
        data = make_ohlcv(length=100)
        result = compute_features(data)
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()


class TestSafeDivide:

    def test_zero_denominator(self):
        """Division by zero should produce NaN."""
        num = pd.Series([1.0, 2.0, 3.0])
        denom = pd.Series([0.0, 0.0, 0.0])
        result = safe_divide(num, denom)
        assert result.isna().all()

    def test_mixed_denominator(self):
        """Mixed zero/non-zero denominators."""
        num = pd.Series([1.0, 2.0, 3.0])
        denom = pd.Series([0.0, 1.0, 2.0])
        result = safe_divide(num, denom)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 2.0
        assert result.iloc[2] == 1.5


class TestCleanFeatures:

    def test_inf_replaced(self):
        """inf values should be replaced with fill_val."""
        df = pd.DataFrame({"a": [1.0, float("inf"), float("-inf")]})
        result = clean_features(df, fill_val=0.0)
        assert result["a"].iloc[0] == 1.0
        assert result["a"].iloc[1] == 0.0
        assert result["a"].iloc[2] == 0.0

    def test_nan_replaced(self):
        """NaN values should be replaced with fill_val."""
        df = pd.DataFrame({"a": [1.0, float("nan")]})
        result = clean_features(df, fill_val=-1.0)
        assert result["a"].iloc[0] == 1.0
        assert result["a"].iloc[1] == -1.0


class TestHurstExponent:

    def test_constant_input(self):
        """Hurst of constant input should be 0.5."""
        ts = np.ones(50)
        h = _hurst_exponent(ts)
        assert h == 0.5

    def test_short_input(self):
        """Hurst of short input (< 20) should be 0.5."""
        ts = np.random.randn(10)
        h = _hurst_exponent(ts)
        assert h == 0.5

    def test_random_walk(self):
        """Hurst of random walk should be in [0, 1] (R/S method can vary)."""
        np.random.seed(42)
        ts = np.cumsum(np.random.randn(500))
        h = _hurst_exponent(ts)
        assert 0.0 <= h <= 1.0

    def test_bounds(self):
        """Hurst should be bounded [0, 1]."""
        np.random.seed(42)
        ts = np.cumsum(np.random.randn(100))
        h = _hurst_exponent(ts)
        assert 0.0 <= h <= 1.0


class TestMergeContextFeatures:

    def test_empty_context(self):
        """Empty context dict should return original df unchanged."""
        df = make_ohlcv(length=50)
        features = compute_features(df)
        feat_cols = get_feature_columns(features)
        original = features[feat_cols].copy()
        result = merge_context_features(features, {})
        pd.testing.assert_frame_equal(result[feat_cols], original)

    def test_context_prefix(self):
        """Context features should be prefixed with symbol name."""
        df = make_ohlcv(length=50)
        ctx_df = make_ohlcv(length=50)
        result = merge_context_features(df, {"SPY": ctx_df})
        # Should have SPY_ prefixed columns
        spy_cols = [c for c in result.columns if c.startswith("SPY_")]
        assert len(spy_cols) > 0


class TestGetFeatureColumns:

    def test_excluded_columns_not_in_features(self):
        """Excluded columns should not appear in feature list."""
        data = make_ohlcv(length=100)
        result = compute_features(data)
        result["symbol"] = "TEST"
        result["target"] = 0
        features = get_feature_columns(result)
        assert "open" not in features
        assert "close" not in features
        assert "symbol" not in features
        assert "target" not in features
