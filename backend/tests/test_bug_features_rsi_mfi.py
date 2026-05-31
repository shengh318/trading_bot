import numpy as np
import pandas as pd
import pytest

from backend.ml.features import compute_features


def _make_data(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [10000] * len(prices),
        },
        index=dates,
    )


class TestRsiEdgeCases:
    def test_rsi_is_100_when_all_gains(self):
        """RSI must be 100 when every bar closes higher (avg_loss = 0)."""
        prices = [100.0 + i for i in range(20)]
        df = _make_data(prices)
        result = compute_features(df)
        # After 14 bars of consecutive gains, RSI should converge to 100
        tail = result["rsi"].iloc[-3:]
        assert all(tail > 99.0), f"RSI should be near 100 on all gains, got {tail.values}"

    def test_rsi_is_0_when_all_losses(self):
        """RSI must be 0 when every bar closes lower (avg_gain = 0)."""
        prices = [200.0 - i for i in range(20)]
        df = _make_data(prices)
        result = compute_features(df)
        tail = result["rsi"].iloc[-3:]
        assert all(tail < 1.0), f"RSI should be near 0 on all losses, got {tail.values}"

    def test_rsi_is_50_on_flat_prices(self):
        """RSI should be 50 (neutral) when price is completely flat."""
        prices = [100.0] * 20
        df = _make_data(prices)
        result = compute_features(df)
        tail = result["rsi"].iloc[-3:]
        assert all(tail == 50.0), f"RSI should be 50 on flat prices, got {tail.values}"

    def test_rsi_is_50_on_alternating_gains_losses(self):
        """RSI should be 50 when gains and losses are equal."""
        prices = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0,
                  100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0,
                  100.0, 101.0, 100.0, 101.0]
        df = _make_data(prices)
        result = compute_features(df)
        tail = result["rsi"].iloc[-3:]
        # With equal gains/losses, RSI should be 50
        assert all(abs(tail - 50.0) <= 1.0), f"RSI should be ~50 on equal gains/losses, got {tail.values}"


class TestMfiEdgeCases:
    def test_mfi_is_100_when_all_positive_mf(self):
        """MFI must be 100 when every bar has positive money flow."""
        prices = [100.0]
        for i in range(1, 20):
            prices.append(100.0 + i)
        df = _make_data(prices)
        result = compute_features(df)
        tail = result["mfi"].iloc[-3:]
        assert all(tail > 99.0), f"MFI should be near 100 on all positive flow, got {tail.values}"

    def test_mfi_is_0_when_all_negative_mf(self):
        """MFI must be 0 when every bar has negative money flow."""
        prices = [200.0]
        for i in range(1, 20):
            prices.append(200.0 - i)
        df = _make_data(prices)
        result = compute_features(df)
        tail = result["mfi"].iloc[-3:]
        assert all(tail < 1.0), f"MFI should be near 0 on all negative flow, got {tail.values}"

    def test_mfi_is_50_on_flat_prices(self):
        """MFI should be 50 when price is flat."""
        prices = [100.0] * 20
        df = _make_data(prices)
        result = compute_features(df)
        tail = result["mfi"].iloc[-3:]
        assert all(tail == 50.0), f"MFI should be 50 on flat prices, got {tail.values}"
