"""Bug: BacktestEngine raises ZeroDivisionError when close price is 0.

Line 81 of engine.py: `qty = buy_cash / price` — if price is 0.0, this crashes.
While zero prices are rare, they can occur with split-adjusted data or corporate actions.
"""

import pandas as pd
import pytest

from backend.backtest.engine import BacktestEngine
from backend.strategies.base import Signal, Strategy


def make_data(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 if p > 0 else 0.0 for p in prices],
            "low": [p * 0.99 if p > 0 else 0.0 for p in prices],
            "close": prices,
            "volume": [10000] * len(prices),
        },
        index=dates,
    )


class BuyOnFirstBar(Strategy):
    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
        if i == 0:
            return Signal.BUY
        return Signal.HOLD


class BuyOnSecondBarAfterZero(Strategy):
    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
        if i == 1 and portfolio.cash > 0:
            return Signal.BUY
        if i == 2 and portfolio.positions.get("TEST", 0) > 0:
            return Signal.SELL
        return Signal.HOLD


class TestBacktestEngineZeroPrice:
    def test_buy_at_zero_price_does_not_crash(self):
        """Price=0 on buy bar causes ZeroDivisionError at qty = buy_cash / price."""
        data = make_data([0.0, 100.0, 110.0])
        try:
            result = BacktestEngine(
                data, BuyOnFirstBar(), symbol="TEST", initial_cash=10000.0
            ).run()
        except ZeroDivisionError:
            pytest.fail(
                "BacktestEngine raised ZeroDivisionError when buying at price=0. "
                "Should handle zero price gracefully (skip trade or use fallback)."
            )

    def test_zero_price_skips_trade_gracefully(self):
        """If price=0, the engine should skip the BUY rather than crash."""
        data = make_data([0.0, 100.0, 110.0])
        result = BacktestEngine(
            data, BuyOnFirstBar(), symbol="TEST", initial_cash=10000.0
        ).run()

        # Bug: currently crashes. If fixed: should have 0 trades
        # because the BUY at price=0 was safely skipped.
        assert len(result.trades) == 0, (
            "When price=0, no trade should be recorded. "
            f"Got {len(result.trades)} trades."
        )
        assert result.metrics["final_equity"] == 10000.0, (
            "Equity should remain at initial cash when no trades occur."
        )

    def test_zero_price_then_normal_trade(self):
        """Zero price on first bar should not prevent later valid trades."""
        data = make_data([0.0, 100.0, 110.0, 110.0])
        result = BacktestEngine(
            data, BuyOnSecondBarAfterZero(), symbol="TEST", initial_cash=10000.0
        ).run()

        # Should have 2 trades (buy at 100, sell at 110) despite first bar having price=0
        assert len(result.trades) == 2, (
            f"Expected 2 trades after zero-price bar, got {len(result.trades)}"
        )
