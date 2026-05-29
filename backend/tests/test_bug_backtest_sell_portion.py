"""Bug: BacktestEngine uses stale `qty` from last BUY when strategy lacks `sell_portion`.

When a strategy does not define `sell_portion` and returns Signal.SELL, the engine
skips the `qty` assignment (line 112-114 of engine.py). The variable `qty` is then
unintentionally reused from the previous BUY iteration. For DCA strategies that buy
multiple times, this means only the last buy quantity is sold, leaving residual shares.

Expected: SELL without `sell_portion` should clear the full position (qty = current_pos).
Actual: SELL reuses the stale `qty` from the most recent BUY.
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
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [10000] * len(prices),
        },
        index=dates,
    )


class DCAThenSellNoSellPortion(Strategy):
    """Buys with half cash on first two bars, sells all on third bar.
    
    Does NOT define `sell_portion` — the engine must fall back to
    selling the full position (current_pos).
    """
    buy_size = 5000.0

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
        if i < 2 and portfolio.cash > 0:
            return Signal.BUY
        if i == 2 and portfolio.positions.get("TEST", 0) > 0:
            return Signal.SELL
        return Signal.HOLD


class TestBacktestEngineSellPortion:
    """Bug: engine should sell full position when strategy lacks sell_portion."""

    def test_dca_sell_clears_full_position(self):
        data = make_data([100.0, 110.0, 120.0, 130.0])
        result = BacktestEngine(
            data, DCAThenSellNoSellPortion(), symbol="TEST", initial_cash=10000.0
        ).run()

        # Should be 3 trades: 2 buys + 1 sell
        assert len(result.trades) == 3, (
            f"Expected 3 trades (2 buys + 1 sell), got {len(result.trades)}"
        )

        sells = result.trades[result.trades["side"] == "sell"]
        buy_qty = result.trades[result.trades["side"] == "buy"]["qty"].sum()
        sell_qty = sells["qty"].sum()

        # BUG: sell_qty should equal buy_qty (95.45), but stale qty from last BUY
        # means only the last buy amount (45.45 at bar 1) is sold.
        assert sell_qty == pytest.approx(buy_qty, abs=0.1), (
            f"Sell quantity ({sell_qty:.2f}) should equal total buy quantity "
            f"({buy_qty:.2f}). Stale `qty` bug causes partial sell."
        )

        # Position should be fully cleared — no residual shares
        # Check via final snapshot equity being purely cash-based
        last_snapshot = result.equity_curve.iloc[-1]
        assert last_snapshot["equity"] == pytest.approx(last_snapshot["cash"], abs=0.01), (
            f"Expected equity ({last_snapshot['equity']:.2f}) to equal cash "
            f"({last_snapshot['cash']:.2f}) after full exit, but residual position remains."
        )

    def test_single_buy_sell_full_clears_position(self):
        """Single BUY + SELL should also clear, not rely on stale qty trick."""
        data = make_data([100.0, 110.0])

        class BuyHalfSellFull(Strategy):
            buy_size = 5000.0

            def init(self, data: pd.DataFrame) -> None:
                pass

            def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
                if i == 0:
                    return Signal.BUY
                if i == 1 and portfolio.positions.get("TEST", 0) > 0:
                    return Signal.SELL
                return Signal.HOLD

        result = BacktestEngine(
            data, BuyHalfSellFull(), symbol="TEST", initial_cash=10000.0
        ).run()

        last_snapshot = result.equity_curve.iloc[-1]
        assert last_snapshot["equity"] == pytest.approx(
            last_snapshot["cash"], abs=0.01
        ), "Position not fully cleared after single buy + sell"

    def test_multi_dca_three_buys_then_sell(self):
        """Three DCA buys followed by sell — stale qty from last buy is wrong."""
        data = make_data([100.0, 105.0, 110.0, 115.0, 120.0])

        class TripleDCA(Strategy):
            buy_size = 3000.0

            def init(self, data: pd.DataFrame) -> None:
                pass

            def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
                if i < 3 and portfolio.cash > 0:
                    return Signal.BUY
                if i == 3 and portfolio.positions.get("TEST", 0) > 0:
                    return Signal.SELL
                return Signal.HOLD

        result = BacktestEngine(
            data, TripleDCA(), symbol="TEST", initial_cash=10000.0
        ).run()

        buy_qty = result.trades[result.trades["side"] == "buy"]["qty"].sum()
        sell_qty = result.trades[result.trades["side"] == "sell"]["qty"].sum()

        assert sell_qty == pytest.approx(buy_qty, abs=0.1), (
            f"Triple DCA: sell qty ({sell_qty:.2f}) != total buy qty ({buy_qty:.2f}). "
            "Stale qty bug causes incomplete exit."
        )
