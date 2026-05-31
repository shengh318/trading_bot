import pandas as pd
import pytest

from backend.backtest.engine import MultiSymbolBacktestEngine
from backend.strategies.base import Strategy, Signal, Portfolio


class _BuyOnceSmall(Strategy):
    """Buys a small fixed amount on bar 0, then holds forever."""
    def __init__(self):
        self.buy_size = 100.0
        self._bought = False

    def init(self, data: pd.DataFrame) -> None:
        self._bought = False

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if not self._bought:
            self._bought = True
            return Signal.BUY
        return Signal.HOLD


def _make_data(prices: list[float], start="2025-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(prices), freq="D")
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


class TestMultiSymbolEquity:
    def test_equity_drops_when_position_excluded_on_missing_date(self):
        """
        BUG: When one symbol lacks data at a union timestamp, its position
        is silently excluded from equity. This causes the equity curve to
        UNDERSTATE true portfolio value.
        """
        # A has 3 days, B has 4 days → union timestamps = Jan 1,2,3,4
        # On Jan 4, A has no data → A's position is excluded from equity
        data = {
            "A": _make_data([100.0, 101.0, 102.0], "2025-01-01"),
            "B": _make_data([50.0, 51.0, 52.0, 53.0], "2025-01-01"),
        }

        engine = MultiSymbolBacktestEngine(data, _BuyOnceSmall, initial_cash=10000.0)
        result = engine.run()

        equity_curve = result.equity_curve

        # Day 0 (Jan 1): cash after buys = 10000 - 100 - 100 = 9800
        #   pos[A] = 1 share @ 100, pos[B] = 2 shares @ 50
        #   equity = 9800 + 1*100 + 2*50 = 10000
        assert equity_curve.iloc[0]["equity"] == 10000.0

        # Day 2 (Jan 3): equity = 9800 + 1*102 + 2*52 = 9800 + 102 + 104 = 10006
        assert equity_curve.iloc[2]["equity"] == 10006.0

        # Day 3 (Jan 4): BUG → equity = 9800 + 0 + 2*53 = 9906 (A excluded)
        #   CORRECT → equity = 9800 + 1*102 + 2*53 = 10008
        # The equity on day 4 must be >= equity on day 3 (both prices went up)
        day4_equity = equity_curve.iloc[3]["equity"]
        day3_equity = equity_curve.iloc[2]["equity"]
        assert day4_equity > day3_equity, (
            f"Equity dropped from {day3_equity} to {day4_equity} on Jan 4 "
            f"(both positions still held, both prices went up). "
            f"Expected > {day3_equity} but got {day4_equity}. "
            f"This means A's position was excluded from equity."
        )

        # The equity on day 4 should be ~10008 (some rounding)
        expected_equity = round(9800 + 1 * 102 + 2 * 53, 2)  # 10008.0
        assert equity_curve.iloc[3]["equity"] == expected_equity, (
            f"Expected equity {expected_equity} but got {equity_curve.iloc[3]['equity']}"
        )
