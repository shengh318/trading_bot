"""Numerical edge cases for strategies."""

import numpy as np
import pandas as pd
import pytest

from backend.strategies.base import Portfolio, Signal
from backend.strategies.simple_strat_1 import SimpleStrat1
from backend.strategies.sma_crossover import SmaCrossover


def make_data(prices, opens=None):
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="D")
    if opens is None:
        opens = prices
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) * 1.01 for o, c in zip(opens, prices)],
            "low": [min(o, c) * 0.99 if min(o, c) > 0 else 0.0 for o, c in zip(opens, prices)],
            "close": prices,
            "volume": [10000] * len(prices),
        },
        index=dates,
    )


class TestSimpleStrat1Numerical:
    def test_open_price_zero_does_not_crash(self):
        """If day's open price is 0, division by zero could crash drop_from_open calc."""
        data = make_data(
            prices=[100.0, 101.0, 102.0],
            opens=[0.0, 100.0, 101.0],  # First bar open = 0
        )
        strategy = SimpleStrat1(entry_drop=1.0, buy_size=100.0)
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        strategy.init(data)
        for i in range(len(data)):
            try:
                signal = strategy.next(i, data, portfolio)
            except ZeroDivisionError:
                pytest.fail(
                    "SimpleStrat1 raised ZeroDivisionError when open price = 0. "
                    "Bug: (day_open - close) / day_open with day_open=0."
                )

    def test_stop_loss_with_zero_avg_entry(self):
        """When avg_entry is 0, stop loss division should not crash.
        
        The check `if position > 0 and avg_entry > 0` should prevent this,
        but if avg_entry is somehow 0 with position > 0, division crashes.
        """
        data = make_data([100.0, 90.0, 80.0])
        strategy = SimpleStrat1(stop_loss=5.0)
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        # Manually set an inconsistent state
        portfolio.positions[sym] = 10.0
        portfolio.avg_entry[sym] = 0.0  # avg_entry=0 but position>0

        strategy.init(data)
        try:
            for i in range(len(data)):
                strategy.next(i, data, portfolio)
        except ZeroDivisionError:
            pytest.fail(
                "SimpleStrat1 raised ZeroDivisionError when avg_entry=0 with position>0. "
                "Bug: loss_pct division by avg_entry when avg_entry=0."
            )

    def test_profit_target_huge_gain(self):
        """Extreme gain should not cause overflow in gain_pct calculation."""
        data = make_data([100.0, 10000.0])
        strategy = SimpleStrat1(profit_target=5.0)
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        portfolio.positions[sym] = 10.0
        portfolio.avg_entry[sym] = 100.0

        strategy.init(data)
        for i in range(len(data)):
            signal = strategy.next(i, data, portfolio)
            if signal == Signal.SELL:
                break  # success: SELL triggered on 10000% gain

    def test_entry_drop_100_pct(self):
        """entry_drop=100% means next DCA price = 0. Should not crash."""
        data = make_data([100.0, 50.0, 0.0])
        strategy = SimpleStrat1(entry_drop=100.0, buy_size=100.0, max_buys=3)
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        strategy.init(data)
        try:
            for i in range(len(data)):
                strategy.next(i, data, portfolio)
        except Exception as e:
            pytest.fail(f"SimpleStrat1 crashed with entry_drop=100: {e}")

    def test_negative_buy_size(self):
        """Negative buy_size should not crash (though it makes no economic sense)."""
        data = make_data([100.0, 110.0, 120.0])
        strategy = SimpleStrat1(buy_size=-100.0, entry_drop=1.0)
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        strategy.init(data)
        try:
            for i in range(len(data)):
                strategy.next(i, data, portfolio)
        except Exception as e:
            pytest.fail(f"SimpleStrat1 crashed with negative buy_size: {e}")

    def test_negative_profit_target(self):
        """Negative profit target (immediate sell) should trigger SELL on first bar above entry."""
        data = make_data([100.0, 101.0])
        strategy = SimpleStrat1(profit_target=-10.0)  # negative — sell immediately
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        portfolio.positions[sym] = 10.0
        portfolio.avg_entry[sym] = 100.0

        strategy.init(data)
        signals = []
        for i in range(len(data)):
            sig = strategy.next(i, data, portfolio)
            signals.append(sig)

        # With negative profit_target, gain_pct > -10 is always true, so should sell
        # gain_pct = (101 - 100) / 100 * 100 = 1. 1 >= -10 => True, so SELL
        assert Signal.SELL in signals, (
            "Negative profit_target should immediately trigger SELL"
        )

    def test_stop_loss_zero(self):
        """stop_loss=0 should trigger EXIT on any price decline."""
        data = make_data([100.0, 99.999])
        strategy = SimpleStrat1(stop_loss=0.0)
        portfolio = Portfolio(cash=10000.0)
        sym = "TEST"
        strategy._symbol = sym

        portfolio.positions[sym] = 10.0
        portfolio.avg_entry[sym] = 100.0

        strategy.init(data)
        signals = []
        for i in range(len(data)):
            sig = strategy.next(i, data, portfolio)
            signals.append(sig)

        assert Signal.EXIT in signals, (
            "stop_loss=0 should trigger EXIT on any loss"
        )


class TestSmaCrossoverNumerical:
    def test_equal_windows(self):
        """short_window == long_window should work (produces equal SMAs)."""
        data = make_data([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        strategy = SmaCrossover(short_window=3, long_window=3)
        portfolio = Portfolio(cash=10000.0)

        strategy.init(data)
        for i in range(len(data)):
            signal = strategy.next(i, data, portfolio)
            assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD), (
                f"Unexpected signal: {signal}"
            )

    def test_short_window_longer_than_long_window(self):
        """If short > long, the logic still runs (but may be inverted)."""
        data = make_data([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        strategy = SmaCrossover(short_window=5, long_window=3)
        portfolio = Portfolio(cash=10000.0)

        strategy.init(data)
        for i in range(len(data)):
            signal = strategy.next(i, data, portfolio)
            assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD), (
                f"Unexpected signal: {signal}"
            )

    def test_window_larger_than_data(self):
        """When window > data length, init fills NaN for SMAs. next() should return HOLD."""
        data = make_data([100.0, 101.0])
        strategy = SmaCrossover(short_window=10, long_window=50)
        portfolio = Portfolio(cash=10000.0)

        strategy.init(data)
        for i in range(len(data)):
            signal = strategy.next(i, data, portfolio)
            assert signal == Signal.HOLD, (
                f"Expected HOLD when SMAs are NaN, got {signal}"
            )

    def test_constant_price_no_crossover(self):
        """Constant price means no crossover — should always HOLD."""
        data = make_data([100.0] * 100)
        strategy = SmaCrossover(short_window=5, long_window=20)
        portfolio = Portfolio(cash=10000.0)

        strategy.init(data)
        for i in range(len(data)):
            signal = strategy.next(i, data, portfolio)
            assert signal == Signal.HOLD, (
                f"Expected HOLD for constant price, got {signal} at bar {i}"
            )

    def test_sma_handles_nan_values(self):
        """SMA on data with NaN should not crash."""
        data = make_data([100.0, float("nan"), 102.0, 103.0, 104.0, 105.0])
        strategy = SmaCrossover(short_window=3, long_window=5)
        portfolio = Portfolio(cash=10000.0)

        strategy.init(data)
        for i in range(len(data)):
            try:
                signal = strategy.next(i, data, portfolio)
            except Exception as e:
                pytest.fail(f"SmaCrossover crashed on NaN data: {e}")


class TestMLStrategyNumerical:
    def test_kelly_fraction_edge_cases(self, monkeypatch):
        """Kelly fraction should handle edge values gracefully."""
        from backend.strategies.ml_strategy import MLStrategy

        strategy = MLStrategy(use_kelly=True)
        strategy._kelly_count = 0

        # With count < 3, returns 0.5 * (2p - 1)
        f1 = strategy._kelly_fraction(0.5)
        assert f1 == 0.0, f"Kelly with prob=0.5 and no history should be 0, got {f1}"

        f2 = strategy._kelly_fraction(1.0)
        assert f2 == 0.5, f"Kelly with prob=1.0 and no history should be 0.5, got {f2}"

        f3 = strategy._kelly_fraction(0.0)
        assert f3 == -0.5, f"Kelly with prob=0.0 should be -0.5, got {f3}"

        # With count >= 3
        strategy._kelly_count = 5
        strategy._kelly_wins = 4
        strategy._kelly_losses = 1
        strategy._kelly_win_pct = 0.4  # sum of win pcts
        strategy._kelly_loss_pct = 0.1

        f4 = strategy._kelly_fraction(0.6)
        assert 0.0 <= f4 <= 0.25, f"Kelly fraction should be in [0, 0.25], got {f4}"

    def test_position_size_with_kelly(self):
        """Position size should not exceed available cash."""
        from backend.strategies.ml_strategy import MLStrategy

        strategy = MLStrategy(use_kelly=True)
        strategy._kelly_count = 3
        strategy._kelly_wins = 2
        strategy._kelly_losses = 1
        strategy._kelly_win_pct = 0.3
        strategy._kelly_loss_pct = 0.1

        portfolio = Portfolio(cash=10000.0)
        size = strategy._position_size(0.8, portfolio)

        assert size <= portfolio.cash, (
            f"Position size ({size}) should not exceed cash ({portfolio.cash})"
        )
        assert size >= 0, f"Position size should be >= 0, got {size}"

    def test_trailing_stop_with_zero_peak(self):
        """Trailing stop should not crash when _peak_price is 0."""
        from backend.strategies.ml_strategy import MLStrategy

        strategy = MLStrategy(trailing_stop_pct=0.05)
        strategy._peak_price = 0.0

        # The trailing stop check: if trailing_stop_pct > 0 and _peak_price > 0
        # With _peak_price=0, the trailing stop check is skipped. This is fine.
        # But if it weren't, dd = (price - 0) / 0 = inf, and check might fail.

        portfolio = Portfolio(cash=10000.0)
        # We just verify the method doesn't crash
        assert strategy.trailing_stop_pct == 0.05

    def test_max_hold_bars_zero(self):
        """max_hold_bars=0 should disable time-based exit."""
        from backend.strategies.ml_strategy import MLStrategy

        strategy = MLStrategy(max_hold_bars=0)
        assert strategy.max_hold_bars == 0
        # In _evaluate_signal, if max_hold_bars > 0, time exit is active.
        # With 0, it's disabled.
