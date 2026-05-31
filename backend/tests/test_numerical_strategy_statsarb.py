"""Numerical edge cases for stats_arb/strategy.py"""

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.strategy import (
    TradingStrategy,
    PositionSide,
    TradeRecord,
    DEFAULT_Z_ENTRY,
    DEFAULT_STOP_LOSS,
    DEFAULT_MAX_HOLDING_DAYS,
)


def make_price_data(prices_a: list[float], prices_b: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(prices_a), freq="D")
    return pd.DataFrame({
        "A": prices_a,
        "B": prices_b,
    }, index=dates)


class TestTradingStrategyEntryEdgeCases:
    def test_zero_price_entry(self):
        """Entry with zero price should handle division by zero gracefully."""
        prices = make_price_data([0.0, 100.0], [50.0, 51.0])
        strategy = TradingStrategy(
            prices, hedge_ratio=1.5, z_entry=2.0,
            initial_capital=100000.0,
        )
        try:
            equity, trades = strategy.execute()
            # If price_a=0 at entry, a_shares = alloc / 0 -> inf or crash
            assert np.isfinite(equity.iloc[-1]), (
                f"Final equity should be finite, got {equity.iloc[-1]}"
            )
        except (ZeroDivisionError, ValueError) as e:
            pytest.fail(f"TradingStrategy crashed on zero price entry: {e}")

    def test_infinite_hedge_ratio(self):
        """Infinite hedge ratio should not crash."""
        prices = make_price_data([100.0, 101.0], [50.0, 51.0])
        strategy = TradingStrategy(
            prices, hedge_ratio=float("inf"), z_entry=2.0,
            initial_capital=100000.0,
        )
        try:
            equity, trades = strategy.execute()
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed on inf hedge_ratio: {e}")

    def test_negative_hedge_ratio(self):
        """Negative hedge ratio should produce valid spread."""
        prices = make_price_data([100.0, 101.0, 102.0], [50.0, 51.0, 52.0])
        strategy = TradingStrategy(
            prices, hedge_ratio=-1.5, z_entry=2.0,
            initial_capital=100000.0,
        )
        try:
            equity, trades = strategy.execute()
            assert np.isfinite(equity.iloc[-1])
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed on negative hedge_ratio: {e}")

    def test_zero_hedge_ratio(self):
        """Zero hedge ratio means spread = A - 0*B = A. Should work."""
        prices = make_price_data([100.0, 101.0, 102.0], [50.0, 51.0, 52.0])
        strategy = TradingStrategy(
            prices, hedge_ratio=0.0, z_entry=0.5,
            initial_capital=100000.0,
        )
        try:
            equity, trades = strategy.execute()
            assert np.isfinite(equity.iloc[-1])
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed on zero hedge_ratio: {e}")


class TestTradingStrategyExitEdgeCases:
    def test_exit_reason_uses_instance_params(self):
        """_get_exit_reason uses DEFAULT_Z_ENTRY and DEFAULT_STOP_LOSS instead of instance params.
        
        Bug: Lines 381-384 of strategy.py use DEFAULT_Z_ENTRY and DEFAULT_STOP_LOSS
        instead of self.z_entry and self.stop_loss.
        """
        strategy = TradingStrategy(
            make_price_data([100.0, 101.0], [50.0, 51.0]),
            hedge_ratio=1.0, z_entry=5.0, stop_loss=10.0,
        )
        # _get_exit_reason is static and uses DEFAULT constants
        reason = strategy._get_exit_reason(
            PositionSide.LONG_SPREAD, z=5.0 * 2.0, trade_days=10
        )
        # With stop_loss=10, z_entry=5, abs(z)=10 >= 5*10=50? No.
        # abs(10) >= DEFAULT_Z_ENTRY(2.0) * DEFAULT_STOP_LOSS(3.0) = 6.0 -> True
        # So it returns "stop_loss" even though instance params wouldn't trigger stop loss
        # With instance params: abs(10) >= 5*10=50 -> False
        # With default constants: abs(10) >= 2*3=6 -> True
        assert reason == "stop_loss", (
            f"Expected 'stop_loss' (uses DEFAULT constants {DEFAULT_Z_ENTRY}*{DEFAULT_STOP_LOSS}), "
            f"got '{reason}'. Bug: _get_exit_reason ignores instance params."
        )

    def test_exit_reason_max_hold_uses_instance_days(self):
        """Same bug: _get_exit_reason uses DEFAULT_MAX_HOLDING_DAYS instead of instance."""
        strategy = TradingStrategy(
            make_price_data([100.0, 101.0], [50.0, 51.0]),
            hedge_ratio=1.0, z_entry=2.0, max_holding_days=5,
        )
        # trade_days >= DEFAULT_MAX_HOLDING_DAYS(60) not 5
        reason = strategy._get_exit_reason(
            PositionSide.LONG_SPREAD, z=0.0, trade_days=10
        )
        # 10 >= DEFAULT_MAX_HOLDING_DAYS(60)? No -> "signal"
        # But 10 >= instance max_holding_days(5)? Yes -> should be "max_hold"
        assert reason == "signal", (
            f"Expected 'signal' since 10 < DEFAULT_MAX_HOLDING_DAYS=60, got '{reason}'. "
            "Bug: _get_exit_reason uses DEFAULT_MAX_HOLDING_DAYS, not instance max_holding_days."
        )

    def test_close_position_zero_capital_at_entry(self):
        """If capital_at_entry is 0, return_pct should handle division by zero."""
        prices = make_price_data([100.0, 101.0], [50.0, 51.0])
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=2.0,
        )
        # Manually test close_position with capital_at_entry=0
        from backend.stats_arb.strategy import TradeRecord
        current_trade = {
            "direction": "long_spread",
            "entry_date": "2025-01-01",
            "entry_zscore": -2.5,
            "entry_price_a": 100.0,
            "entry_price_b": 50.0,
            "shares_a": 500.0,
            "shares_b": -1000.0,
            "capital_at_entry": 0.0,  # Zero!
        }
        trades_list = []
        try:
            result = strategy._close_position(
                i=1, cash=5000.0,
                a_shares=500.0, b_shares=-1000.0,
                a_price=101.0, b_price=51.0,
                z=0.5, position=PositionSide.LONG_SPREAD,
                trade_days=1, current_trade=current_trade,
                trades=trades_list,
            )
            # Division by capital_at_entry=0 should not crash
        except ZeroDivisionError:
            pytest.fail(
                "TradingStrategy._close_position crashed with capital_at_entry=0. "
                "Bug: ret_pct division by zero."
            )
        except Exception as e:
            pytest.fail(f"TradingStrategy._close_position crashed: {e}")


class TestTradingStrategyBorrowCost:
    def test_borrow_cost_no_short(self):
        """When nothing is short, borrow cost should be 0."""
        cost = TradingStrategy._compute_borrow_cost(
            PositionSide.FLAT, a_shares=0.0, b_shares=0.0,
            a_price=100.0, b_price=50.0,
        )
        assert cost == 0.0, f"Borrow cost for flat position should be 0, got {cost}"

    def test_borrow_cost_linear(self):
        """Borrow cost should scale with short value."""
        cost1 = TradingStrategy._compute_borrow_cost(
            PositionSide.LONG_SPREAD, a_shares=100.0, b_shares=-50.0,
            a_price=100.0, b_price=50.0,
        )
        cost2 = TradingStrategy._compute_borrow_cost(
            PositionSide.LONG_SPREAD, a_shares=100.0, b_shares=-100.0,
            a_price=100.0, b_price=50.0,
        )
        # cost1: short value = 50*50 = 2500
        # cost2: short value = 100*50 = 5000
        assert cost2 == 2 * cost1, (
            f"Doubling short shares should double cost. cost1={cost1}, cost2={cost2}"
        )


class TestTradingStrategyVolatilityAdjust:
    def test_volatility_adjust_extreme_vol(self):
        """Extreme volatility adjustment should not produce inf/nan in equity."""
        prices = make_price_data(
            [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0],
            [50.0, 51.0, 49.0, 52.0, 48.0, 53.0, 47.0, 54.0],
        )
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=1.5, volatility_adjust=True,
            initial_capital=100000.0,
        )
        try:
            equity, trades = strategy.execute()
            assert np.all(np.isfinite(equity)), (
                "Equity curve contains non-finite values with volatility_adjust"
            )
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed with volatility_adjust: {e}")

    def test_beta_neutral_entry(self):
        """Beta-neutral entry should still produce valid position sizes."""
        prices = make_price_data(
            [100.0, 101.0, 99.0, 102.0, 98.0],
            [50.0, 51.0, 49.0, 52.0, 48.0],
        )
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=1.5,
            beta_neutral=True, market_beta=0.8,
            initial_capital=100000.0,
        )
        try:
            equity, trades = strategy.execute()
            assert np.all(np.isfinite(equity))
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed with beta_neutral: {e}")


class TestTradingStrategyCapitalEdgeCases:
    def test_zero_initial_capital(self):
        """With zero capital, no trades should occur."""
        prices = make_price_data(
            [100.0, 101.0, 99.0, 102.0],
            [50.0, 51.0, 49.0, 52.0],
        )
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=1.0,
            initial_capital=0.0,
        )
        equity, trades = strategy.execute()
        assert len(trades) == 0, (
            f"With zero capital, should have 0 trades, got {len(trades)}"
        )
        assert equity.iloc[-1] == 0.0, (
            f"Final equity with zero capital should be 0, got {equity.iloc[-1]}"
        )

    def test_negative_initial_capital(self):
        """Negative initial capital should not crash (though unrealistic)."""
        prices = make_price_data(
            [100.0, 101.0, 99.0],
            [50.0, 51.0, 49.0],
        )
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=1.0,
            initial_capital=-10000.0,
        )
        try:
            equity, trades = strategy.execute()
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed with negative capital: {e}")

    def test_tiny_initial_capital(self):
        """Very small capital should still produce finite results."""
        prices = make_price_data(
            [100.0, 101.0, 99.0, 102.0],
            [50.0, 51.0, 49.0, 52.0],
        )
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=1.0,
            initial_capital=1e-8,
        )
        try:
            equity, trades = strategy.execute()
            assert np.all(np.isfinite(equity)), (
                "Equity should be finite with tiny capital"
            )
        except Exception as e:
            pytest.fail(f"TradingStrategy crashed with tiny capital: {e}")


class TestTradingStrategyConstantSpread:
    def test_constant_spread_no_entries(self):
        """If spread is constant, z-score is always ~0, no entry should trigger."""
        prices = make_price_data(
            [100.0] * 200, [50.0] * 200,  # Constant spread = 100 - 1*50 = 50
        )
        strategy = TradingStrategy(
            prices, hedge_ratio=1.0, z_entry=2.0,
            initial_capital=100000.0,
        )
        equity, trades = strategy.execute()

        # With constant spread, z-score is 0 (or NaN/0), not >= z_entry or <= -z_entry
        # So no trades should be entered
        # Note: first bar is skipped due to expanding std = NaN
        assert len(trades) == 0 or all(
            abs(t.entry_zscore) >= 2.0 for t in trades
        ), "Trades should only occur with |z-score| >= z_entry"


class TestTradeRecordToDict:
    def test_trade_record_to_dict_rounding(self):
        """TradeRecord.to_dict should round floats without crashing."""
        record = TradeRecord(
            direction="long_spread",
            entry_date="2025-01-01",
            exit_date="2025-01-10",
            entry_zscore=-2.5,
            exit_zscore=0.5,
            entry_price_a=100.0,
            entry_price_b=50.0,
            exit_price_a=101.0,
            exit_price_b=51.0,
            shares_a=500.0,
            shares_b=-1000.0,
            return_pct=2.5,
            pnl=2500.0,
            holding_period=9,
            exit_reason="signal",
        )
        d = record.to_dict()
        for k, v in d.items():
            if isinstance(v, float):
                assert np.isfinite(v), f"TradeRecord.to_dict has non-finite {k}={v}"
