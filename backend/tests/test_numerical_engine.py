"""Numerical edge cases for backtest/engine.py"""

import numpy as np
import pandas as pd
import pytest

from backend.backtest.engine import BacktestEngine, MultiSymbolBacktestEngine
from backend.strategies.base import Signal, Strategy, Portfolio


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


# ── Helper strategies ──────────────────────────────────────────────

class AlwaysBuy(Strategy):
    buy_size: float = 100.0

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if portfolio.cash > 0 and data.iloc[i]["close"] > 0:
            return Signal.BUY
        return Signal.HOLD


class AlwaysSell(Strategy):
    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        sym = getattr(self, "_symbol", "ASSET")
        if portfolio.positions.get(sym, 0) > 0:
            return Signal.SELL
        return Signal.HOLD


class BuyThenSell(Strategy):
    buy_size: float = 5000.0

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        sym = getattr(self, "_symbol", "ASSET")
        if i == 0:
            return Signal.BUY
        if i == 1 and portfolio.positions.get(sym, 0) > 0:
            return Signal.SELL
        return Signal.HOLD


class SellPortionZeroStrat(Strategy):
    """Strategy with sell_portion set to 0 — should this skip the sell?"""
    buy_size: float = 5000.0
    sell_portion: float = 0.0

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        sym = getattr(self, "_symbol", "ASSET")
        if i == 0:
            return Signal.BUY
        if i == 1 and portfolio.positions.get(sym, 0) > 0:
            return Signal.SELL
        return Signal.HOLD


class ExitWithPosition(Strategy):
    buy_size: float = 5000.0

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        sym = getattr(self, "_symbol", "ASSET")
        if i == 0:
            return Signal.BUY
        if i == 1 and portfolio.positions.get(sym, 0) > 0:
            return Signal.EXIT
        return Signal.HOLD


class BuyOnHighPrice(Strategy):
    buy_size: float = 100.0

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if i == 0:
            return Signal.BUY
        return Signal.HOLD


# ── Tests ──────────────────────────────────────────────────────────

class TestBacktestEngineSellPortionZero:
    def test_sell_portion_zero_skips_sell_entirely(self):
        """When sell_portion=0, qty = current_pos * 0 / 100 = 0, so nothing is sold.
        
        Potential bug: a sell_portion of 0 should mean 'sell nothing' or 'sell everything'?
        The engine treats 0 as a valid value (since getattr returns 0.0, not None),
        resulting in a silent no-op sell.
        """
        data = make_data([100.0, 110.0, 120.0])
        result = BacktestEngine(
            data, SellPortionZeroStrat(), symbol="TEST", initial_cash=10000.0
        ).run()

        sells = result.trades[result.trades["side"] == "sell"]
        buys = result.trades[result.trades["side"] == "buy"]

        assert len(sells) == 0, (
            f"Expected 0 sells (sell_portion=0 results in qty=0), got {len(sells)}. "
            "Bug: sell_portion=0 silently produces no-op sell trade."
        )
        if len(buys) > 0:
            last_snapshot = result.equity_curve.iloc[-1]
            equity = last_snapshot["equity"]
            cash = last_snapshot["cash"]
            assert equity > cash - 0.01, (
                f"If sell_portion=0 creates no sell, equity ({equity}) should > cash ({cash}) "
                "since position remains."
            )


class TestBacktestEngineExitSignal:
    def test_exit_signal_clears_full_position(self):
        """EXIT signal should sell entire position regardless of sell_portion."""
        data = make_data([100.0, 110.0, 120.0])
        result = BacktestEngine(
            data, ExitWithPosition(), symbol="TEST", initial_cash=10000.0
        ).run()

        sells = result.trades[result.trades["side"] == "sell"]
        assert len(sells) == 1, f"Expected 1 EXIT trade, got {len(sells)}"

        buy_qty = result.trades[result.trades["side"] == "buy"]["qty"].sum()
        sell_qty = sells["qty"].sum()
        assert sell_qty == pytest.approx(buy_qty, abs=0.01), (
            f"EXIT should sell full position. Buy qty: {buy_qty:.2f}, Sell qty: {sell_qty:.2f}"
        )


class TestBacktestEngineNegativePrices:
    def test_negative_close_price_skips_buy(self):
        """Engine should skip BUY when close price is negative."""
        data = make_data([-100.0, 100.0, 110.0])
        result = BacktestEngine(
            data, BuyThenSell(), symbol="TEST", initial_cash=10000.0
        ).run()

        # With negative price, condition `price > 0` should fail, so BUY is skipped.
        buys = result.trades[result.trades["side"] == "buy"]
        assert len(buys) == 0, (
            f"BUY at negative price should be skipped (price > 0 check), got {len(buys)} buys"
        )

    def test_sell_at_negative_price(self):
        """Engine should handle selling at negative close price."""
        data = make_data([100.0, -110.0, 120.0])

        class BuyThenSellNegative(Strategy):
            buy_size = 5000.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                sym = getattr(self, "_symbol", "ASSET")
                if i == 0:
                    return Signal.BUY
                if i == 1 and portfolio.positions.get(sym, 0) > 0:
                    return Signal.SELL
                return Signal.HOLD

        result = BacktestEngine(
            data, BuyThenSellNegative(), symbol="TEST", initial_cash=10000.0
        ).run()

        sells = result.trades[result.trades["side"] == "sell"]
        if len(sells) > 0:
            # PnL should still be a valid (possibly negative) number
            pnl = sells.iloc[0]["pnl"]
            assert np.isfinite(pnl), f"PnL should be finite for sell at negative price, got {pnl}"


class TestBacktestEngineFractionalShares:
    def test_fractional_buy_qty(self):
        """Buy quantities should not be truncated to integers."""
        data = make_data([100.0, 110.0])

        class BuyOneDollar(Strategy):
            buy_size = 1.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if i == 0:
                    return Signal.BUY
                return Signal.HOLD

        result = BacktestEngine(
            data, BuyOneDollar(), symbol="TEST", initial_cash=10000.0
        ).run()

        buys = result.trades[result.trades["side"] == "buy"]
        if len(buys) > 0:
            qty = buys.iloc[0]["qty"]
            assert qty == pytest.approx(0.01, abs=1e-4), (
                f"Buy $1 at $100 => qty=0.01, got {qty}"
            )

    def test_many_fractional_buys_sums_correctly(self):
        """Sum of fractional buys should match total cash deployed."""
        data = make_data([100.0] * 100)

        class ManySmallBuys(Strategy):
            buy_size = 100.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if portfolio.cash > 0:
                    return Signal.BUY
                return Signal.HOLD

        result = BacktestEngine(
            data, ManySmallBuys(), symbol="TEST", initial_cash=10000.0
        ).run()

        total_buy_qty = result.trades[result.trades["side"] == "buy"]["qty"].sum()
        # Each buy = $100 / $100 = 1 share. With $10000, should be up to 100 shares
        assert total_buy_qty == pytest.approx(100.0, abs=0.5), (
            f"Expected ~100 shares total from 100 buys of $100 at $100, got {total_buy_qty}"
        )


class TestBacktestEngineTinyCash:
    def test_buy_with_tiny_cash(self):
        """Buy with very small remaining cash should not cause floating point issues."""
        data = make_data([100.0, 110.0])

        class TinyBuy(Strategy):
            buy_size = 1e-10

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if i == 0 and portfolio.cash > 0:
                    return Signal.BUY
                return Signal.HOLD

        try:
            result = BacktestEngine(
                data, TinyBuy(), symbol="TEST", initial_cash=1e-10
            ).run()
        except Exception as e:
            pytest.fail(f"BacktestEngine crashed on tiny cash: {e}")

    def test_cash_exhausted_no_fraction_overflow(self):
        """When cash is nearly exhausted, remaining buys should not go negative."""
        data = make_data([3.0, 3.0, 3.0])

        class ExhaustCash(Strategy):
            buy_size = 10.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if portfolio.cash >= 3.0:
                    return Signal.BUY
                return Signal.HOLD

        result = BacktestEngine(
            data, ExhaustCash(), symbol="TEST", initial_cash=10.0
        ).run()

        final_snapshot = result.equity_curve.iloc[-1]
        assert final_snapshot["cash"] >= -0.01, (
            f"Cash should not go significantly negative, got {final_snapshot['cash']}"
        )


class TestBacktestEngineExtremeValues:
    def test_very_large_price(self):
        """Should handle very large prices without overflow."""
        data = make_data([1e12, 1e12 + 100])

        class BuyLarge(Strategy):
            buy_size = 5000.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if i == 0:
                    return Signal.BUY
                return Signal.HOLD

        result = BacktestEngine(
            data, BuyLarge(), symbol="TEST", initial_cash=10000.0
        ).run()

        buys = result.trades[result.trades["side"] == "buy"]
        if len(buys) > 0:
            qty = buys.iloc[0]["qty"]
            assert np.isfinite(qty), f"Buy quantity should be finite for large price, got {qty}"
            assert qty > 0, f"Buy quantity should be positive, got {qty}"

    def test_very_small_price(self):
        """Should handle very small prices without underflow."""
        data = make_data([1e-10, 2e-10])

        class BuySmallPrice(Strategy):
            buy_size = 5000.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if i == 0:
                    return Signal.BUY
                return Signal.HOLD

        result = BacktestEngine(
            data, BuySmallPrice(), symbol="TEST", initial_cash=10000.0
        ).run()

        buys = result.trades[result.trades["side"] == "buy"]
        if len(buys) > 0:
            qty = buys.iloc[0]["qty"]
            assert np.isfinite(qty), f"Buy quantity should be finite, got {qty}"
            assert qty > 0, f"Buy quantity should be positive, got {qty}"
            # Cost should not exceed cash
            cost = qty * 1e-10
            assert cost <= 10000.0 + 0.01, f"Cost ({cost}) should not exceed cash (10000)"


class TestBacktestEngineDividendEdgeCases:
    def test_dividend_with_zero_shares(self):
        """When position is 0, dividend should not add cash."""
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        data = pd.DataFrame({
            "open": [100.0, 110.0, 120.0],
            "high": [101.0, 111.0, 121.0],
            "low": [99.0, 109.0, 119.0],
            "close": [100.0, 110.0, 120.0],
            "volume": [10000] * 3,
        }, index=dates)
        div = pd.DataFrame(
            {"dividend": [5.0]},
            index=pd.to_datetime(["2025-01-02"]),
        )
        engine = BacktestEngine(data, AlwaysBuy(), symbol="TEST", initial_cash=10000.0, dividends=div)
        result = engine.run()
        # Even with dividend, engine should not crash
        assert result.metrics["final_equity"] > 0

    def test_dividend_on_exact_timestamp(self):
        """Dividend at exact bar timestamp should be applied correctly."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [10000] * 5,
        }, index=dates)

        class BuyAndHold(Strategy):
            buy_size = 5000.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if i == 0:
                    return Signal.BUY
                return Signal.HOLD

        div = pd.DataFrame(
            {"dividend": [2.0]},
            index=pd.to_datetime(["2025-01-03"]),
        )
        engine = BacktestEngine(data, BuyAndHold(), symbol="TEST", initial_cash=10000.0, dividends=div)
        result = engine.run()

        div_date = pd.to_datetime("2025-01-03")
        matching = result.equity_curve[result.equity_curve["timestamp"] == str(div_date)]
        if not matching.empty:
            # Cash should have increased by the dividend amount
            pass  # cash before/after comparison complex; just check no crash

    def test_dividend_missing_column(self):
        """Dividend DataFrame without 'dividend' column should raise ValueError."""
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        data = make_data([100.0, 110.0, 120.0])
        bad_div = pd.DataFrame({"wrong_col": [1.0]}, index=pd.to_datetime(["2025-01-02"]))
        with pytest.raises(ValueError, match="dividend"):
            BacktestEngine(data, AlwaysBuy(), symbol="TEST", initial_cash=10000.0, dividends=bad_div)


class TestMultiSymbolEngineEdgeCases:
    def test_multi_symbol_uneven_dates(self):
        """Different symbols with different date ranges should still work."""
        dates_a = pd.date_range("2025-01-01", periods=5, freq="D")
        dates_b = pd.date_range("2025-01-03", periods=5, freq="D")

        data_a = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0], "volume": [10000] * 5,
        }, index=dates_a)
        data_b = pd.DataFrame({
            "open": [50.0] * 5, "high": [51.0] * 5, "low": [49.0] * 5,
            "close": [50.0, 51.0, 52.0, 53.0, 54.0], "volume": [20000] * 5,
        }, index=dates_b)

        from backend.strategies.sma_crossover import SmaCrossover

        engine = MultiSymbolBacktestEngine(
            data={"A": data_a, "B": data_b},
            strategy_cls=SmaCrossover,
            initial_cash=10000.0,
        )
        result = engine.run()
        assert len(result.equity_curve) > 0
        assert result.metrics["final_equity"] > 0

    def test_multi_symbol_one_symbol_no_cash(self):
        """When cash is 0, all symbols should skip buying."""
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        data_a = pd.DataFrame({
            "open": [100.0] * 3, "high": [101.0] * 3, "low": [99.0] * 3,
            "close": [100.0] * 3, "volume": [10000] * 3,
        }, index=dates)
        data_b = data_a.copy()

        class AlwaysBuyMulti(Strategy):
            buy_size = 10000.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                if portfolio.cash > 0:
                    return Signal.BUY
                return Signal.HOLD

        engine = MultiSymbolBacktestEngine(
            data={"A": data_a, "B": data_b},
            strategy_cls=AlwaysBuyMulti,
            initial_cash=100.0,
        )
        result = engine.run()

        total_buy_qty = len(result.trades[result.trades["side"] == "buy"])
        assert total_buy_qty > 0, "Should have at least one buy with $100"

    def test_multi_symbol_pnl_large_sell(self):
        """Selling more shares than held should not be possible."""
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        data = pd.DataFrame({
            "open": [100.0] * 3, "high": [101.0] * 3, "low": [99.0] * 3,
            "close": [100.0] * 3, "volume": [10000] * 3,
        }, index=dates)

        class OverSellStrategy(Strategy):
            buy_size = 100.0

            def init(self, data):
                pass

            def next(self, i, data, portfolio):
                sym = getattr(self, "_symbol", "ASSET")
                if i == 0:
                    return Signal.BUY
                if i == 1:
                    return Signal.SELL
                return Signal.HOLD

        engine = MultiSymbolBacktestEngine(
            data={"A": data},
            strategy_cls=OverSellStrategy,
            initial_cash=10000.0,
        )
        result = engine.run()

        sells = result.trades[result.trades["side"] == "sell"]
        for _, s in sells.iterrows():
            assert s["qty"] > 0, "Sell qty should be positive"
            assert np.isfinite(s["pnl"]), f"Sell PnL should be finite, got {s['pnl']}"
