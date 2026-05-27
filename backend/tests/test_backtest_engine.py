import numpy as np
import pandas as pd
import pytest

from backend.backtest.engine import BacktestEngine, BacktestResult
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


class HoldStrategy(Strategy):
    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
        return Signal.HOLD


class BuyOnFirstSellOnLast(Strategy):
    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
        if i == 0:
            return Signal.BUY
        if i == len(data) - 1:
            return Signal.SELL
        return Signal.HOLD


class BuySellAlternating(Strategy):
    def __init__(self):
        self.trade_days = [1, 3, 5, 7]

    def init(self, data: pd.DataFrame) -> None:
        pass

    def next(self, i: int, data: pd.DataFrame, portfolio) -> str:
        if i in self.trade_days:
            pos = portfolio.positions.get("TEST", 0)
            return Signal.SELL if pos > 0 else Signal.BUY
        return Signal.HOLD


class TestBacktestEngineValidation:
    def test_raises_on_missing_columns(self):
        data = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2025-01-01", periods=1))
        strategy = HoldStrategy()
        with pytest.raises(ValueError, match="missing required columns"):
            BacktestEngine(data, strategy)

    def test_raises_on_partial_columns(self):
        data = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "close": [100.0]},
            index=pd.date_range("2025-01-01", periods=1),
        )
        strategy = HoldStrategy()
        with pytest.raises(ValueError, match="missing required columns"):
            BacktestEngine(data, strategy)


class TestBacktestEngineHold:
    def test_hold_strategy_returns_no_trades(self):
        data = make_data([100.0, 101.0, 102.0])
        result = BacktestEngine(data, HoldStrategy()).run()

        assert result.trades.empty
        assert result.initial_cash == 10000.0

    def test_hold_strategy_equity_is_flat(self):
        data = make_data([100.0] * 5)
        result = BacktestEngine(data, HoldStrategy()).run()

        assert len(result.equity_curve) == 5
        assert all(result.equity_curve["equity"] == 10000.0)


class TestBacktestEngineBuySell:
    def test_buy_then_sell_profits(self):
        data = make_data([100.0, 100.0, 110.0, 110.0])
        strategy = BuyOnFirstSellOnLast()
        result = BacktestEngine(data, strategy, symbol="TEST", initial_cash=10000.0).run()

        assert len(result.trades) == 2
        assert result.trades.iloc[0]["side"] == "buy"
        assert result.trades.iloc[1]["side"] == "sell"
        assert result.trades.iloc[0]["qty"] == 100  # 10000 // 100
        assert result.trades.iloc[1]["pnl"] == 1000.0  # 100 * (110 - 100)

    def test_buy_then_sell_loss(self):
        data = make_data([100.0, 100.0, 90.0, 90.0])
        strategy = BuyOnFirstSellOnLast()
        result = BacktestEngine(data, strategy, symbol="TEST", initial_cash=10000.0).run()

        assert result.trades.iloc[1]["pnl"] == -1000.0  # 100 * (90 - 100)

    def test_multiple_trade_cycles(self):
        data = make_data([100.0, 105.0, 100.0, 110.0, 100.0, 115.0, 100.0, 120.0])
        strategy = BuySellAlternating()
        result = BacktestEngine(data, strategy, symbol="TEST", initial_cash=10000.0).run()

        assert len(result.trades) == 4  # 2 buys + 2 sells
        sells = result.trades[result.trades["side"] == "sell"]
        assert len(sells) == 2
        assert all(sells["pnl"] > 0)  # bought low sold high each cycle

    def test_equity_curve_is_monotonic_length(self):
        data = make_data([100.0, 101.0, 102.0, 103.0, 104.0])
        strategy = BuyOnFirstSellOnLast()
        result = BacktestEngine(data, strategy, symbol="TEST").run()

        assert len(result.equity_curve) == 5
        assert result.equity_curve.iloc[0]["equity"] == 10000.0

    def test_initial_cash_preserved_in_result(self):
        data = make_data([100.0, 101.0])
        result = BacktestEngine(data, HoldStrategy(), initial_cash=50000.0).run()
        assert result.initial_cash == 50000.0
        assert result.metrics["final_equity"] == 50000.0


class TestBacktestEngineStream:
    def test_stream_yields_per_bar(self):
        data = make_data([100.0, 101.0, 102.0])
        engine = BacktestEngine(data, BuyOnFirstSellOnLast(), symbol="TEST")
        events = list(engine.stream())
        assert len(events) == 3

    def test_stream_events_have_expected_keys(self):
        data = make_data([100.0, 101.0])
        engine = BacktestEngine(data, HoldStrategy(), symbol="TEST")
        event = list(engine.stream())[0]
        assert "snapshot" in event
        assert "trade" in event
        assert "signal" in event
        assert "equity" in event["snapshot"]
        assert "cash" in event["snapshot"]
        assert "bar_index" in event["snapshot"]

    def test_stream_first_bar_buy_signal(self):
        data = make_data([100.0, 101.0])
        engine = BacktestEngine(data, BuyOnFirstSellOnLast(), symbol="TEST")
        events = list(engine.stream())
        assert events[0]["signal"] == Signal.BUY
        assert events[0]["trade"] is not None
        assert events[0]["trade"]["side"] == "buy"
