"""Tests verifying bugs in strategy implementations (C1-C3, C6, H6, H2, M8)."""

import pandas as pd
import pytest

from backend.strategies.simple_strat_1 import SimpleStrat1
from backend.strategies.base import Portfolio, Signal


def make_intraday_data(prices: list[float], days: int = 1) -> pd.DataFrame:
    """Create intraday data spanning *days* calendar days with given price sequence per day."""
    rows: list[pd.DataFrame] = []
    for d in range(days):
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=d)
        day_data = pd.DataFrame(
            {
                "open": [prices[0]] + prices[1:],
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [10000] * len(prices),
            },
            index=pd.date_range(date, periods=len(prices), freq="h"),
        )
        rows.append(day_data)
    return pd.concat(rows)


# ── C1 / C2: _bought_levels / _last_buy_avg mutated before engine confirms ──


class TestC1_StateMutatedBeforeTradeConfirmed:
    """C1: _bought_levels += 1 before engine executes BUY.
       C2: _last_buy_avg set to stale pre-trade average.
    """

    def test_bought_levels_not_incremented_when_no_cash(self):
        """C1: Strategy should NOT increment _bought_levels if cash is insufficient."""
        prices = [100.0, 99.0, 98.0, 97.0, 96.0]
        data = make_intraday_data(prices, days=1)
        strat = SimpleStrat1(buy_size=100.0, entry_drop=0.5, max_buys=3)
        strat._symbol = "TEST"
        strat.init(data)

        portfolio = Portfolio(cash=10.0)  # not enough for any $100 buy

        for i in range(len(data)):
            strat.next(i, data, portfolio)

        assert strat._bought_levels == 0

    def test_last_buy_avg_is_buy_price_not_stale_avg_entry(self):
        """C2: _last_buy_avg should be the buy price, not the pre-trade avg_entry."""
        prices = [200.0, 199.0, 198.0, 197.0, 196.0]
        data = make_intraday_data(prices, days=1)
        strat = SimpleStrat1(buy_size=100.0, entry_drop=0.5, max_buys=1)
        strat._symbol = "TEST"
        strat.init(data)

        portfolio = Portfolio(cash=10000.0)

        for i in range(len(data)):
            signal = strat.next(i, data, portfolio)
            if signal == Signal.BUY:
                close = float(data["close"].iloc[i])
                qty = strat.buy_size / close
                portfolio.cash -= qty * close
                old_shares = portfolio.positions.get("TEST", 0)
                old_avg = portfolio.avg_entry.get("TEST", 0.0)
                portfolio.positions["TEST"] = old_shares + qty
                if old_shares > 0 and old_avg > 0:
                    portfolio.avg_entry["TEST"] = (old_avg * old_shares + close * qty) / (old_shares + qty)
                else:
                    portfolio.avg_entry["TEST"] = close
                strat.on_trade("buy", "TEST", qty, close)

        # After first buy at 199, _last_buy_avg should be 199 (the close price)
        # Not the portfolio avg_entry (which starts at 0 then gets blended)
        assert strat._last_buy_avg == pytest.approx(199.0, abs=0.01)

    def test_last_buy_avg_not_set_to_stale_avg_entry_on_dca(self):
        """C2 on DCA buy: _last_buy_avg must not be set to portfolio avg_entry (stale)."""
        prices = [200.0, 190.0, 188.0, 170.0, 160.0]
        data = make_intraday_data(prices, days=1)
        strat = SimpleStrat1(buy_size=100.0, entry_drop=0.5, max_buys=2)
        strat._symbol = "TEST"
        strat.init(data)

        portfolio = Portfolio(cash=10000.0)

        # First buy happens at bar 1 (close=190, dropped from 200 open)
        # After first buy, portfolio avg_entry gets set to 190
        # At bar 2 (close=188), we simulate that avg_entry is 190,
        # so _last_buy_avg (190) * (1-0.5/100) = 189.05
        # close=188 <= 189.05 so DCA triggers (and stop loss doesn't since loss < 3%)
        # C2 bug would set _last_buy_avg = portfolio.avg_entry = ~189 (blended)
        # Fix should set _last_buy_avg = close = 188

        for i in range(len(data)):
            signal = strat.next(i, data, portfolio)
            if signal == Signal.BUY:
                close = float(data["close"].iloc[i])
                qty = strat.buy_size / close
                portfolio.cash -= qty * close
                old_shares = portfolio.positions.get("TEST", 0)
                old_avg = portfolio.avg_entry.get("TEST", 0.0)
                portfolio.positions["TEST"] = old_shares + qty
                if old_shares > 0 and old_avg > 0:
                    portfolio.avg_entry["TEST"] = (old_avg * old_shares + close * qty) / (old_shares + qty)
                else:
                    portfolio.avg_entry["TEST"] = close
                strat.on_trade("buy", "TEST", qty, close)

        # After second buy (DCA) at 188, _last_buy_avg should be 188, NOT the blended avg_entry
        assert strat._bought_levels == 2
        assert strat._last_buy_avg == pytest.approx(188.0, abs=0.01)

    def test_bought_levels_respected_when_cash_insufficient(self):
        """C1: max_buys cap should not be consumed when buys are skipped due to cash."""
        prices = [100.0, 99.0, 98.0]
        data = make_intraday_data(prices, days=1)
        strat = SimpleStrat1(buy_size=100.0, entry_drop=0.5, max_buys=1)
        strat._symbol = "TEST"
        strat.init(data)

        portfolio = Portfolio(cash=50.0)  # not enough for a $100 buy

        for i in range(len(data)):
            strat.next(i, data, portfolio)

        # No buy should have been counted because cash is insufficient
        # Bug: _bought_levels gets incremented before cash check
        assert strat._bought_levels == 0


# ── C3: MLStrategy consecutive BUY signals reset _entry_bar / _peak_price ──


class TestC3_MLStrategyConsecutiveBuyResets:
    """C3: Consecutive ML BUY signals while holding reset safety rails."""

    def _make_ml_strat_with_mock_model(self):
        """Create an MLStrategy with a mock model that always predicts buy."""
        from unittest.mock import MagicMock

        strat = SimpleStrat1  # placeholder, we actually test via mock
        return None

    def test_entry_bar_not_overwritten_on_consecutive_buys_when_holding(self):
        """C3: _entry_bar should NOT be reset on BUY signal if already holding."""
        from backend.strategies.ml_strategy import MLStrategy
        import numpy as np
        from unittest.mock import MagicMock

        data = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000] * 5,
        }, index=pd.date_range("2025-01-01", periods=5, freq="D"))

        strat = MLStrategy(model_name="nonexistent", max_hold_bars=3)

        # Manually set up so init doesn't fail
        strat._model_loaded = True
        strat._features_df = data.copy()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        strat.model = mock_model
        strat.feature_columns = data.columns.tolist()

        portfolio = Portfolio(cash=10000.0)

        # Bar 0: first BUY -> _entry_bar = 0
        # Bar 1: second BUY while holding -> _entry_bar should stay 0 (not reset to 1)
        sig0 = strat.next(0, data, portfolio)
        entry_bar_after_first = strat._entry_bar

        sig1 = strat.next(1, data, portfolio)
        entry_bar_after_second = strat._entry_bar

        assert entry_bar_after_first == 0
        assert entry_bar_after_second == 0, (
            "_entry_bar should not be reset on consecutive BUY while holding"
        )

    def test_peak_price_not_overwritten_on_consecutive_buys_when_holding(self):
        """C3: _peak_price should NOT be reset on BUY signal if already holding."""
        from backend.strategies.ml_strategy import MLStrategy
        import numpy as np
        from unittest.mock import MagicMock

        data = pd.DataFrame({
            "open": [100, 105, 104, 106, 108],
            "high": [101, 108, 106, 108, 110],
            "low": [99, 103, 102, 104, 106],
            "close": [100, 105, 104, 106, 108],
            "volume": [1000] * 5,
        }, index=pd.date_range("2025-01-01", periods=5, freq="D"))

        strat = MLStrategy(model_name="nonexistent", max_hold_bars=5)
        strat._model_loaded = True
        strat._features_df = data.copy()

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        strat.model = mock_model
        strat.feature_columns = data.columns.tolist()

        portfolio = Portfolio(cash=10000.0)

        strat.next(0, data, portfolio)  # buy at 100, _peak_price = 100
        peak_after_first = strat._peak_price

        strat._peak_price = 110  # simulate price running up
        sig = strat.next(1, data, portfolio)  # another BUY signal (close=105, not trailing stop)
        peak_after_second = strat._peak_price

        assert peak_after_second == 110, (
            "_peak_price should track actual peak, not be reset to entry price on re-buy"
        )


# ── H6: _last_buy_avg reset to 0 on day change kills DCA ──


class TestH6_LastBuyAvgResetOnDayChangeKillsDCA:
    """H6: _last_buy_avg reset to 0 on day change while holding position."""

    def test_dca_fires_across_day_change_when_position_held(self):
        """H6: Day change should not reset _last_buy_avg to 0 when holding position."""
        prices_day1 = [200.0, 195.0, 190.0, 185.0, 180.0]
        prices_day2 = [188.0, 185.0]

        day1 = pd.DataFrame({
            "open": [200.0] + prices_day1[1:],
            "high": [p * 1.01 for p in prices_day1],
            "low": [p * 0.99 for p in prices_day1],
            "close": prices_day1,
            "volume": [10000] * len(prices_day1),
        }, index=pd.date_range("2025-01-01", periods=len(prices_day1), freq="h"))

        day2 = pd.DataFrame({
            "open": [190.0, 186.0],
            "high": [192.0, 188.0],
            "low": [187.0, 184.0],
            "close": prices_day2,
            "volume": [10000] * len(prices_day2),
        }, index=pd.date_range("2025-01-02", periods=len(prices_day2), freq="h"))

        data = pd.concat([day1, day2])
        strat = SimpleStrat1(buy_size=100.0, entry_drop=0.5, max_buys=2, stop_loss=50.0)
        strat._symbol = "TEST"
        strat.init(data)

        portfolio = Portfolio(cash=10000.0)

        # Process all bars (simulate engine's on_trade after buy)
        for i in range(len(data)):
            signal = strat.next(i, data, portfolio)
            if signal == Signal.BUY:
                close = float(data["close"].iloc[i])
                qty = strat.buy_size / close
                portfolio.cash -= qty * close
                old_shares = portfolio.positions.get("TEST", 0)
                old_avg = portfolio.avg_entry.get("TEST", 0.0)
                portfolio.positions["TEST"] = old_shares + qty
                if old_shares > 0 and old_avg > 0:
                    portfolio.avg_entry["TEST"] = (old_avg * old_shares + close * qty) / (old_shares + qty)
                else:
                    portfolio.avg_entry["TEST"] = close
                strat.on_trade("buy", "TEST", qty, close)

        # After day change, if holding position, _last_buy_avg should NOT be 0
        # If _last_buy_avg == 0.0, next_dca_price = 0.0, DCA never fires (close <= 0 is false)
        # We expect at least one DCA buy across the day boundary
        assert strat._bought_levels >= 1


# ── H2: registry.py crashes on None param values ──


class TestH2_RegistryNoneParamValues:
    """H2: registry.py crashes when params contain None for int/float fields."""

    def test_int_none_does_not_crash(self):
        """H2: Passing None for an int param should not raise TypeError."""
        from backend.strategies.registry import get_strategy

        strat = get_strategy("Simple Strat 1", {"max_buys": None})
        assert strat is not None

    def test_float_none_does_not_crash(self):
        """H2: Passing None for a float param should not raise TypeError."""
        from backend.strategies.registry import get_strategy

        strat = get_strategy("Simple Strat 1", {"buy_size": None})
        assert strat is not None

    def test_str_none_does_not_crash(self):
        """H2: Passing None for a str param should not raise TypeError."""
        from backend.strategies.registry import get_strategy

        strat = get_strategy("ML Strategy", {"model_name": None})
        assert strat is not None

    def test_bool_none_does_not_crash(self):
        """H2: Passing None for a bool param should not raise."""
        from backend.strategies.registry import get_strategy

        strat = get_strategy("ML Strategy", {"use_kelly": None})
        assert strat is not None


# ── M8: Model metadata silently overrides user params ──


class TestM8_MetadataOverridesUserParams:
    """M8: Model metadata should not silently override user-passed params."""

    def test_user_confidence_threshold_not_overridden_by_metadata(self):
        from backend.strategies.ml_strategy import MLStrategy
        from unittest.mock import MagicMock

        strat = MLStrategy(confidence_threshold=0.70)

        mock_metadata = MagicMock()
        mock_metadata.params = {"confidence_threshold": 0.55}

        strat._apply_metadata_params(mock_metadata)

        assert strat.confidence_threshold == 0.70, (
            "User's confidence_threshold should not be overridden by metadata"
        )

    def test_use_kelly_not_overridden_when_user_explicit(self):
        from backend.strategies.ml_strategy import MLStrategy
        from unittest.mock import MagicMock

        strat = MLStrategy(use_kelly=True)
        mock_metadata = MagicMock()
        mock_metadata.params = {"use_kelly": False}

        strat._apply_metadata_params(mock_metadata)

        assert strat.use_kelly is True, (
            "User's use_kelly should not be overridden by metadata"
        )


# ── M18: parseInt("") returns NaN in strategy params (frontend) ──
# Tested on the backend side via registry

class TestM18_ParseIntEmptyReturnsNaN:
    """M18: Empty string for int params should not produce NaN."""

    def test_empty_string_for_int_param_falls_back_to_default(self):
        from backend.strategies.registry import get_strategy

        strat = get_strategy("Simple Strat 1", {"max_buys": ""})
        # Should use default (1) or handle gracefully
        assert strat.max_buys == 1
