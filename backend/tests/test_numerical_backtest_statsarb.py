"""Numerical edge cases for stats_arb/backtest.py"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.stats_arb.backtest import BacktestEngine, BacktestResult


class TestBacktestEngineReturnEdgeCases:
    def test_total_loss_100_percent(self):
        """100% loss should not crash annualised return calculation.
        
        Bug: When total_ret <= -100, (1 + total_ret/100) <= 0,
        so ((1 + total_ret/100) ** (1/years)) is invalid (negative base,
        fractional exponent).
        """
        equity = pd.Series([100000.0, 0.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()

        assert result.total_return_pct == pytest.approx(-100.0, abs=0.01), (
            f"Total return should be -100%, got {result.total_return_pct}"
        )
        assert np.isfinite(result.annualised_return_pct) or result.annualised_return_pct == -100.0, (
            f"Annualised return should be finite with -100% total return, "
            f"got {result.annualised_return_pct}. "
            "Bug: (1 + total_ret/100) ** (1/years) with negative base."
        )

    def test_total_loss_more_than_100_percent(self):
        """Losing more than 100% should still produce finite metrics."""
        equity = pd.Series([100000.0, -50000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()

        assert result.total_return_pct == pytest.approx(-150.0, abs=0.01), (
            f"Total return should be -150%, got {result.total_return_pct}"
        )
        assert np.isfinite(result.annualised_return_pct), (
            f"Annualised return should be finite, got {result.annualised_return_pct}"
        )

    def test_zero_initial_capital(self):
        """Zero initial capital should return safe default values."""
        equity = pd.Series([0.0, 100.0])
        engine = BacktestEngine(equity, [], initial_capital=0.0)
        try:
            result = engine.run()
            assert np.isfinite(result.total_return_pct) or result.total_return_pct == 0.0, (
                f"total_return_pct should be finite, got {result.total_return_pct}"
            )
        except ZeroDivisionError:
            pytest.fail("BacktestEngine crashed with zero initial_capital: division by zero")

    def test_single_bar_equity(self):
        """Single bar equity should return empty_result safely."""
        equity = pd.Series([100000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.total_return_pct == 0.0
        assert result.num_trades == 0

    def test_two_bar_equity_zero_return(self):
        """Two identical bars should give 0 return."""
        equity = pd.Series([100000.0, 100000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.total_return_pct == 0.0
        assert result.sharpe_ratio == 0.0

    def test_huge_return(self):
        """Extremely large return should not overflow."""
        equity = pd.Series([100.0, 1e15])
        engine = BacktestEngine(equity, [], initial_capital=100.0)
        result = engine.run()
        assert np.isfinite(result.total_return_pct), (
            f"Total return should be finite for huge gain, got {result.total_return_pct}"
        )
        assert np.isfinite(result.annualised_return_pct)


class TestBacktestEngineSharpeEdgeCases:
    def test_sharpe_zero_std_returns(self):
        """Zero standard deviation of returns should produce 0 Sharpe."""
        equity = pd.Series([100000.0, 100000.0, 100000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.sharpe_ratio == 0.0, (
            f"Sharpe should be 0 with constant returns, got {result.sharpe_ratio}"
        )

    def test_sharpe_single_return(self):
        """Single data point after pct_change gives nan, Sharpe should be 0."""
        equity = pd.Series([100000.0, 101000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        # With 2 bars, pct_change gives 1 non-nan return
        # If std > 0, Sharpe is computed
        if result.sharpe_ratio != 0.0:
            # With a 1% return in 1 day, annualised Sharpe = 0.01/0 * sqrt(252) = 0 if std=0
            # Actually, with a single return, std is 0 -> Sharpe=0
            assert result.sharpe_ratio == 0.0, (
                f"Sharpe with single return should be 0 (std=0), got {result.sharpe_ratio}"
            )


class TestBacktestEngineDrawdownEdgeCases:
    def test_drawdown_positive_cummax_zero(self):
        """Drawdown calculation should handle cummax = 0 gracefully."""
        equity = pd.Series([-100.0, -200.0])
        engine = BacktestEngine(equity, [], initial_capital=-100.0)
        result = engine.run()
        assert np.isfinite(result.max_drawdown_pct), (
            f"Drawdown should be finite with negative equity, got {result.max_drawdown_pct}"
        )

    def test_drawdown_single_bar(self):
        """Single bar should have 0 drawdown."""
        equity = pd.Series([100000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.max_drawdown_pct == 0.0


class TestBacktestEngineTradesEdgeCases:
    def test_win_rate_all_wins(self):
        """All winning trades should give 100% win rate."""
        equity = pd.Series([100000.0, 110000.0])

        class MockTrade:
            pnl = 100.0
            return_pct = 10.0
            holding_period = 5
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        trades = [MockTrade() for _ in range(5)]
        engine = BacktestEngine(equity, trades, initial_capital=100000.0)
        result = engine.run()
        assert result.win_rate_pct == 100.0, (
            f"Win rate should be 100% with all wins, got {result.win_rate_pct}"
        )

    def test_win_rate_all_losses(self):
        """All losing trades should give 0% win rate."""
        equity = pd.Series([100000.0, 90000.0])

        class MockTrade:
            pnl = -100.0
            return_pct = -10.0
            holding_period = 5
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        trades = [MockTrade() for _ in range(3)]
        engine = BacktestEngine(equity, trades, initial_capital=100000.0)
        result = engine.run()
        assert result.win_rate_pct == 0.0, (
            f"Win rate should be 0% with all losses, got {result.win_rate_pct}"
        )

    def test_avg_loss_empty_losses(self):
        """When there are no losses, avg_loss_pct should be 0."""
        equity = pd.Series([100000.0, 110000.0])

        class MockTrade:
            pnl = 100.0
            return_pct = 10.0
            holding_period = 5
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        trades = [MockTrade() for _ in range(3)]
        engine = BacktestEngine(equity, trades, initial_capital=100000.0)
        result = engine.run()
        assert result.avg_loss_pct == 0.0


class TestBacktestEngineProfitFactorEdgeCases:
    def test_profit_factor_infinity(self):
        """All positive PnL => profit_factor = inf."""
        equity = pd.Series([100000.0, 110000.0])

        class MockTrade:
            pnl = 100.0
            return_pct = 10.0
            holding_period = 5
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        trades = [MockTrade() for _ in range(2)]
        engine = BacktestEngine(equity, trades, initial_capital=100000.0)
        result = engine.run()
        assert result.profit_factor == 999999.0, (
                f"profit_factor should be 999999.0 with all positive PnL, got {result.profit_factor}"
            )

    def test_profit_factor_zero(self):
        """All negative PnL => profit_factor = 0."""
        equity = pd.Series([100000.0, 90000.0])

        class MockTrade:
            pnl = -100.0
            return_pct = -10.0
            holding_period = 5
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        trades = [MockTrade() for _ in range(2)]
        engine = BacktestEngine(equity, trades, initial_capital=100000.0)
        result = engine.run()
        assert result.profit_factor == 0.0, (
            f"profit_factor should be 0 with all negative PnL, got {result.profit_factor}"
        )


class TestBacktestEngineCalmarEdgeCases:
    def test_calmar_zero_drawdown(self):
        """Calmar with 0 max drawdown should be 0 (division guard)."""
        equity = pd.Series([100000.0, 100000.0, 100000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.calmar_ratio == 0.0, (
            f"Calmar should be 0 with 0 drawdown, got {result.calmar_ratio}"
        )

    def test_calmar_positive_return(self):
        """Calmar with positive return and finite drawdown should be positive."""
        equity = pd.Series([100000.0, 110000.0, 105000.0, 120000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        if result.annualised_return_pct > 0 and result.max_drawdown_pct < 0:
            assert result.calmar_ratio > 0, (
                f"Calmar should be positive, got {result.calmar_ratio}"
            )


class TestBacktestEngineMarketBeta:
    def test_beta_computation(self):
        """Beta to market should be computed correctly."""
        equity = pd.Series([100000.0, 101000.0, 102000.0, 103000.0])
        market = pd.Series([10000.0, 10100.0, 10200.0, 10300.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0, market_returns=market)
        result = engine.run()
        # Both have identical returns (1%), so beta should be ~1.0
        assert result.beta == pytest.approx(1.0, abs=0.1), (
            f"Beta should be ~1.0 for identical returns, got {result.beta}"
        )

    def test_alpha_zero_for_identical_returns(self):
        """Alpha should be ~0 with identical strategy and benchmark returns."""
        equity = pd.Series([100000.0, 101000.0, 102000.0, 103000.0])
        market = pd.Series([10000.0, 10100.0, 10200.0, 10300.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0, market_returns=market)
        result = engine.run()
        assert result.alpha == pytest.approx(0.0, abs=0.01), (
            f"Alpha should be ~0 for identical returns, got {result.alpha}"
        )

    def test_beta_with_flat_market(self):
        """When market returns are 0, beta should be 0."""
        equity = pd.Series([100000.0, 101000.0, 102000.0])
        market = pd.Series([10000.0, 10000.0, 10000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0, market_returns=market)
        result = engine.run()
        assert result.beta == 0.0, (
            f"Beta should be 0 with flat market, got {result.beta}"
        )


class TestBacktestEngineExposure:
    def test_exposure_with_no_trades(self):
        """Exposure should be 0% with no trades."""
        equity = pd.Series([100000.0, 101000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.exposure_pct == 0.0, (
            f"Exposure should be 0 with no trades, got {result.exposure_pct}"
        )

    def test_exposure_bounded_at_100(self):
        """Exposure should never exceed 100%."""
        equity = pd.Series([100000.0, 101000.0])

        class MockTrade:
            pnl = 100.0
            return_pct = 10.0
            holding_period = 9999  # more than equity length
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        trades = [MockTrade()]
        engine = BacktestEngine(equity, trades, initial_capital=100000.0)
        result = engine.run()
        assert result.exposure_pct <= 100.0, (
            f"Exposure should be capped at 100%, got {result.exposure_pct}"
        )


class TestBacktestEngineTurnover:
    def test_turnover_no_trades(self):
        """Turnover with no trades should be 0."""
        equity = pd.Series([100000.0, 101000.0])
        engine = BacktestEngine(equity, [], initial_capital=100000.0)
        result = engine.run()
        assert result.turnover == 0.0


class TestBacktestEngineEmptyResult:
    def test_empty_result_defaults(self):
        """Empty result should return sensible default values."""
        engine = BacktestEngine(pd.Series([100000.0]), [], initial_capital=100000.0)
        result = engine.run()
        assert result.total_return_pct == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown_pct == 0.0
        assert result.num_trades == 0
        assert result.final_equity == 100000.0

    def test_to_dict_rounding(self):
        """to_dict should round floats without errors, including inf."""
        class MockTrade:
            pnl = 100.0
            return_pct = 10.0
            holding_period = 5
            shares_a = 100.0
            shares_b = -100.0
            entry_price_a = 100.0
            entry_price_b = 50.0

        equity = pd.Series([100000.0, 110000.0])
        engine = BacktestEngine(equity, [MockTrade()], initial_capital=100000.0)
        result = engine.run()

        d = result.to_dict()
        for k, v in d.items():
            if isinstance(v, float):
                assert np.isfinite(v) or v == float("inf"), (
                    f"to_dict has non-finite value for {k}: {v}"
                )
