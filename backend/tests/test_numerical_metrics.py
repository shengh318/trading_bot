"""Numerical edge cases for backtest/metrics.py"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.backtest.metrics import calculate_metrics


class TestMetricsTotalReturnPct:
    def test_large_initial_cash_vs_small_return(self):
        """total_return_pct should be sensible for huge initial cash."""
        equity = pd.DataFrame({"equity": [1e9, 1e9 + 1], "cash": [1e9, 1e9 + 1]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=1e9)
        assert metrics["total_return_pct"] == pytest.approx(1e-7, abs=1e-6), (
            f"Expected ~1e-7%, got {metrics['total_return_pct']}"
        )

    def test_negative_initial_cash(self):
        """If initial_cash is negative, total_return_pct should still be handled."""
        equity = pd.DataFrame({"equity": [-500.0, -300.0], "cash": [-500.0, -300.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=-500.0)
        assert np.isfinite(metrics["total_return_pct"]), (
            f"total_return_pct should be finite with negative cash, got {metrics['total_return_pct']}"
        )

    def test_very_small_positive_initial_cash(self):
        """With initial_cash=1e-8, a $1 gain should not overflow."""
        equity = pd.DataFrame({"equity": [1e-8, 1.00000001], "cash": [1e-8, 1.00000001]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=1e-8)
        assert np.isfinite(metrics["total_return_pct"]), (
            f"total_return_pct should be finite for tiny initial cash, got {metrics['total_return_pct']}"
        )


class TestMetricsSharpeRatio:
    def test_sharpe_with_all_positive_returns(self):
        """Sharpe should be positive when all equity changes are positive."""
        equity = pd.DataFrame({"equity": [100.0, 101.0, 102.0, 103.0, 104.0], "cash": [100.0] * 5})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["sharpe_ratio"] > 0, (
            f"Sharpe should be positive with consistent gains, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_with_all_negative_returns(self):
        """Sharpe should be negative when equity consistently declines."""
        equity = pd.DataFrame({"equity": [100.0, 99.0, 98.0, 97.0, 96.0], "cash": [100.0] * 5})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["sharpe_ratio"] < 0, (
            f"Sharpe should be negative with consistent losses, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_with_constant_equity(self):
        """Sharpe should be 0 when equity never changes."""
        equity = pd.DataFrame({"equity": [100.0, 100.0, 100.0, 100.0, 100.0], "cash": [100.0] * 5})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["sharpe_ratio"] == 0.0, (
            f"Sharpe should be 0 with constant equity, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_single_bar(self):
        """With only 1 bar of equity data, Sharpe should be 0 (can't compute)."""
        equity = pd.DataFrame({"equity": [100.0], "cash": [100.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["sharpe_ratio"] == 0.0, (
            f"Sharpe should be 0 with 1 bar, got {metrics['sharpe_ratio']}"
        )

    def test_sharpe_two_bars_constant(self):
        """With 2 bars and no change, Sharpe should be 0."""
        equity = pd.DataFrame({"equity": [100.0, 100.0], "cash": [100.0] * 2})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["sharpe_ratio"] == 0.0, (
            f"Sharpe should be 0 with constant 2-bar equity, got {metrics['sharpe_ratio']}"
        )

    def test_excess_return_calculation_with_risk_free_rate(self):
        """With risk_free_rate > 0, excess return formula should be sensible.
        
        Bug suspicion: risk_free_rate / annual_factor uses sqrt(252) instead of 252.
        """
        equity = pd.DataFrame({"equity": [100.0, 101.0, 102.0, 103.0, 104.0], "cash": [100.0] * 5})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics_high_rf = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=0.5)
        metrics_low_rf = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=0.01)

        assert metrics_high_rf["sharpe_ratio"] <= metrics_low_rf["sharpe_ratio"], (
            "Sharpe with higher risk-free rate should be <= Sharpe with lower risk-free rate. "
            "Bug: risk_free_rate is divided by sqrt(252) instead of 252, giving wrong excess returns."
        )

    def test_sharpe_with_inf_returns(self):
        """If returns contain inf values, Sharpe should not crash."""
        equity = pd.DataFrame({"equity": [100.0, float("inf"), 200.0], "cash": [100.0, 100.0, 200.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        try:
            metrics = calculate_metrics(equity, trades, initial_cash=100.0)
            assert np.isfinite(metrics["sharpe_ratio"]) or metrics["sharpe_ratio"] == 0.0, (
                f"Sharpe should be finite or 0 with inf returns, got {metrics['sharpe_ratio']}"
            )
        except Exception as e:
            pytest.fail(f"calculate_metrics crashed on inf returns: {e}")

    def test_sharpe_with_nan_returns(self):
        """If returns contain NaN values, Sharpe should not crash."""
        equity = pd.DataFrame({"equity": [100.0, float("nan"), 200.0], "cash": [100.0, 100.0, 200.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        try:
            metrics = calculate_metrics(equity, trades, initial_cash=100.0)
            assert np.isfinite(metrics["sharpe_ratio"]) or metrics["sharpe_ratio"] == 0.0, (
                f"Sharpe should be finite or 0 with nan returns, got {metrics['sharpe_ratio']}"
            )
        except Exception as e:
            pytest.fail(f"calculate_metrics crashed on nan returns: {e}")


class TestMetricsDrawdown:
    def test_max_drawdown_negative(self):
        """max_drawdown_pct should be 0 or negative, never positive."""
        equity = pd.DataFrame({
            "equity": [100.0, 120.0, 90.0, 110.0],
            "cash": [100.0, 120.0, 90.0, 110.0],
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["max_drawdown_pct"] <= 0, (
            f"max_drawdown_pct should be <= 0, got {metrics['max_drawdown_pct']}"
        )

    def test_max_drawdown_single_bar(self):
        """Single bar drawdown should be 0."""
        equity = pd.DataFrame({"equity": [100.0], "cash": [100.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["max_drawdown_pct"] == 0.0, (
            f"max_drawdown_pct should be 0 with 1 bar, got {metrics['max_drawdown_pct']}"
        )

    def test_max_drawdown_monotonic_up(self):
        """Monotonically increasing equity should have 0 drawdown."""
        equity = pd.DataFrame({
            "equity": [100.0, 110.0, 125.0, 130.0],
            "cash": [100.0, 110.0, 125.0, 130.0],
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["max_drawdown_pct"] == 0.0, (
            f"Monotonic up equity should have 0 drawdown, got {metrics['max_drawdown_pct']}"
        )

    def test_max_drawdown_50_percent(self):
        """50% peak-to-trough drop should yield -50% drawdown."""
        equity = pd.DataFrame({
            "equity": [100.0, 200.0, 100.0],
            "cash": [100.0, 200.0, 100.0],
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["max_drawdown_pct"] == pytest.approx(-50.0, abs=0.01), (
            f"Expected -50% drawdown, got {metrics['max_drawdown_pct']}"
        )

    def test_drawdown_with_zero_equity(self):
        """If equity drops to 0, drawdown should be -100%."""
        equity = pd.DataFrame({
            "equity": [100.0, 0.0],
            "cash": [100.0, 0.0],
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["max_drawdown_pct"] == pytest.approx(-100.0, abs=0.01), (
            f"Expected -100% drawdown at zero equity, got {metrics['max_drawdown_pct']}"
        )

    def test_drawdown_with_negative_equity(self):
        """If equity goes negative, drawdown should still be computed."""
        equity = pd.DataFrame({
            "equity": [100.0, -50.0, 200.0],
            "cash": [100.0, -50.0, 200.0],
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["max_drawdown_pct"] <= 0, (
            f"Drawdown should be <= 0 with negative equity, got {metrics['max_drawdown_pct']}"
        )


class TestMetricsProfitFactor:
    def test_profit_factor_inf_all_wins(self):
        """If all trades have positive PnL (no losses), profit_factor should be inf."""
        equity = pd.DataFrame({"equity": [100.0, 200.0], "cash": [100.0, 200.0]})
        trades = pd.DataFrame({
            "bar_index": [0], "timestamp": ["2025-01-01"],
            "symbol": ["T"], "side": ["sell"],
            "qty": [10], "price": [100.0], "pnl": [50.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["profit_factor"] == 999999.0, (
                f"profit_factor should be 999999.0 sentinel with all wins, got {metrics['profit_factor']}"
            )

    def test_profit_factor_zero_gross_profit(self):
        """If all trades have negative PnL, profit_factor should be 0."""
        equity = pd.DataFrame({"equity": [100.0, 50.0], "cash": [100.0, 50.0]})
        trades = pd.DataFrame({
            "bar_index": [0], "timestamp": ["2025-01-01"],
            "symbol": ["T"], "side": ["sell"],
            "qty": [10], "price": [50.0], "pnl": [-50.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["profit_factor"] == 0.0, (
            f"profit_factor should be 0 with all losses, got {metrics['profit_factor']}"
        )

    def test_profit_factor_zero_pnl_trades(self):
        """Trades with PnL=0 should not count as wins or losses for profit_factor."""
        equity = pd.DataFrame({"equity": [100.0, 100.0], "cash": [100.0, 100.0]})
        trades = pd.DataFrame({
            "bar_index": [0], "timestamp": ["2025-01-01"],
            "symbol": ["T"], "side": ["sell"],
            "qty": [10], "price": [100.0], "pnl": [0.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["profit_factor"] == 0.0, (
            f"profit_factor should be 0 with zero-PnL trades, got {metrics['profit_factor']}"
        )


class TestMetricsEmptyEdgeCases:
    def test_empty_equity_curve(self):
        """Empty equity curve should return safe defaults."""
        equity = pd.DataFrame({"equity": [], "cash": []})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        assert metrics["total_return_pct"] == 0.0
        assert metrics["final_equity"] == 10000.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown_pct"] == 0.0
        assert metrics["num_trades"] == 0

    def test_no_trades_dataframe(self):
        """No trades should yield 0 for trade-related metrics."""
        equity = pd.DataFrame({"equity": [100.0, 110.0], "cash": [100.0, 110.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=100.0)
        assert metrics["num_trades"] == 0
        assert metrics["win_rate_pct"] == 0.0
        assert metrics["profit_factor"] == 0.0

    def test_equity_with_zero_column_name(self):
        """If equity column somehow has unexpected name, should still handle gracefully."""
        equity_bad = pd.DataFrame({"bad_col": [100.0, 110.0], "cash": [100.0, 110.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        try:
            metrics = calculate_metrics(equity_bad, trades, initial_cash=100.0)
        except (KeyError, Exception) as e:
            pytest.fail(f"calculate_metrics crashed with bad column name: {e}")


class TestMetricsAnnualizationBug:
    def test_annual_factor_scaling(self):
        """The annual_factor uses sqrt(252 * n / max(n,1)) = sqrt(252).
        
        But risk_free_rate / annual_factor divides by sqrt(252) instead of 252.
        For a daily RF rate, this is wrong: daily_rf = annual_rf / 252, not annual_rf / sqrt(252).
        """
        n_bars = 10
        equity = pd.DataFrame({
            "equity": [100.0 + i for i in range(n_bars + 1)],
            "cash": [100.0] * (n_bars + 1),
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])

        metrics_0rf = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=0.0)
        metrics_high_rf = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=0.05)

        assert metrics_high_rf["sharpe_ratio"] < metrics_0rf["sharpe_ratio"] or (
            metrics_high_rf["sharpe_ratio"] == 0.0 and metrics_0rf["sharpe_ratio"] == 0.0
        ), (
            "Higher risk-free rate should reduce Sharpe. "
            "If this fails, the excess return formula (risk_free_rate / sqrt(252)) may be wrong."
        )
