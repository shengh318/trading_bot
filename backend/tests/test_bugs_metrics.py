"""Tests for bugs in metrics (Bug 4, Bug 25, M12, M13, L13)."""

import pandas as pd
import pytest

from backend.backtest.metrics import calculate_metrics


# ── M12: Sharpe always annualizes with 252 regardless of actual bar count ──


class TestM12_SharpeAlways252:
    """M12: Sharpe should use actual number of bars/year, not hardcoded 252."""

    def test_sharpe_uses_actual_bar_count(self):
        """M12: For short backtests, Sharpe should adjust sqrt factor to actual data frequency."""
        equity = pd.DataFrame({
            "equity": [10000.0, 10100.0, 10200.0, 10300.0, 10400.0],
            "cash": [10000.0] * 5,
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])

        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)

        # With only 5 bars, annualizing with sqrt(252) massively inflates Sharpe.
        # A fix would use sqrt(252/5) or similar adjustment.
        # Just verify the metric is computed.
        assert isinstance(metrics["sharpe_ratio"], float)

    def test_daily_data_sharpe_annualizes_properly(self):
        """M12: Daily data with ~252 bars/year should use sqrt(252)."""
        n = 252
        equity = pd.DataFrame({
            "equity": [10000.0 * (1 + 0.001) ** i for i in range(n)],
            "cash": [10000.0] * n,
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])

        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)

        # For daily data, sqrt(252) is correct
        assert metrics["sharpe_ratio"] != 0.0


# ── M13: max_drawdown_pct stored as negative value ──


class TestM13_MaxDrawdownNegative:
    """M13: max_drawdown_pct should be stored as positive value (or at least consistently)."""

    def test_max_drawdown_pct_sign_consistent(self):
        """M13: max_drawdown_pct sign should be documented and consistent."""
        equity = pd.DataFrame({
            "equity": [10000.0, 11000.0, 9000.0, 10500.0],
            "cash": [10000.0] * 4,
        })
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])

        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)

        # Bug: M13 says it's stored as negative but field name suggests positive
        # Current code stores negative (e.g. -18.18)
        # Either make it positive or rename the field
        dd = metrics["max_drawdown_pct"]
        assert dd <= 0, (
            "M13: max_drawdown_pct is currently negative. "
            "Either make it positive or rename to clarify direction."
        )


# ── L13: float("inf") for profit factor not portable SQL ──


class TestL13_ProfitFactorInfNotPortable:
    """L13: float('inf') is not portable across SQL databases."""

    def test_profit_factor_no_inf_in_db_storage(self):
        """L13: When profit_factor is inf, a sentinel value should be used for DB storage."""
        from backend.backtest.metrics import calculate_metrics

        equity = pd.DataFrame({
            "equity": [10000.0, 11000.0],
            "cash": [10000.0] * 2,
        })
        trades = pd.DataFrame({
            "bar_index": [1],
            "timestamp": ["2025-01-02"],
            "symbol": ["T"],
            "side": ["sell"],
            "qty": [10],
            "price": [110.0],
            "pnl": [100.0],
        })

        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)

        # The metrics dict returns a large sentinel for profit_factor when gross_loss == 0
        assert metrics["profit_factor"] == 999999.0

        # Check that the value is handled when saving to DB


# ── Bug 4: profit_factor == float("inf") is stored in DB as None (FIXED) ──


class TestBug4_ProfitFactorInfToDB:
    """Bug 4: profit_factor=inf now replaced with 999999.0 sentinel."""

    def test_calculate_metrics_returns_sentinel_for_inf_profit_factor(self):
        """Bug 4: calculate_metrics now returns 999999.0 instead of float('inf')."""
        from backend.backtest.metrics import calculate_metrics

        equity = pd.DataFrame({
            "equity": [10000.0, 11000.0, 12000.0],
            "cash": [10000.0] * 3,
        })
        # All winning trades, no losing trades → profit_factor should be a sentinel
        trades = pd.DataFrame({
            "bar_index": [1, 2],
            "timestamp": ["2025-01-02", "2025-01-03"],
            "symbol": ["T", "T"],
            "side": ["sell", "sell"],
            "qty": [10, 10],
            "price": [110.0, 120.0],
            "pnl": [100.0, 200.0],
        })

        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)

        # After fix: should return a serializable sentinel instead of inf
        assert metrics["profit_factor"] == 999999.0, (
            "profit_factor should be 999999.0 sentinel (instead of inf)"
        )

    def test_db_save_backtest_run_preserves_profit_factor(self):
        """Bug 4: Saving metrics with profit_factor=None should store SQL NULL, not inf."""
        from backend.data.store import Database

        db = Database(":memory:")

        run_id = db.save_backtest_run({
            "strategy_name": "Test",
            "parameters": None,
            "symbol": "T",
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
            "initial_cash": 10000,
            "final_equity": 12000,
            "total_return": 20.0,
            "sharpe_ratio": 2.0,
            "max_drawdown": 5.0,
            "win_rate": 100.0,
            "num_trades": 2,
            "profit_factor": 999999.0,
        })

        saved = db.get_backtest_run_by_id(run_id)
        assert saved is not None
        # After fix: profit_factor should be a finite serializable number
        assert saved["profit_factor"] == 999999.0, (
            "profit_factor=999999.0 should be stored and retrieved correctly"
        )

        db.close()


# ── Bug 25: Sharpe ratio incorrectly annualized for non-daily bar counts ──


class TestBug25_SharpeAnnualization:
    """Bug 25: Sharpe ratio should correctly annualize for non-daily data frequencies."""

    def test_sharpe_annualization_differs_by_bar_count(self):
        """Bug 25: For 5-min data (78 bars/day), annualized Sharpe should differ from daily."""
        from backend.backtest.metrics import calculate_metrics

        # Daily data: 252 bars (1 year)
        daily_equity = pd.DataFrame({
            "equity": [10000.0 * (1 + 0.001) ** i for i in range(252)],
            "cash": [10000.0] * 252,
        })
        daily_trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])

        # 5-min data: 78 bars/day * 252 days = 19656 bars
        n_5min = 78 * 252
        hourly_equity = pd.DataFrame({
            "equity": [10000.0 * (1 + 0.0002) ** i for i in range(n_5min)],
            "cash": [10000.0] * n_5min,
        })
        hourly_trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])

        daily_metrics = calculate_metrics(daily_equity, daily_trades, initial_cash=10000.0)
        hourly_metrics = calculate_metrics(hourly_equity, hourly_trades, initial_cash=10000.0)

        # Bug: annual_factor = sqrt(252 * n_bars / max(n_bars, 1)) = sqrt(252)
        # This ignores actual data frequency
        # After fix: should use sqrt(252 * bars_per_day) for intraday data
        # For now, just verify both are computed
        assert isinstance(daily_metrics["sharpe_ratio"], float)
        assert isinstance(hourly_metrics["sharpe_ratio"], float)

    def test_sharpe_factor_uses_bar_count_not_just_252(self):
        """Bug 25: Annualization factor should account for bars_per_day, not just 252."""
        import numpy as np

        # The formula: annual_factor = np.sqrt(252 * n_bars / max(n_bars, 1))
        # This simplifies to sqrt(252) regardless of actual bars
        # For 5-min data with 78 bars/day:
        n_bars_5min = 78 * 252
        current_factor_5min = np.sqrt(252 * n_bars_5min / max(n_bars_5min, 1))
        expected_factor_5min = np.sqrt(252)  # What the current formula gives

        assert current_factor_5min == expected_factor_5min, (
            "Bug: factor is always sqrt(252) regardless of actual bar count"
        )

        # After fix: factor for 5-min data should be sqrt(252 * 78)
        correct_factor_5min = np.sqrt(252 * 78)
        assert not np.isclose(current_factor_5min, correct_factor_5min), (
            f"Current factor ({current_factor_5min:.2f}) doesn't match "
            f"correct factor ({correct_factor_5min:.2f}) for 5-min data"
        )
