"""Math audit tests verifying critical financial mathematics.

Validates that all financial formulae are scientifically correct:
- Sharpe ratio: proper risk-free rate conversion (annual → daily)
- Sortino ratio: correct downside deviation (not std of negatives)
- Hurst exponent: multi-lag R/S regression (not single-n ratio)
- Profit factor: handles infinite/gross_loss=0 without crash
- Structural break exit: no UnboundLocalError on ret_pct
"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.backtest.metrics import calculate_metrics
from backend.ml.features import _hurst_exponent
from backend.stats_arb.strategy import TradingStrategy, PositionSide


class TestSharpeRatioRiskFreeConversion:
    """BUG 1: Verifies risk-free rate is divided by 252 (trading days),
    not sqrt(252), when converting to daily rate."""

    def test_sharpe_rfr_daily_conversion_exact(self):
        """With known daily returns, compute expected Sharpe by hand.

        The function rounds to 4 decimal places, so compare with abs=0.0001.
        """
        np.random.seed(42)
        daily_rets = np.random.randn(252) * 0.01 + 0.0005
        equity = 100.0 * np.cumprod(1 + daily_rets)
        equity = np.concatenate([[100.0], equity])
        df = pd.DataFrame({"equity": equity, "cash": equity})
        trades = pd.DataFrame(
            columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"]
        )

        annual_rfr = 0.05
        metrics = calculate_metrics(df, trades, initial_cash=100.0, risk_free_rate=annual_rfr)

        daily_returns = df["equity"].pct_change().fillna(0).values
        daily_rfr = annual_rfr / 252
        excess = daily_returns - daily_rfr
        expected_sharpe = (excess.mean() / excess.std()) * math.sqrt(252)

        assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=0.0001), (
            f"Sharpe {metrics['sharpe_ratio']} != expected {expected_sharpe}. "
            "The risk-free rate conversion (annual/252) is likely wrong."
        )

    def test_sharpe_rfr_lower_than_no_rfr(self):
        """Sharpe with positive risk-free rate must be strictly lower
        than Sharpe with zero risk-free rate (same equity curve)."""
        equity = pd.DataFrame({
            "equity": [100.0, 101.0, 102.0, 103.0, 104.0],
            "cash": [100.0] * 5,
        })
        trades = pd.DataFrame(
            columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"]
        )
        m0 = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=0.0)
        m1 = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=0.05)
        assert m1["sharpe_ratio"] < m0["sharpe_ratio"], (
            "Sharpe with 5% risk-free rate should be strictly lower than with 0%."
        )

    def test_sharpe_rfr_does_not_divide_by_sqrt252(self):
        """Verify that using the old (buggy) sqrt(252) divisor would give
        a materially different (wrong) result."""
        equity = pd.DataFrame({
            "equity": [100.0, 100.5, 101.0, 100.8, 101.2],
            "cash": [100.0] * 5,
        })
        trades = pd.DataFrame(
            columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"]
        )
        annual_rfr = 0.05
        metrics = calculate_metrics(equity, trades, initial_cash=100.0, risk_free_rate=annual_rfr)

        daily_returns = equity["equity"].pct_change().fillna(0).values
        annual_factor = math.sqrt(252)

        # Old buggy formula: rfr / sqrt(252)
        buggy_rfr_daily = annual_rfr / annual_factor
        buggy_excess = daily_returns - buggy_rfr_daily
        buggy_sharpe = (buggy_excess.mean() / buggy_excess.std()) * annual_factor

        # The correct value should differ from the buggy one
        assert metrics["sharpe_ratio"] != pytest.approx(buggy_sharpe, abs=1e-10), (
            "Sharpe should NOT match the buggy rfr/sqrt(252) formula."
        )


class TestSortinoDownsideDeviation:
    """BUG 2: Verifies Sortino uses sqrt(mean(min(0, r)^2)), not std(negative returns)."""

    def _sortino_manual(self, returns: np.ndarray, annual_factor: int = 252) -> float:
        """Correct Sortino: sqrt(mean(min(0, r)^2)) denominator."""
        mean_ret = float(returns.mean())
        downside = returns.copy()
        downside[downside > 0] = 0.0
        d_dev = float(np.sqrt(np.mean(downside ** 2)))
        if d_dev == 0:
            return 0.0
        return mean_ret / d_dev * math.sqrt(annual_factor)

    def _sortino_wrong_std_of_negatives(self, returns: np.ndarray, annual_factor: int = 252) -> float:
        """OLD buggy Sortino: std() of only negative returns."""
        neg = returns[returns < 0]
        if len(neg) < 2 or neg.std() == 0:
            return 0.0
        return float(returns.mean() / neg.std() * math.sqrt(annual_factor))

    def test_sortino_all_negative_equal(self):
        """All returns equally negative: std of negatives = 0 (all same),
        but proper downside deviation > 0 since values != 0."""
        returns = pd.Series(np.full(100, -0.01))
        from backend.stats_arb.backtest import BacktestEngine as StatsArbBacktestEngine

        engine = StatsArbBacktestEngine(
            equity_curve=pd.Series(np.cumprod(1 + returns.values)),
            trades=[],
            initial_capital=100.0,
        )

        # Manual correct value
        correct = self._sortino_manual(returns.values)
        correct_result = engine._compute_sortino(returns)

        assert correct_result == pytest.approx(correct, abs=1e-10), (
            f"Sortino {correct_result} != correct {correct}. "
            "Check that downside deviation = sqrt(mean(min(0,r)^2)), not std(negative returns)."
        )

    def test_sortino_all_positive(self):
        """All returns positive → downside deviation = 0 → Sortino = 0."""
        returns = pd.Series(np.full(100, 0.01))
        from backend.stats_arb.backtest import BacktestEngine as StatsArbBacktestEngine

        engine = StatsArbBacktestEngine(
            equity_curve=pd.Series(np.cumprod(1 + returns.values)),
            trades=[],
            initial_capital=100.0,
        )
        result = engine._compute_sortino(returns)
        assert result == 0.0, f"Sortino should be 0 when all returns positive, got {result}"

    def test_sortino_differs_from_wrong_std_method(self):
        """For asymmetric downside, the correct formula diverges from std-of-negatives."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.02)
        from backend.stats_arb.backtest import BacktestEngine as StatsArbBacktestEngine

        engine = StatsArbBacktestEngine(
            equity_curve=pd.Series(np.cumprod(1 + returns.values)),
            trades=[],
            initial_capital=100.0,
        )
        correct = engine._compute_sortino(returns)
        wrong = self._sortino_wrong_std_of_negatives(returns.values)

        assert correct != pytest.approx(wrong, abs=1e-6), (
            f"Correct Sortino {correct} should differ from buggy std-of-negatives {wrong}. "
            "The fix changed the formula."
        )

    def test_sortino_single_negative(self):
        """Single negative return should not crash."""
        returns = pd.Series([0.01] * 10 + [-0.02] + [0.01] * 10)
        from backend.stats_arb.backtest import BacktestEngine as StatsArbBacktestEngine

        engine = StatsArbBacktestEngine(
            equity_curve=pd.Series(np.cumprod(1 + returns.values)),
            trades=[],
            initial_capital=100.0,
        )
        result = engine._compute_sortino(returns)
        assert np.isfinite(result) or result == 0.0, (
            f"Sortino should be finite, got {result}"
        )


class TestHurstExponentMLFeatures:
    """BUG 3: Verifies ML features Hurst exponent uses multi-lag R/S regression,
    not the incorrect single-n ratio."""

    def test_hurst_white_noise_near_05(self):
        """White noise (stationary) should give Hurst ≈ 0.5 via multi-lag R/S.

        The R/S method is designed for stationary processes; white noise
        (IID returns) is the canonical stationary series with H ≈ 0.5.
        """
        np.random.seed(42)
        wn = np.random.randn(1000)
        h = _hurst_exponent(wn)
        assert 0.3 <= h <= 0.7, (
            f"White noise Hurst should be ~0.5, got {h}"
        )

    def test_hurst_trending_above_05(self):
        """Trending series (Brownian motion with drift) → H > 0.5."""
        np.random.seed(42)
        trending = np.cumsum(np.random.randn(500) * 0.5 + 0.01)
        h = _hurst_exponent(trending)
        assert h > 0.5, f"Trending series Hurst should be > 0.5, got {h}"

    def test_hurst_mean_reverting_below_05(self):
        """Mean-reverting (anti-persistent) series → H < 0.5."""
        np.random.seed(42)
        mr = np.random.randn(500) * 0.5
        for i in range(1, len(mr)):
            mr[i] = mr[i] - 0.3 * mr[i - 1]
        h = _hurst_exponent(mr)
        assert h < 0.5, f"Mean-reverting series Hurst should be < 0.5, got {h}"

    def test_hurst_constant_series(self):
        """Constant series → H = 0.5 (undefined / random walk default)."""
        h = _hurst_exponent(np.ones(100))
        assert h == 0.5, f"Constant series Hurst should be 0.5, got {h}"

    def test_hurst_short_series(self):
        """Series < 20 elements → H = 0.5."""
        h = _hurst_exponent(np.random.randn(10))
        assert h == 0.5, f"Short series Hurst should be 0.5, got {h}"

    def test_hurst_bounded_01(self):
        """Hurst should always be clamped to [0, 1]."""
        np.random.seed(42)
        for _ in range(20):
            ts = np.random.randn(100) * (np.random.rand() * 10)
            h = _hurst_exponent(ts)
            assert 0.0 <= h <= 1.0, f"Hurst {h} outside [0, 1]"

    def test_hurst_consistent_with_spread_module(self):
        """New features.py _hurst_exponent matches stats_arb/spread.py version."""
        from backend.stats_arb.spread import SpreadAnalyzer
        np.random.seed(42)
        ts = np.cumsum(np.random.randn(200))
        h_features = _hurst_exponent(ts)
        h_spread = SpreadAnalyzer._hurst_exponent(ts)
        assert h_features == pytest.approx(h_spread, abs=1e-10), (
            f"Hurst mismatch: features={h_features} spread={h_spread}"
        )


class TestProfitFactorInfHandling:
    """BUG 4: Profit factor with gross_loss=0 should return inf, not crash on round()."""

    def test_profit_factor_all_wins_inf(self):
        """All trades win → gross_loss = 0 → profit_factor = inf."""
        equity = pd.DataFrame({"equity": [10000.0, 10500.0], "cash": [10000.0, 10000.0]})
        trades = pd.DataFrame({
            "bar_index": [1, 3],
            "timestamp": ["2025-01-02", "2025-01-04"],
            "symbol": ["T", "T"],
            "side": ["sell", "sell"],
            "qty": [10, 10],
            "price": [102.0, 103.0],
            "pnl": [200.0, 300.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        assert metrics["profit_factor"] == 999999.0, (
            f"Expected 999999.0, got {metrics['profit_factor']} of type {type(metrics['profit_factor'])}"
        )

    def test_profit_factor_zero_trades(self):
        """No trades → gross_loss = 0, gross_profit = 0 → profit_factor = 0."""
        equity = pd.DataFrame({"equity": [10000.0], "cash": [10000.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        assert metrics["profit_factor"] == 0.0

    def test_profit_factor_no_losses(self):
        """Only winning trades, no losses → should be 999999.0 sentinel."""
        equity = pd.DataFrame({"equity": [10000.0, 11000.0], "cash": [10000.0, 10000.0]})
        trades = pd.DataFrame({
            "bar_index": [1, 2],
            "timestamp": ["2025-01-02", "2025-01-03"],
            "symbol": ["T", "T"],
            "side": ["sell", "sell"],
            "qty": [10, 10],
            "price": [102.0, 103.0],
            "pnl": [200.0, 300.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        assert metrics["profit_factor"] == 999999.0

    def test_profit_factor_no_wins(self):
        """No wins, all losses → profit_factor = 0."""
        equity = pd.DataFrame({"equity": [10000.0, 9000.0], "cash": [10000.0, 10000.0]})
        trades = pd.DataFrame({
            "bar_index": [1],
            "timestamp": ["2025-01-02"],
            "symbol": ["T"],
            "side": ["sell"],
            "qty": [10],
            "price": [98.0],
            "pnl": [-200.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        assert metrics["profit_factor"] == 0.0

    def test_profit_factor_mixed_returns_finite(self):
        """Mixed wins/losses → profit_factor should be a finite float."""
        equity = pd.DataFrame({"equity": [10000.0, 10500.0], "cash": [10000.0, 10000.0]})
        trades = pd.DataFrame({
            "bar_index": [1, 3],
            "timestamp": ["2025-01-02", "2025-01-04"],
            "symbol": ["T", "T"],
            "side": ["sell", "sell"],
            "qty": [10, 10],
            "price": [102.0, 98.0],
            "pnl": [200.0, -100.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        assert np.isfinite(metrics["profit_factor"]), (
            f"Mixed trades profit_factor should be finite, got {metrics['profit_factor']}"
        )
        assert metrics["profit_factor"] == pytest.approx(2.0, abs=0.01)


class TestStructuralBreakExit:
    """BUG 5: Verifies _close_position does not raise UnboundLocalError
    when structural_break_exit=True."""

    def test_structural_break_exit_no_crash(self):
        """Exit due to structural break should not crash with UnboundLocalError
        on ret_pct."""
        np.random.seed(42)
        length = 200
        e1 = np.random.randn(length)
        e2 = np.random.randn(length)
        y = np.cumsum(e1)
        x = 0.5 * y + e2
        prices = pd.DataFrame({"A": y, "B": x})

        strategy = TradingStrategy(
            prices, hedge_ratio=0.5,
            initial_capital=100000.0,
            structural_break_detected=True,
        )
        try:
            equity, trades = strategy.execute()
        except UnboundLocalError as e:
            pytest.fail(f"UnboundLocalError on structural break exit: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected exception on structural break exit: {type(e).__name__}: {e}")

        assert isinstance(equity, pd.Series), "Equity curve should be a Series"
        assert len(equity) > 0, "Equity curve should not be empty"
        assert isinstance(trades, list), "Trades should be a list"

    def test_structural_break_exit_records_trade(self):
        """Exit due to structural break should produce a TradeRecord with
        exit_reason='structural_break'."""
        np.random.seed(42)
        length = 300
        e1 = np.random.randn(length)
        e2 = np.random.randn(length) * 0.5
        y = np.cumsum(e1)
        x = 0.5 * y + e2
        prices = pd.DataFrame({"A": y, "B": x})

        strategy = TradingStrategy(
            prices, hedge_ratio=0.5,
            z_entry=0.5, z_exit=0.0,
            initial_capital=100000.0,
            structural_break_detected=True,
        )
        equity, trades = strategy.execute()

        if trades:
            for t in trades:
                assert t.exit_reason == "structural_break", (
                    f"Expected 'structural_break', got '{t.exit_reason}'"
                )
                assert np.isfinite(t.return_pct), (
                    f"return_pct should be finite, got {t.return_pct}"
                )
                assert np.isfinite(t.pnl), (
                    f"pnl should be finite, got {t.pnl}"
                )

    def test_structural_break_exit_ret_pct_defined(self):
        """ret_pct should be defined (not reference-before-assignment) even
        when structural_break_exit=True."""
        np.random.seed(42)
        length = 200
        e1 = np.random.randn(length)
        e2 = np.random.randn(length)
        y = np.cumsum(e1)
        x = 0.5 * y + e2
        prices = pd.DataFrame({"A": y, "B": x})

        strategy = TradingStrategy(
            prices, hedge_ratio=0.5,
            z_entry=0.5, z_exit=0.0,
            initial_capital=100000.0,
            structural_break_detected=True,
        )
        equity, trades = strategy.execute()

        for t in trades:
            assert t.return_pct is not None, "return_pct should not be None"
            assert not (isinstance(t.return_pct, float) and math.isnan(t.return_pct)), (
                "return_pct should not be NaN"
            )
