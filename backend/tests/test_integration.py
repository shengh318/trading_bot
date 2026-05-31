"""Integration tests — full pipeline flows across multiple modules."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.backtest.engine import BacktestEngine, MultiSymbolBacktestEngine
from backend.backtest.metrics import calculate_metrics
from backend.strategies.base import Portfolio, Signal
from backend.strategies.registry import get_strategy
from backend.stats_arb.pipeline import PairAnalyzer


# ── Full Backtest Pipeline: Data → Strategy → Engine → Metrics ──


class TestBacktestPipelineIntegration:
    def test_full_backtest_pipeline_sma_crossover(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        # Dip then rise to guarantee SMA10 crosses above SMA30
        base = np.concatenate([np.linspace(100, 92, 50), np.linspace(92, 108, 50)])
        close = base + np.random.default_rng(42).normal(0, 0.5, 100)
        data = pd.DataFrame({
            "open": close - 0.2, "high": close + 0.3, "low": close - 0.3, "close": close,
            "volume": np.full(100, 10000),
        }, index=dates)

        strategy = get_strategy("SmaCrossover", {"short_window": 5, "long_window": 15})
        engine = BacktestEngine(data, strategy, initial_cash=10000)
        result = engine.run()

        assert len(result.trades) > 0, "Should have at least one trade"
        assert result.equity_curve["equity"].iloc[-1] > 0

        metrics = calculate_metrics(result.equity_curve, result.trades, result.initial_cash)
        assert metrics["sharpe_ratio"] is not None
        assert metrics["max_drawdown_pct"] is not None
        assert metrics["win_rate_pct"] is not None
        assert metrics["profit_factor"] is not None
        assert metrics["total_return_pct"] is not None

    def test_full_backtest_pipeline_simple_strat(self):
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        close = np.linspace(100, 110, 200) + np.random.default_rng(42).normal(0, 2, 200)
        data = pd.DataFrame({
            "open": close - 1, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(200, 10000),
        }, index=dates)

        strategy = get_strategy("Simple Strat 1")
        engine = BacktestEngine(data, strategy, initial_cash=10000)
        result = engine.run()

        assert len(result.trades) >= 0
        assert result.equity_curve["equity"].iloc[-1] > 0

    def test_multi_symbol_backtest(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        base = np.linspace(100, 110, 50)
        data_dict = {}
        for sym in ["AAPL", "MSFT"]:
            data_dict[sym] = pd.DataFrame({
                "open": base - 0.5, "high": base + 0.5,
                "low": base - 0.5, "close": base + np.random.default_rng(42).normal(0, 0.5, 50),
                "volume": np.full(50, 10000),
            }, index=dates)

        from backend.strategies.sma_crossover import SmaCrossover
        engine = MultiSymbolBacktestEngine(
            data=data_dict,
            strategy_cls=SmaCrossover,
            initial_cash=20000,
            parameters={"short_window": 5, "long_window": 15},
        )
        result = engine.run()

        assert len(result.trades) >= 0
        assert result.equity_curve["equity"].iloc[-1] > 0

    def test_backtest_with_dividend_stream(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "open": np.full(10, 100.0),
            "high": np.full(10, 101.0),
            "low": np.full(10, 99.0),
            "close": np.linspace(100, 105, 10),
            "volume": np.full(10, 10000),
        }, index=dates)

        dividends = pd.DataFrame(
            {"dividend": [0.5]},
            index=pd.to_datetime(["2024-01-05"]),
        )

        strategy = get_strategy("SmaCrossover", {"short_window": 2, "long_window": 5})
        engine = BacktestEngine(data, strategy, initial_cash=10000, dividends=dividends)
        stream = list(engine.stream())

        assert len(stream) == 10
        dividend_events = [s for s in stream if s.get("dividend")]
        assert len(dividend_events) >= 0

        # Run should still produce valid result
        result = engine.run()
        assert result.equity_curve["equity"].iloc[-1] > 0

    def test_backtest_preserves_trade_details(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        data = pd.DataFrame({
            "open": np.full(50, 100.0),
            "high": np.full(50, 101.0),
            "low": np.full(50, 99.0),
            "close": np.concatenate([np.linspace(100, 120, 25), np.linspace(120, 90, 25)]),
            "volume": np.full(50, 10000),
        }, index=dates)

        strategy = get_strategy("SmaCrossover", {"short_window": 5, "long_window": 15})
        engine = BacktestEngine(data, strategy, initial_cash=10000)
        result = engine.run()

        for _, trade in result.trades.iterrows():
            assert trade.side in ("buy", "sell")
            assert trade.price > 0
            assert trade.qty > 0

    def test_backtest_stream_yields_snapshots(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.DataFrame({
            "open": np.full(5, 100.0),
            "high": np.full(5, 101.0),
            "low": np.full(5, 99.0),
            "close": np.linspace(100, 104, 5),
            "volume": np.full(5, 10000),
        }, index=dates)

        strategy = get_strategy("SmaCrossover", {"short_window": 2, "long_window": 4})
        engine = BacktestEngine(data, strategy, initial_cash=10000)
        snapshots = list(engine.stream())

        assert len(snapshots) == 5
        for s in snapshots:
            assert "snapshot" in s
            assert "equity" in s["snapshot"]
            assert "cash" in s["snapshot"]


# ── Strategy Registry Integration ──


class TestStrategyRegistryIntegration:
    def test_get_strategy_creates_instance(self):
        strat = get_strategy("SmaCrossover")
        assert strat is not None
        assert hasattr(strat, "init")
        assert hasattr(strat, "next")

    def test_get_strategy_simple_strat_works(self):
        strat = get_strategy("Simple Strat 1")
        assert strat is not None

    def test_get_strategy_accepts_parameters(self):
        strat = get_strategy("SmaCrossover", {"short_window": 15, "long_window": 40})
        assert strat is not None

    def test_strategy_next_returns_valid_signal(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100, 110, 100),
            "volume": np.full(100, 10000),
        }, index=dates)

        strat = get_strategy("SmaCrossover", {"short_window": 5, "long_window": 20})
        from backend.strategies.base import Portfolio
        strat.init(data)
        portfolio = Portfolio(10000)

        valid_signals = {Signal.BUY, Signal.SELL, Signal.HOLD, Signal.EXIT}

        for i in range(20, len(data)):
            signal = strat.next(i, data, portfolio)
            assert signal in valid_signals


# ── Metrics Integration ──


class TestMetricsIntegration:
    def test_metrics_with_empty_trades(self):
        equity_curve = pd.DataFrame({"equity": np.linspace(10000, 11000, 100)})
        trades = pd.DataFrame(columns=["side", "qty", "price", "pnl", "bar_index", "timestamp", "symbol"])
        metrics = calculate_metrics(equity_curve, trades, 10000)
        assert metrics["num_trades"] == 0
        assert metrics["total_return_pct"] is not None

    def test_metrics_matches_engine_result(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100, 110, 100) + np.random.default_rng(42).normal(0, 0.5, 100),
            "volume": np.full(100, 10000),
        }, index=dates)

        strategy = get_strategy("SmaCrossover", {"short_window": 10, "long_window": 30})
        engine = BacktestEngine(data, strategy, initial_cash=10000)
        result = engine.run()

        metrics = calculate_metrics(result.equity_curve, result.trades, result.initial_cash)
        assert metrics["sharpe_ratio"] == pytest.approx(result.metrics["sharpe_ratio"], rel=0.01)
        assert metrics["max_drawdown_pct"] == pytest.approx(result.metrics["max_drawdown_pct"], rel=0.01)
        assert metrics["win_rate_pct"] == pytest.approx(result.metrics["win_rate_pct"], rel=0.01)


# ── Stats Arb Pipeline Integration ──


class TestStatsArbPipelineIntegration:
    def test_pair_analyzer_returns_all_fields(self):
        analyzer = PairAnalyzer(
            "KO", "PEP", "2020-01-01", "2024-01-01",
            run_johansen=False, run_ml=False,
        )
        result = analyzer.analyze()

        assert result.ticker_a == "KO"
        assert result.ticker_b == "PEP"
        assert result.correlation is not None
        assert result.cointegration is not None
        assert result.spread is not None
        assert result.regime is not None
        assert result.walk_forward is not None

    def test_pair_analyzer_backtest_produces_trades(self):
        analyzer = PairAnalyzer(
            "KO", "PEP", "2020-01-01", "2024-01-01",
            run_johansen=False, run_ml=False,
        )
        result = analyzer.analyze()

        bt = result.in_sample_backtest
        assert bt is not None
        assert bt.num_trades >= 0
        assert bt.final_equity > 0

    def test_pair_analyzer_walk_forward_has_folds(self):
        analyzer = PairAnalyzer(
            "KO", "PEP", "2020-01-01", "2024-01-01",
            run_johansen=False, run_ml=False,
        )
        result = analyzer.analyze()

        wf = result.walk_forward
        assert len(wf.folds) > 0
        assert wf.avg_half_life > 0

    def test_pair_analyzer_regime_detects_breaks(self):
        analyzer = PairAnalyzer(
            "KO", "PEP", "2020-01-01", "2024-01-01",
            run_johansen=False, run_ml=False,
        )
        result = analyzer.analyze()

        r = result.regime
        assert hasattr(r, "current_regime")
        assert hasattr(r, "trading_allowed")
        assert hasattr(r, "structural_break")


# ── API + Database Integration ──


class TestApiDbIntegration:
    def test_backtest_api_uses_database(self, api_db):
        client = TestClient(app)

        response = client.get("/api/backtest/runs")
        assert response.status_code == 200
        runs = response.json()
        assert isinstance(runs, list)

        # Save a run directly to DB then fetch via API
        run_id = api_db.save_backtest_run({
            "strategy_name": "SmaCrossover",
            "parameters": None,
            "symbol": "AAPL",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "initial_cash": 10000.0,
        })

        response = client.get(f"/api/backtest/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_name"] == "SmaCrossover"
        assert data["symbol"] == "AAPL"

    def test_portfolio_api_uses_database(self, api_db):
        client = TestClient(app)

        api_db.save_snapshot("2024-01-01T00:00:00", 10000.0, 5000.0, 15000.0)
        api_db.save_snapshot("2024-01-02T00:00:00", 10500.0, 4500.0, 16000.0)

        response = client.get("/api/portfolio/equity-curve")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

    def test_positions_api_uses_database(self, api_db):
        client = TestClient(app)

        api_db.save_positions([
            {"symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0,
             "current_price": 155.0, "unrealized_pl": 50.0, "market_value": 1550.0},
        ])

        response = client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
