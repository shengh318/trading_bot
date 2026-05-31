from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


class TestStrategies:
    def test_list_strategies_returns_sma_crossover(self, api_db):
        response = client.get("/api/strategies")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        names = [s["name"] for s in data]
        assert "SmaCrossover" in names

    def test_sma_crossover_has_params(self, api_db):
        response = client.get("/api/strategies")
        sma = [s for s in response.json() if s["name"] == "SmaCrossover"][0]
        param_names = [p["name"] for p in sma["params"]]
        assert "short_window" in param_names
        assert "long_window" in param_names


class TestPortfolioSummary:
    def test_returns_defaults_when_empty(self, api_db):
        response = client.get("/api/portfolio/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["cash"] == 0.0
        assert data["portfolio_value"] == 0.0
        assert data["buying_power"] == 0.0

    def test_returns_populated_values(self, api_db):
        api_db.save_snapshot("2026-01-01T00:00:00", 15000.0, 5000.0, 20000.0)
        response = client.get("/api/portfolio/summary")
        data = response.json()
        assert data["portfolio_value"] == 15000.0
        assert data["cash"] == 5000.0


class TestEquityCurve:
    def test_empty_when_no_snapshots(self, api_db):
        response = client.get("/api/portfolio/equity-curve")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_saved_snapshots(self, api_db):
        api_db.save_snapshot("2026-01-01T00:00:00", 10000.0, 5000.0, 15000.0)
        api_db.save_snapshot("2026-01-02T00:00:00", 11000.0, 6000.0, 17000.0)
        response = client.get("/api/portfolio/equity-curve")
        data = response.json()
        assert len(data) == 2
        assert data[0]["total_equity"] == 10000.0


class TestPositions:
    def test_empty_when_no_positions(self, api_db):
        response = client.get("/api/positions")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_saved_positions(self, api_db):
        api_db.save_positions([{
            "symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0,
            "current_price": 155.0, "unrealized_pl": 50.0, "market_value": 1550.0,
        }])
        response = client.get("/api/positions")
        data = response.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"


class TestOrders:
    def test_empty_when_no_orders(self, api_db):
        response = client.get("/api/orders")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_saved_orders(self, api_db):
        api_db.save_order({
            "id": "ord_1", "symbol": "AAPL", "side": "buy", "qty": 10,
            "filled_qty": 10, "filled_avg_price": 150.0, "status": "filled",
            "type": "market", "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
        })
        response = client.get("/api/orders")
        data = response.json()
        assert len(data) == 1
        assert data[0]["symbol"] == "AAPL"


class TestBacktestRuns:
    def test_empty_when_no_runs(self, api_db):
        response = client.get("/api/backtest/runs")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_saved_runs(self, api_db):
        api_db.save_backtest_run({
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0, "final_equity": 11000.0,
            "total_return": 10.0, "sharpe_ratio": 1.5,
            "max_drawdown": -5.0, "win_rate": 55.0, "num_trades": 5,
        })
        response = client.get("/api/backtest/runs")
        data = response.json()
        assert len(data) == 1
        assert data[0]["strategy_name"] == "SmaCrossover"

    def test_get_run_by_id_not_found(self, api_db):
        response = client.get("/api/backtest/runs/999")
        assert response.status_code == 404

    def test_get_run_by_id_found(self, api_db):
        run_id = api_db.save_backtest_run({
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0,
        })
        response = client.get(f"/api/backtest/runs/{run_id}")
        assert response.status_code == 200
        assert response.json()["id"] == run_id


class TestBacktestRunPost:
    def test_unknown_strategy_returns_400(self, api_db):
        response = client.post("/api/backtest/run", json={
            "strategy_name": "FakeStrategy",
            "symbol": "AAPL",
            "start_date": "2025-01-01",
            "end_date": "2025-06-01",
        })
        assert response.status_code == 400
        assert "Unknown strategy" in response.json()["detail"]

    @patch("backend.data.loader.DataLoader")
    def test_runs_backtest_and_saves_to_db(self, mock_loader_cls, api_db):
        # Prices dip then rise then fall — triggers SMA crossover (buy) then crossunder (sell)
        closes = [100.0, 99.0, 98.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 97.0]
        dates = pd.date_range("2025-01-01", periods=len(closes), freq="D")
        mock_df = pd.DataFrame({
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [10000] * len(closes),
        }, index=dates)
        mock_loader_cls.return_value.load_bars.return_value = mock_df

        response = client.post("/api/backtest/run", json={
            "strategy_name": "SmaCrossover",
            "symbol": "TEST",
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
            "initial_cash": 10000.0,
            "parameters": {"short_window": 2, "long_window": 4},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_name"] == "SmaCrossover"
        assert data["symbol"] == "TEST"

        runs = api_db.get_backtest_runs()
        assert len(runs) >= 1
        trades = api_db.get_backtest_trades(data["id"])
        assert len(trades) > 0
        snapshots = api_db.get_backtest_snapshots(data["id"])
        assert len(snapshots) == len(closes)

    @patch("backend.data.loader.DataLoader")
    def test_backtest_run_returns_metrics(self, mock_loader_cls, api_db):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        mock_df = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.0] * 5,
            "low": [99.0] * 5, "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10000] * 5,
        }, index=dates)
        mock_loader_cls.return_value.load_bars.return_value = mock_df

        response = client.post("/api/backtest/run", json={
            "strategy_name": "SmaCrossover",
            "symbol": "TEST",
            "start_date": "2025-01-01",
            "end_date": "2025-01-05",
            "initial_cash": 10000.0,
            "parameters": {"short_window": 2, "long_window": 4},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["initial_cash"] == 10000.0
