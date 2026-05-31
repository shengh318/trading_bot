"""Tests for live trading API routes (/api/live/status, /api/live/stop)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


class TestLiveStatus:
    def test_status_returns_not_running_when_no_engine(self):
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_get.return_value = None
            response = client.get("/api/live/status")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is False
            assert data["strategy_name"] is None
            assert data["symbols"] == []
            assert data["cash"] == 0.0
            assert data["equity"] == 0.0
            assert data["bar_count"] == 0

    def test_status_returns_not_running_when_engine_stopped(self):
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_engine = MockEngine(running=False)
            mock_get.return_value = mock_engine
            response = client.get("/api/live/status")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is False

    def test_status_returns_running_engine(self):
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_engine = MockEngine(
                running=True,
                strategy_name="SmaCrossover",
                symbols=["AAPL", "MSFT"],
                timeframe_str="1Day",
                portfolio_cash=10000.0,
                portfolio_positions={},
                data_buffers={},
                bar_count=42,
            )
            mock_get.return_value = mock_engine
            response = client.get("/api/live/status")
            assert response.status_code == 200
            data = response.json()
            assert data["running"] is True
            assert data["strategy_name"] == "SmaCrossover"
            assert data["symbols"] == ["AAPL", "MSFT"]
            assert data["timeframe"] == "1Day"
            assert data["cash"] == 10000.0
            assert data["bar_count"] == 42

    def test_status_equity_with_positions(self):
        import pandas as pd
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            df = pd.DataFrame({"close": [150.0]}, index=pd.to_datetime(["2024-01-01"]))
            mock_engine = MockEngine(
                running=True,
                strategy_name="SmaCrossover",
                symbols=["AAPL"],
                timeframe_str="1Day",
                portfolio_cash=5000.0,
                portfolio_positions={"AAPL": 10},
                data_buffers={"AAPL": df},
                bar_count=5,
            )
            mock_get.return_value = mock_engine
            response = client.get("/api/live/status")
            data = response.json()
            assert data["equity"] == 5000.0 + 10 * 150.0  # cash + position value

    def test_status_handles_empty_buffer(self):
        import pandas as pd
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_engine = MockEngine(
                running=True,
                strategy_name="SmaCrossover",
                symbols=["AAPL"],
                timeframe_str="1Day",
                portfolio_cash=5000.0,
                portfolio_positions={"AAPL": 10},
                data_buffers={"AAPL": pd.DataFrame()},
                bar_count=0,
            )
            mock_get.return_value = mock_engine
            response = client.get("/api/live/status")
            data = response.json()
            assert data["equity"] == 5000.0  # buffer empty, no position value added


class TestLiveStop:
    def test_stop_stops_running_engine(self):
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_engine = MockEngine(running=True)
            mock_get.return_value = mock_engine
            response = client.post("/api/live/stop")
            assert response.status_code == 200
            assert mock_engine.stopped is True

    def test_stop_no_engine_returns_ok(self):
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_get.return_value = None
            response = client.post("/api/live/stop")
            assert response.status_code == 200
            assert response.json()["status"] == "stopped"

    def test_stop_already_stopped_engine_returns_ok(self):
        with patch("backend.api.live_routes.get_live_engine") as mock_get:
            mock_engine = MockEngine(running=False)
            mock_get.return_value = mock_engine
            response = client.post("/api/live/stop")
            assert response.status_code == 200


class MockEngine:
    """Minimal mock of LiveEngine for testing API routes."""

    def __init__(self, running=False, strategy_name=None, symbols=None,
                 timeframe_str=None, portfolio_cash=0.0, portfolio_positions=None,
                 data_buffers=None, bar_count=0):
        self.running = running
        self.strategy_name = strategy_name
        self.symbols = symbols or []
        self.timeframe_str = timeframe_str
        self._bar_count = bar_count
        self.stopped = False

        class MockPortfolio:
            def __init__(self, cash, positions):
                self.cash = cash
                self.positions = positions

        self.portfolio = MockPortfolio(portfolio_cash, portfolio_positions or {})
        self.data_buffers = data_buffers or {}

    def stop(self):
        self.stopped = True
        self.running = False
