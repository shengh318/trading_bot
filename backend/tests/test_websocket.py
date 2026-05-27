from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


class TestWebSocketUnknownAction:
    def test_unknown_action_returns_error(self, api_db):
        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({"action": "invalid"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "unknown" in response["message"].lower()


class TestWebSocketReplay:
    def test_replay_nonexistent_run_returns_error(self, api_db):
        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({"action": "replay", "run_id": 999})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "not found" in response["message"].lower()

    def test_replay_existing_run_streams_bars(self, api_db):
        run_id = api_db.save_backtest_run({
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-01-03",
            "initial_cash": 10000.0,
        })
        api_db.save_backtest_snapshots(run_id, [
            {"bar_index": 0, "timestamp": "2025-01-01", "equity": 10000.0, "cash": 10000.0},
            {"bar_index": 1, "timestamp": "2025-01-02", "equity": 10100.0, "cash": 9000.0},
        ])
        api_db.save_backtest_trades(run_id, [
            {"bar_index": 0, "timestamp": "2025-01-01", "symbol": "AAPL",
             "side": "buy", "qty": 10, "price": 100.0, "pnl": None},
        ])

        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({"action": "replay", "run_id": run_id})

            bar1 = ws.receive_json()
            assert bar1["type"] == "bar"
            assert bar1["bar_index"] == 0
            assert bar1["trade"] is not None

            bar2 = ws.receive_json()
            assert bar2["type"] == "bar"
            assert bar2["bar_index"] == 1
            assert bar2["trade"] is None

            complete = ws.receive_json()
            assert complete["type"] == "complete"
            assert complete["run_id"] == run_id


class TestWebSocketRun:
    @patch("backend.api.websocket.DataLoader")
    def test_run_streams_bars_and_completes(self, mock_loader_cls, api_db):
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        mock_df = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.0] * 5,
            "low": [99.0] * 5, "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [10000] * 5,
        }, index=dates)
        mock_loader_cls.return_value.load_bars.return_value = mock_df

        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({
                "action": "run",
                "strategy_name": "SmaCrossover",
                "symbol": "TEST",
                "start_date": "2025-01-01",
                "end_date": "2025-01-05",
                "initial_cash": 10000.0,
                "parameters": {"short_window": 2, "long_window": 4},
            })

            bars_received = 0
            while True:
                msg = ws.receive_json()
                if msg["type"] == "complete":
                    assert msg["run_id"] is not None
                    break
                elif msg["type"] == "bar":
                    bars_received += 1
                    assert "equity" in msg
                    assert "cash" in msg
                    assert "signal" in msg

            assert bars_received == 5
