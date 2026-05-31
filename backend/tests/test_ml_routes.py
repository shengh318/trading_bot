"""Tests for ML training API routes (/api/ml/*)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


class TestMlModels:
    def test_list_models_returns_empty_when_no_models(self):
        with patch("backend.api.ml_routes.list_models") as mock_list:
            mock_list.return_value = []
            response = client.get("/api/ml/models")
            assert response.status_code == 200
            assert response.json() == []

    def test_list_models_returns_models(self):
        mock_models = [
            {"name": "RF_v1", "version": 1, "model_type": "rf",
             "train_date": "2025-01-01", "train_symbols": ["AAPL"],
             "context_symbols": [], "validation_metrics": {},
             "beat_baselines": False, "versions": [1]},
        ]
        with patch("backend.api.ml_routes.list_models") as mock_list:
            mock_list.return_value = mock_models
            response = client.get("/api/ml/models")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "RF_v1"

    def test_get_model_returns_404_for_unknown(self):
        with patch("backend.api.ml_routes.list_models") as mock_list:
            mock_list.return_value = []
            response = client.get("/api/ml/models/unknown")
            assert response.status_code == 404

    def test_get_model_returns_model_info(self):
        mock_models = [
            {"name": "RF_v1", "version": 1, "model_type": "rf",
             "train_date": "2025-01-01", "train_symbols": ["AAPL"],
             "context_symbols": [], "validation_metrics": {},
             "beat_baselines": False, "versions": [1]},
        ]
        with patch("backend.api.ml_routes.list_models") as mock_list:
            mock_list.return_value = mock_models
            response = client.get("/api/ml/models/RF_v1")
            assert response.status_code == 200
            assert response.json()["name"] == "RF_v1"

    def test_delete_model_calls_delete(self):
        with patch("backend.api.ml_routes.delete_model") as mock_delete:
            response = client.delete("/api/ml/models/test_model")
            assert response.status_code == 200
            mock_delete.assert_called_with("test_model", version=None)

    def test_delete_model_with_version(self):
        with patch("backend.api.ml_routes.delete_model") as mock_delete:
            response = client.delete("/api/ml/models/test_model?version=2")
            assert response.status_code == 200
            mock_delete.assert_called_with("test_model", version=2)


class TestMlRetrain:
    def test_retrain_starts_training(self):
        with patch("backend.api.ml_routes.subprocess.Popen") as mock_popen, \
             patch("backend.api.ml_routes.Path.mkdir"), \
             patch("backend.api.ml_routes.Path.exists", return_value=False):
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc

            response = client.post("/api/ml/retrain", json={
                "symbols": "AAPL",
                "years": 5,
                "name": "test_model",
                "model_types": "rf,gbt",
                "beat_baselines": False,
                "grid_search": False,
                "walk_forward": 0,
                "stacking": False,
                "meta_labeling": False,
                "regularize": False,
                "prune": 0,
                "kelly": False,
                "auto_threshold": False,
                "labeling": "next_bar",
                "forecast_horizon": 1,
                "regime_aware": False,
            })
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert data["pid"] == 12345
            assert "test_model" in data["message"]

    def test_retrain_generates_name_when_empty(self):
        with patch("backend.api.ml_routes.subprocess.Popen") as mock_popen, \
             patch("backend.api.ml_routes.Path.mkdir"), \
             patch("backend.api.ml_routes.Path.exists", return_value=False):
            mock_proc = MagicMock()
            mock_proc.pid = 12346
            mock_popen.return_value = mock_proc

            response = client.post("/api/ml/retrain", json={
                "symbols": "AAPL",
                "years": 5,
                "name": "",
                "model_types": "rf",
                "beat_baselines": False,
                "grid_search": False,
                "walk_forward": 0,
                "stacking": False,
                "meta_labeling": False,
                "regularize": False,
                "prune": 0,
                "kelly": False,
                "auto_threshold": False,
                "labeling": "next_bar",
                "forecast_horizon": 1,
                "regime_aware": False,
            })
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert "model_" in data["message"]

    def test_retrain_status_unknown_for_nonexistent(self):
        with patch("backend.api.ml_routes._active_pids", {}), \
             patch("backend.api.ml_routes.list_models", return_value=[]):
            response = client.get("/api/ml/retrain/status/nonexistent")
            assert response.status_code == 200
            assert response.json()["status"] == "unknown"

    def test_retrain_status_completed_when_model_exists(self):
        mock_models = [
            {"name": "existing_model", "version": 1, "model_type": "rf",
             "train_date": "2025-01-01", "train_symbols": ["AAPL"],
             "context_symbols": [], "validation_metrics": {},
             "beat_baselines": False, "versions": [1]},
        ]
        with patch("backend.api.ml_routes._active_pids", {}), \
             patch("backend.api.ml_routes.list_models", return_value=mock_models):
            response = client.get("/api/ml/retrain/status/existing_model")
            assert response.status_code == 200
            assert response.json()["status"] == "completed"

    def test_retrain_status_running_when_pid_active(self):
        with patch("backend.api.ml_routes._active_pids", {"running_model": 99999}), \
             patch("backend.api.ml_routes.os.kill") as mock_kill:
            mock_kill.return_value = None
            response = client.get("/api/ml/retrain/status/running_model")
            assert response.status_code == 200
            assert response.json()["status"] == "running"

    def test_retrain_status_completed_when_pid_inactive(self):
        with patch("backend.api.ml_routes._active_pids", {"finished_model": 99999}), \
             patch("backend.api.ml_routes.os.kill") as mock_kill, \
             patch("backend.api.ml_routes._save_pids"):
            mock_kill.side_effect = OSError("No such process")
            response = client.get("/api/ml/retrain/status/finished_model")
            assert response.status_code == 200
            assert response.json()["status"] == "completed"

    def test_retrain_builds_correct_command_with_all_flags(self):
        with patch("backend.api.ml_routes.subprocess.Popen") as mock_popen, \
             patch("backend.api.ml_routes.Path.mkdir"), \
             patch("backend.api.ml_routes.Path.exists", return_value=False):
            mock_proc = MagicMock()
            mock_proc.pid = 12347
            mock_popen.return_value = mock_proc

            response = client.post("/api/ml/retrain", json={
                "symbols": "AAPL,MSFT",
                "years": 10,
                "name": "full_test",
                "model_types": "rf,gbt,xgb",
                "beat_baselines": True,
                "grid_search": True,
                "walk_forward": 5,
                "stacking": True,
                "meta_labeling": True,
                "regularize": True,
                "prune": 3,
                "kelly": True,
                "auto_threshold": True,
                "labeling": "triple_barrier",
                "forecast_horizon": 3,
                "context_symbols": "SPY,VOO",
                "multi_horizon": "1,3,5",
                "regime_aware": True,
                "embargo": 10,
                "cutoff_date": "2023-01-01",
                "val_split": 0.7,
                "triple_barrier_pct": 0.03,
                "triple_barrier_max_bars": 15,
                "n_estimators": 300,
                "max_depth": 15,
                "learning_rate": 0.05,
                "confidence_threshold": 0.6,
                "base_buy_size": 2000,
                "model_dir": "/tmp/models",
            })
            assert response.status_code == 200

    def test_retrain_handles_subprocess_error(self):
        with patch("backend.api.ml_routes.subprocess.Popen") as mock_popen, \
             patch("backend.api.ml_routes.Path.mkdir"), \
             patch("backend.api.ml_routes.Path.exists", return_value=False):
            mock_popen.side_effect = FileNotFoundError("python not found")

            response = client.post("/api/ml/retrain", json={
                "symbols": "AAPL",
                "years": 5,
                "name": "fail_test",
                "model_types": "rf",
                "beat_baselines": False,
                "grid_search": False,
                "walk_forward": 0,
                "stacking": False,
                "meta_labeling": False,
                "regularize": False,
                "prune": 0,
                "kelly": False,
                "auto_threshold": False,
                "labeling": "next_bar",
                "forecast_horizon": 1,
                "regime_aware": False,
            })
            assert response.status_code == 500
