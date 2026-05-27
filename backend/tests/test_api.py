from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


class TestHealth:
    def test_health_endpoint_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_misconfigured_without_keys(self):
        with patch("backend.api.main.validate_config") as mock_validate:
            mock_validate.return_value = ["ALPACA_API_KEY is not set.", "ALPACA_SECRET_KEY is not set."]
            response = client.get("/api/health")
            data = response.json()
            assert data["status"] == "misconfigured"
            assert len(data["errors"]) == 2

    def test_health_returns_ok_when_configured(self):
        with patch("backend.api.main.validate_config") as mock_validate:
            mock_validate.return_value = []
            response = client.get("/api/health")
            data = response.json()
            assert data["status"] == "ok"
            assert data["errors"] == []

    def test_health_returns_cors_header(self):
        response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
