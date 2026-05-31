"""Tests for pairs trading API routes (/api/pairs/*)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


class TestAnalyzePair:
    def test_analyze_returns_pair_data(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
            "start": "2020-01-01",
            "end": "2024-01-01",
            "significance": 0.05,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["pair"]["ticker_a"] == "AAPL"
        assert data["pair"]["ticker_b"] == "MSFT"

    def test_analyze_returns_error_when_analysis_fails(self):
        with patch("backend.stats_arb.pipeline.PairAnalyzer.analyze") as mock_analyze:
            mock_analyze.side_effect = ValueError("No data for ticker")
            response = client.post("/api/pairs/analyze", json={
                "ticker_a": "BAD",
                "ticker_b": "DATA",
                "start": "2020-01-01",
            })
            assert response.status_code == 200
            data = response.json()
            assert "error" in data["status"]
            assert data["pair"]["ticker_a"] == "BAD"
            assert data["pair"]["ticker_b"] == "DATA"

    def test_analyze_uppercases_tickers(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "aapl",
            "ticker_b": "msft",
        })
        data = response.json()
        assert data["pair"]["ticker_a"] == "AAPL"
        assert data["pair"]["ticker_b"] == "MSFT"

    def test_analyze_includes_johansen_when_requested(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
            "run_johansen": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "johansen" in data["pair"]

    def test_analyze_returns_spread_series(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "KO",
            "ticker_b": "PEP",
            "start": "2020-01-01",
        })
        data = response.json()
        assert "spread_series" in data["pair"]["spread"]
        assert "zscore_series" in data["pair"]["spread"]

    def test_analyze_returns_backtest_metrics(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
        })
        data = response.json()
        bt = data["pair"]["backtest"]
        assert "sharpe_ratio" in bt
        assert "total_return_pct" in bt
        assert "max_drawdown_pct" in bt

    def test_analyze_returns_regime_data(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
        })
        data = response.json()
        regime = data["pair"]["regime"]
        assert "current_regime" in regime
        assert "trading_allowed" in regime

    def test_analyze_handles_empty_correlation(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
            "start": "2020-01-01",
        })
        data = response.json()
        assert "correlations" in data["pair"]

    def test_analyze_handles_missing_end_date(self):
        response = client.post("/api/pairs/analyze", json={
            "ticker_a": "AAPL",
            "ticker_b": "MSFT",
            "start": "2020-01-01",
        })
        assert response.status_code == 200


class TestRankPairs:
    def test_rank_returns_ranked_pairs(self):
        response = client.post("/api/pairs/rank", json={
            "pairs": [["AAPL", "MSFT"], ["KO", "PEP"]],
            "start": "2020-01-01",
            "top_n": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert isinstance(data["ranked_pairs"], list)

    def test_rank_allows_empty_pairs(self):
        response = client.post("/api/pairs/rank", json={
            "pairs": [],
            "start": "2020-01-01",
        })
        assert response.status_code == 200
        assert response.json()["ranked_pairs"] == []

    def test_rank_handles_single_pair(self):
        response = client.post("/api/pairs/rank", json={
            "pairs": [["NVDA", "AMD"]],
            "top_n": 1,
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["ranked_pairs"]) <= 1

    def test_rank_skips_failed_pairs(self):
        response = client.post("/api/pairs/rank", json={
            "pairs": [["AAPL", "MSFT"]],
            "start": "2020-01-01",
            "end": "2024-01-01",
            "significance": 0.05,
        })
        assert response.status_code == 200


class TestHeatmap:
    def test_heatmap_returns_matrix(self):
        response = client.post("/api/pairs/heatmap", json={
            "tickers": ["NVDA", "AMD", "INTC"],
            "start": "2020-01-01",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "NVDA" in data["matrix"]
        assert "AMD" in data["matrix"]
        assert data["tickers"] == ["NVDA", "AMD", "INTC"]

    def test_heatmap_diagonal_is_zero(self):
        response = client.post("/api/pairs/heatmap", json={
            "tickers": ["AAPL", "MSFT"],
            "start": "2020-01-01",
        })
        data = response.json()
        assert data["matrix"]["AAPL"]["AAPL"] == 0.0
        assert data["matrix"]["MSFT"]["MSFT"] == 0.0

    def test_heatmap_is_symmetric(self):
        response = client.post("/api/pairs/heatmap", json={
            "tickers": ["AAPL", "MSFT"],
            "start": "2020-01-01",
        })
        data = response.json()
        assert data["matrix"]["AAPL"]["MSFT"] == data["matrix"]["MSFT"]["AAPL"]

    def test_heatmap_handles_single_ticker(self):
        response = client.post("/api/pairs/heatmap", json={
            "tickers": ["AAPL"],
            "start": "2020-01-01",
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["matrix"]) == 1
        assert data["matrix"]["AAPL"]["AAPL"] == 0.0

    def test_heatmap_returns_error_on_exception(self):
        with patch("itertools.combinations") as mock_comb:
            mock_comb.side_effect = Exception("Processing error")
            response = client.post("/api/pairs/heatmap", json={
                "tickers": ["AAPL", "MSFT"],
            })
            assert response.status_code == 200
            assert "error" in response.json()["status"]

    def test_heatmap_values_are_non_negative(self):
        response = client.post("/api/pairs/heatmap", json={
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "start": "2020-01-01",
        })
        data = response.json()
        for row in data["matrix"].values():
            for val in row.values():
                assert val >= 0.0
