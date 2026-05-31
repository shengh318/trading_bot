"""Tests for Pearson correlation endpoint (GET /api/correlation/data).

Covers:
- _safe helper function edge cases
- API endpoint: success, error codes, response structure
- Model validation
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.models import CorrelationDataResponse, CorrelationPoint

client = TestClient(app)


# ── _safe helper ──────────────────────────────────────────────────────────


class TestSafe:
    def test_returns_finite_value(self):
        from backend.api.ml_routes import _safe
        assert _safe(3.14) == 3.14

    def test_returns_zero_for_none(self):
        from backend.api.ml_routes import _safe
        assert _safe(None) == 0.0

    def test_returns_zero_for_nan(self):
        from backend.api.ml_routes import _safe
        assert _safe(float("nan")) == 0.0

    def test_returns_zero_for_inf(self):
        from backend.api.ml_routes import _safe
        assert _safe(float("inf")) == 0.0

    def test_returns_zero_for_neg_inf(self):
        from backend.api.ml_routes import _safe
        assert _safe(float("-inf")) == 0.0

    def test_custom_default(self):
        from backend.api.ml_routes import _safe
        assert _safe(None, default=-1.0) == -1.0

    def test_handles_string_input(self):
        from backend.api.ml_routes import _safe
        assert _safe("bad") == 0.0


# ── Correlation API endpoint ──────────────────────────────────────────────


def _mock_yf_download(symbols, start, end, auto_adjust, progress):
    """Return a realistic-looking DataFrame mimicking yfinance output."""
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    np.random.seed(42)
    a = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    b = 150.0 + np.cumsum(np.random.randn(n) * 0.3)
    midx = pd.MultiIndex.from_product([["Close"], list(symbols)])
    df = pd.DataFrame(
        np.column_stack([a, b]),
        index=dates,
        columns=midx,
    )
    df.index.name = "Date"
    return df


class TestCorrelationEndpoint:
    def test_returns_valid_response_structure(self, api_db):
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?symbol_a=NVDA&symbol_b=SPY&years=1&windows=20,60")
        assert resp.status_code == 200
        data = resp.json()

        assert data["symbol_a"] == "NVDA"
        assert data["symbol_b"] == "SPY"

        assert "correlations" in data
        assert "20" in data["correlations"]
        assert "60" in data["correlations"]
        for w_key, pts in data["correlations"].items():
            assert len(pts) > 0
            for pt in pts:
                assert "time" in pt
                assert "value" in pt
                assert isinstance(pt["value"], float)

        assert "cumulative_returns" in data
        assert "NVDA" in data["cumulative_returns"]
        assert "SPY" in data["cumulative_returns"]
        for sym, pts in data["cumulative_returns"].items():
            assert len(pts) > 0
            assert abs(pts[0]["value"] - 100.0) < 1e-6

        assert "statistics" in data
        stats = data["statistics"]
        for key in ("pearson_r", "pearson_p", "spearman_r", "spearman_p",
                     "kendall_tau", "kendall_p", "rolling_corr_std", "oos_corr_drop"):
            assert key in stats, f"Missing statistic: {key}"

    def test_multiple_windows_returned(self, api_db):
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?windows=10,30,90")
        data = resp.json()
        assert set(data["correlations"].keys()) == {"10", "30", "90"}

    def test_single_window(self, api_db):
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?windows=50")
        data = resp.json()
        assert list(data["correlations"].keys()) == ["50"]

    def test_statistics_are_reasonable_values(self, api_db):
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?symbol_a=NVDA&symbol_b=SPY&years=2")
        stats = resp.json()["statistics"]
        assert -1.0 <= stats["pearson_r"] <= 1.0
        assert 0.0 <= stats["pearson_p"] <= 1.0
        assert -1.0 <= stats["spearman_r"] <= 1.0
        assert -1.0 <= stats["kendall_tau"] <= 1.0
        assert stats["rolling_corr_std"] >= 0.0
        assert stats["oos_corr_drop"] >= 0.0

    def test_returns_502_on_empty_data(self, api_db):
        def empty_download(*args, **kwargs):
            return pd.DataFrame()
        with patch("yfinance.download", empty_download):
            resp = client.get("/api/correlation/data")
        assert resp.status_code == 502
        assert "No data" in resp.json()["detail"]

    def test_returns_502_when_close_column_missing(self, api_db):
        def bad_download(*args, **kwargs):
            return pd.DataFrame({"Open": [1.0]})
        with patch("yfinance.download", bad_download):
            resp = client.get("/api/correlation/data")
        assert resp.status_code == 502

    def test_returns_404_when_symbol_not_in_data(self, api_db):
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        midx = pd.MultiIndex.from_product([["Close"], ["SPY"]])
        df = pd.DataFrame(np.random.randn(10), index=dates, columns=midx)
        with patch("yfinance.download", return_value=df):
            resp = client.get("/api/correlation/data?symbol_a=NVDA&symbol_b=SPY")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_returns_422_on_too_few_points(self, api_db):
        """pct_change drops 1 row, so 4 raw rows → 3 returns → <5 raises 422."""
        dates = pd.date_range("2025-01-01", periods=4, freq="D")
        midx = pd.MultiIndex.from_product([["Close"], ["A", "B"]])
        df = pd.DataFrame(
            np.random.randn(4, 2),
            index=dates,
            columns=midx,
        )
        with patch("yfinance.download", return_value=df):
            resp = client.get("/api/correlation/data?symbol_a=A&symbol_b=B&years=1")
        assert resp.status_code == 422
        assert "Not enough data" in resp.json()["detail"]

    def test_correlation_value_range(self, api_db):
        """Each correlation point should be between -1 and 1."""
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?windows=20")
        data = resp.json()
        for pts in data["correlations"].values():
            for pt in pts:
                assert -1.0 <= pt["value"] <= 1.0, (
                    f"Correlation {pt['value']} out of [-1, 1]"
                )

    def test_oos_corr_drop_is_non_negative(self, api_db):
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?years=3")
        stats = resp.json()["statistics"]
        assert stats["oos_corr_drop"] >= 0.0

    def test_identical_series_produces_correlation_one(self, api_db):
        """Two identical return series should give approx pearson_r = 1.0."""
        def identical_download(symbols, **kwargs):
            dates = pd.date_range("2025-01-01", periods=252, freq="D")
            vals = np.random.randn(252).cumsum() + 100.0
            midx = pd.MultiIndex.from_product([["Close"], list(symbols)])
            df = pd.DataFrame(
                np.column_stack([vals, vals]),
                index=dates,
                columns=midx,
            )
            df.index.name = "Date"
            return df
        with patch("yfinance.download", identical_download):
            resp = client.get("/api/correlation/data?symbol_a=A&symbol_b=B&years=1")
        pearson_r = resp.json()["statistics"]["pearson_r"]
        assert abs(pearson_r - 1.0) < 1e-4, f"Expected ~1.0, got {pearson_r}"

    def test_inverse_series_produces_correlation_neg_one(self, api_db):
        """Two perfectly inverse return series should give pearson_r ≈ -1.0."""
        def inverse_download(symbols, **kwargs):
            dates = pd.date_range("2025-01-01", periods=252, freq="D")
            np.random.seed(42)
            raw = np.random.randn(252) * 0.01
            a = 100.0 * (1 + np.cumsum(raw))
            b = 100.0 * (1 + np.cumsum(-raw))
            midx = pd.MultiIndex.from_product([["Close"], list(symbols)])
            df = pd.DataFrame(
                np.column_stack([a, b]),
                index=dates,
                columns=midx,
            )
            df.index.name = "Date"
            return df
        with patch("yfinance.download", inverse_download):
            resp = client.get("/api/correlation/data?symbol_a=A&symbol_b=B&years=1")
        pearson_r = resp.json()["statistics"]["pearson_r"]
        assert pearson_r < -0.99, f"Expected ≈ -1.0, got {pearson_r}"

    def test_default_parameters(self, api_db):
        """Defaults should be NVDA, SPY, 5 years, windows=20,60,120."""
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol_a"] == "NVDA"
        assert data["symbol_b"] == "SPY"
        assert set(data["correlations"].keys()) == {"20", "60", "120"}

    def test_cumulative_returns_begin_at_100(self, api_db):
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data")
        data = resp.json()
        for sym, pts in data["cumulative_returns"].items():
            assert pts[0]["value"] == pytest.approx(100.0, abs=1e-4)

    def test_yfinance_timeout_handling(self, api_db):
        """yfinance returning empty raises 502, not 500."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            resp = client.get("/api/correlation/data")
        assert resp.status_code == 502

    def test_response_matches_pydantic_model(self, api_db):
        """Response should deserialize cleanly into CorrelationDataResponse."""
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data")
        model = CorrelationDataResponse(**resp.json())
        assert model.symbol_a == resp.json()["symbol_a"]
        assert isinstance(model.correlations, dict)
        for pts in model.correlations.values():
            assert all(isinstance(p, CorrelationPoint) for p in pts)

    # ── Bug-revealing tests ─────────────────────────────────────────────

    def test_leading_nan_in_prices_produces_non_empty_cumulative_returns(self, api_db):
        """When symbol_a starts trading later, its leading NaN should not
        cause all-NaN cumulative returns.  The first non-NaN close should be
        used as the normalization base."""
        def late_start_download(symbols, **kwargs):
            dates = pd.date_range("2025-01-01", periods=100, freq="D")
            midx = pd.MultiIndex.from_product([["Close"], list(symbols)])
            a = 100.0 + np.arange(100, dtype=float)
            b = 200.0 + np.arange(100, dtype=float)
            a[:20] = np.nan  # symbol_a has no data for first 20 days
            df = pd.DataFrame(
                np.column_stack([a, b]),
                index=dates,
                columns=midx,
            )
            df.index.name = "Date"
            return df
        with patch("yfinance.download", late_start_download):
            resp = client.get("/api/correlation/data?symbol_a=A&symbol_b=B&years=1")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert len(data["cumulative_returns"]["A"]) > 0, (
            "Symbol A cumulative returns should not be empty despite leading NaN"
        )
        for sym, pts in data["cumulative_returns"].items():
            assert all(isinstance(p["value"], float) and not np.isnan(p["value"]) for p in pts), (
                f"{sym} cumulative returns contain NaN"
            )

    def test_invalid_windows_param_returns_422(self, api_db):
        """Non-numeric windows like 'abc' should return 422, not 500."""
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?windows=abc")
        assert resp.status_code == 422, (
            f"Expected 422 for invalid window, got {resp.status_code}: {resp.text}"
        )

    def test_empty_windows_string_does_not_crash(self, api_db):
        """An empty windows string should not crash the endpoint."""
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?windows=")
        assert resp.status_code == 200
        data = resp.json()
        assert data["correlations"] == {}

    def test_oos_corr_drop_rounded_consistently(self, api_db):
        """oos_corr_drop should be rounded (like other statistics)."""
        with patch("yfinance.download", _mock_yf_download):
            resp = client.get("/api/correlation/data?years=3")
        stats = resp.json()["statistics"]
        oos_val = stats["oos_corr_drop"]
        assert isinstance(oos_val, float)
        if oos_val != 0.0:
            places = len(str(oos_val).split(".")[1])
            assert places <= 4, f"oos_corr_drop {oos_val} has {places} decimal places, expected ≤4"

    def test_all_nan_column_returns_502(self, api_db):
        """When yfinance returns all NaN for one symbol, should return 502 not 500."""
        def nan_column_download(symbols, **kwargs):
            dates = pd.date_range("2025-01-01", periods=50, freq="D")
            midx = pd.MultiIndex.from_product([["Close"], list(symbols)])
            a = 100.0 + np.arange(50, dtype=float)
            b = np.full(50, np.nan)  # second symbol has no data
            df = pd.DataFrame(
                np.column_stack([a, b]),
                index=dates,
                columns=midx,
            )
            df.index.name = "Date"
            return df
        with patch("yfinance.download", nan_column_download):
            resp = client.get("/api/correlation/data?symbol_a=A&symbol_b=B&years=1")
        assert resp.status_code == 502, (
            f"Expected 502 for all-NaN column, got {resp.status_code}: {resp.text}"
        )

    def test_constant_prices_do_not_crash(self, api_db):
        """When both symbols have constant prices, endpoint should not crash."""
        def flat_download(symbols, **kwargs):
            dates = pd.date_range("2025-01-01", periods=50, freq="D")
            midx = pd.MultiIndex.from_product([["Close"], list(symbols)])
            df = pd.DataFrame(
                np.column_stack([np.full(50, 100.0), np.full(50, 200.0)]),
                index=dates,
                columns=midx,
            )
            df.index.name = "Date"
            return df
        with patch("yfinance.download", flat_download):
            resp = client.get("/api/correlation/data?symbol_a=A&symbol_b=B&years=1")
        assert resp.status_code == 200, (
            f"Expected 200 for constant prices, got {resp.status_code}: {resp.text}"
        )

    def test_pct_change_no_future_warning(self, api_db):
        """pct_change should use fill_method=None to suppress FutureWarning."""
        import warnings
        from backend.api.ml_routes import get_correlation_data
        import inspect
        source = inspect.getsource(get_correlation_data)
        occurrences = source.count("pct_change()")
        # All pct_change calls should use fill_method=None
        assert "fill_method=None" in source, (
            "pct_change() must use fill_method=None to suppress FutureWarning"
        )


# ── Model validation ──────────────────────────────────────────────────────


class TestCorrelationModels:
    def test_correlation_point_valid(self):
        pt = CorrelationPoint(time="2025-01-01", value=0.75)
        assert pt.time == "2025-01-01"
        assert pt.value == 0.75

    def test_correlation_point_negative_value(self):
        pt = CorrelationPoint(time="2025-01-01", value=-0.5)
        assert pt.value == -0.5

    def test_correlation_point_boundary(self):
        pt = CorrelationPoint(time="2025-01-01", value=1.0)
        assert pt.value == 1.0
        pt = CorrelationPoint(time="2025-01-01", value=-1.0)
        assert pt.value == -1.0

    def test_correlation_data_response_empty_correlations(self):
        resp = CorrelationDataResponse(
            symbol_a="A",
            symbol_b="B",
            correlations={},
            cumulative_returns={"A": [], "B": []},
            statistics={"pearson_r": 0.0},
        )
        assert resp.correlations == {}
        assert resp.statistics["pearson_r"] == 0.0

    def test_correlation_data_response_full(self):
        resp = CorrelationDataResponse(
            symbol_a="NVDA",
            symbol_b="SPY",
            correlations={
                "20": [CorrelationPoint(time="2025-01-01", value=0.8)],
            },
            cumulative_returns={
                "NVDA": [CorrelationPoint(time="2025-01-01", value=100.0)],
                "SPY": [CorrelationPoint(time="2025-01-01", value=100.0)],
            },
            statistics={
                "pearson_r": 0.75,
                "pearson_p": 0.001,
                "spearman_r": 0.72,
                "spearman_p": 0.002,
                "kendall_tau": 0.65,
                "kendall_p": 0.003,
                "rolling_corr_std": 0.12,
                "oos_corr_drop": 0.05,
            },
        )
        assert resp.symbol_a == "NVDA"
        assert len(resp.correlations["20"]) == 1
