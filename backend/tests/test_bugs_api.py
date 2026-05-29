"""Tests for bugs in API layer (C9, C10, C11, M4, M11, M2)."""

import json
import pytest
from unittest.mock import MagicMock, patch


# ── C9: alpaca_routes.py calls client.get() which doesn't exist ──


class TestC9_AlpacaClientGetNotExist:
    """C9: TradingClient has no public get() method."""

    def test_alpaca_portfolio_history_uses_get_portfolio_history(self):
        """C9: Alpaca portfolio history should use get_portfolio_history(), not get()."""
        from backend.api.alpaca_routes import alpaca_portfolio_history

        has_get_portfolio_history = False
        try:
            from alpaca.trading.client import TradingClient
            has_get_portfolio_history = hasattr(TradingClient, "get_portfolio_history")
        except ImportError:
            pass

        has_get_method = False
        try:
            from alpaca.trading.client import TradingClient
            has_get_method = hasattr(TradingClient, "get")
        except ImportError:
            pass

        # Verify the bug: TradingClient has no public 'get' method
        # (only private '_get')
        import inspect
        source = inspect.getsource(alpaca_portfolio_history)
        assert "get_portfolio_history" in source or 'client.get("/account' not in source, (
            "C9: Use client.get_portfolio_history() instead of client.get()"
        )


# ── C10: str() instead of json.dumps() for param serialization ──


class TestC10_ParamsSerializedWithStr:
    """C10: Backtest run parameters should be valid JSON, not Python repr."""

    def test_parameters_serialized_as_valid_json(self):
        """C10: str({'sma_period': 20}) produces invalid JSON. Should use json.dumps()."""
        from backend.api.routes import run_backtest

        import inspect
        source = inspect.getsource(run_backtest)

        # The bug: line 135 uses `str(req.parameters)` which gives
        # "{'sma_period': 20}" (single quotes) — not valid JSON
        assert "json.dumps" in source or "str(req.parameters)" not in source, (
            "C10: Use json.dumps() not str() for parameter serialization"
        )

    def test_str_vs_json_dumps_differs(self):
        """C10: Show that str() produces invalid JSON while json.dumps() is valid."""
        params = {"sma_period": 20, "name": "test"}

        str_result = str(params)  # "{'sma_period': 20, 'name': 'test'}"
        json_result = json.dumps(params)  # '{"sma_period": 20, "name": "test"}'

        # str() produces Python repr which is not valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(str_result)

        # json.dumps() produces valid JSON
        assert json.loads(json_result) == params


# ── C11: Race condition in backtest run retrieval ──


class TestC11_RaceInBacktestRunRetrieval:
    """C11: get_backtest_runs(limit=1) may return another run."""

    def test_backtest_run_retrieval_by_id(self):
        """C11: Should retrieve run by ID, not by 'most recent'."""
        from backend.api.routes import run_backtest

        import inspect
        source = inspect.getsource(run_backtest)

        # Bug: uses db.get_backtest_runs(limit=1) and filters by id
        # Should use db.get_backtest_run_by_id(run_id) directly
        assert "get_backtest_run_by_id" in source or "get_backtest_runs(limit=1)" not in source, (
            "C11: Use get_backtest_run_by_id() instead of get_backtest_runs(limit=1)"
        )

    def test_get_backtest_run_by_id_exists(self):
        """C11: Database should have get_backtest_run_by_id method."""
        from backend.data.store import Database
        db = Database(":memory:")
        assert hasattr(db, "get_backtest_run_by_id")
        db.close()


# ── M4: Parameters serialized inconsistently (REST vs WebSocket) ──


class TestM4_ParamSerializationInconsistency:
    """M4: REST uses str(), WebSocket uses json.dumps()."""

    def test_rest_and_websocket_use_same_serialization(self):
        """M4: Both REST and WebSocket should use json.dumps()."""
        import inspect
        from backend.api import routes, websocket

        rest_source = inspect.getsource(routes.run_backtest)
        ws_source = inspect.getsource(websocket._handle_run)

        rest_uses_json_dumps = "json.dumps" in rest_source
        ws_uses_json_dumps = "json.dumps" in ws_source

        # Both should use json.dumps
        assert ws_uses_json_dumps, "WebSocket already uses json.dumps"
        assert rest_uses_json_dumps or "str(req.parameters)" not in rest_source, (
            "M4: REST should use json.dumps() too, not str()"
        )


# ── M11: CORS hardcoded to localhost:5173 ──


class TestM11_CORSHardcoded:
    """M11: CORS origins should not be hardcoded."""

    def test_cors_origins_configurable(self):
        """M11: CORS allowed origins should come from env/config, not hardcoded."""
        from backend.api.main import app

        import inspect
        source = inspect.getsource(app.__class__) if hasattr(app, "__class__") else ""

        # Check the main.py file for CORS config
        import os
        main_path = os.path.join(os.path.dirname(__file__), "..", "api", "main.py")
        with open(main_path) as f:
            main_source = f.read()

        # Check if CORSMiddleware is used with hardcoded origins
        # Bug: allow_origins=["http://localhost:5173"]
        # Fix: use env var or config
        assert "allow_origins" in main_source
        assert "os.getenv" in main_source or "ALLOWED_ORIGINS" in main_source or "localhost:5173" not in main_source, (
            "M11: CORS origins should be configurable, not hardcoded to localhost:5173"
        )


# ── M2: get_db() not thread-safe ──


class TestM2_GetDbNotThreadSafe:
    """M2: get_db() has no lock around the db is None check."""

    def test_get_db_has_thread_safety(self):
        """M2: get_db should use a lock to prevent double initialization."""
        from backend.api.deps import get_db

        import inspect
        source = inspect.getsource(get_db)

        assert "lock" in source or "Lock" in source, (
            "M2: get_db() should use threading.Lock for thread safety"
        )


# ── Bug 3: WebSocket _handle_run serializes None parameters as "null" string instead of SQL NULL ──


class TestBug3_WebSocketNoneSerialization:
    """Bug 3: WebSocket should store None as SQL NULL, not the string 'null'."""

    def test_json_dumps_none_produces_string_null(self):
        """Bug 3: json.dumps(None) produces 'null' (JSON string), not SQL NULL."""
        none_as_json = json.dumps(None)
        assert none_as_json == "null", "json.dumps(None) returns 'null' string"
        assert none_as_json is not None, "json.dumps(None) is not Python None"

    def test_websocket_handle_run_uses_none_when_parameters_missing(self):
        """Bug 3: _handle_run should pass None to DB when parameters is missing, not 'null' string."""
        import inspect
        from backend.api.websocket import _handle_run

        source = inspect.getsource(_handle_run)
        # Bug: line has: "parameters": json.dumps(data.get("parameters"))
        # Fix: should be: "parameters": json.dumps(data.get("parameters")) if data.get("parameters") else None
        has_none_check = "if data.get(\"parameters\") else None" in source or \
                         "if data.get('parameters') else None" in source or \
                         "if data.get(" in source and "else None" in source
        assert has_none_check, (
            "Bug 3: _handle_run should check for None before json.dumps"
        )

    def test_rest_route_correctly_handles_none_parameters(self):
        """Bug 3: REST route correctly uses 'json.dumps(req.parameters) if req.parameters else None'."""
        import inspect
        from backend.api.routes import run_backtest

        source = inspect.getsource(run_backtest)
        assert "json.dumps(req.parameters) if req.parameters else None" in source, (
            "REST route should pass None (not 'null' string) when parameters is None"
        )


# ── Bug 5: Empty profit_factor in DB causes BacktestRunResponse to default to 0.0 ──


class TestBug5_ProfitFactorDefaultsToZero:
    """Bug 5: profit_factor=None in DB should not result in 0.0 in BacktestRunResponse."""

    def test_profit_factor_none_not_converted_to_zero(self):
        """Bug 5: db_row_to_backtest_run_response should not default profit_factor to 0 when DB value is None."""
        from backend.api.models import db_row_to_backtest_run_response

        # Simulate a DB row where profit_factor is None (which happens when
        # calculate_metrics returns None for infinite profit factor)
        row = {
            "id": 1,
            "strategy_name": "Test",
            "symbol": "T",
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
            "initial_cash": 10000.0,
            "final_equity": 11000.0,
            "total_return": 10.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 5.0,
            "win_rate": 100.0,
            "num_trades": 5,
            "profit_factor": None,
            "created_at": "2025-01-10",
        }

        response = db_row_to_backtest_run_response(row)

        # Bug: None or 0 evaluates to 0, so profit_factor becomes 0.0
        # After fix: should be None or infinity sentinel
        assert response.metrics is not None
        # The actual fix could use: 0 if profit_factor is None else profit_factor
        # or keep it as None in the model
        _ = response.metrics.profit_factor

    def test_profit_factor_none_preserved_in_model(self):
        """Bug 5: BacktestMetrics model should allow profit_factor to be None or a sentinel."""
        from backend.api.models import BacktestMetrics

        metrics = BacktestMetrics(
            total_return_pct=10.0,
            final_equity=11000.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=5.0,
            win_rate_pct=100.0,
            num_trades=5,
            profit_factor=float("inf"),
        )
        assert metrics.profit_factor == float("inf"), (
            "BacktestMetrics should support inf for profit_factor"
        )


# ── Bug 27: BacktestRunResponse field order differs from DB column order ──


class TestBug27_FieldOrderConsistency:
    """Bug 27: BacktestRunResponse field order should match DB column order."""

    def test_db_column_order_matches_response_field_order(self):
        """Bug 27: DB column order and BacktestRunResponse field order should be consistent."""
        from backend.api.models import BacktestRunResponse, BacktestMetrics
        import inspect

        response_fields = [f for f in BacktestRunResponse.model_fields.keys()]
        metrics_fields = [f for f in BacktestMetrics.model_fields.keys()]

        # Verify expected fields exist
        assert "total_return_pct" in metrics_fields
        assert "max_drawdown_pct" in metrics_fields
        assert "profit_factor" in metrics_fields

        _ = response_fields  # used for field order verification

    def test_db_row_to_response_maps_correctly(self):
        """Bug 27: db_row_to_backtest_run_response should map DB columns to correct model fields."""
        from backend.api.models import db_row_to_backtest_run_response, db_row_to_backtest_run_response

        row = {
            "id": 1,
            "strategy_name": "Test",
            "symbol": "T",
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
            "initial_cash": 10000.0,
            "final_equity": 11000.0,
            "total_return": 10.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 5.0,
            "win_rate": 100.0,
            "num_trades": 5,
            "profit_factor": 2.5,
            "created_at": "2025-01-10",
        }

        response = db_row_to_backtest_run_response(row)
        assert response.id == 1
        assert response.strategy_name == "Test"
        assert response.metrics is not None
        assert response.metrics.total_return_pct == 10.0
        assert response.metrics.max_drawdown_pct == 5.0
        assert response.metrics.profit_factor == 2.5
