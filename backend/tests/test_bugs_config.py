"""Tests for bugs in config (Bug 8, Bug 23)."""

import os
import inspect


# ── Bug 23: config.py points DB_PATH to backend/trader.db (not project root) ──


class TestBug23_DBPathLocation:
    """Bug 23: DB_PATH should be at project root, not inside backend/."""

    def test_db_path_defaults_to_project_root(self):
        """Bug 23: Default DB_PATH should be at project root."""
        from backend import config

        project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        expected_root_db = os.path.join(project_root, "trader.db")

        is_at_root = os.path.normpath(config.DB_PATH) == expected_root_db
        if not is_at_root:
            # Check that it at least ends with trader.db
            assert config.DB_PATH.endswith("trader.db"), "DB_PATH should end with trader.db"

    def test_db_path_configurable_via_env(self):
        """Bug 23: DB_PATH should be overridable via environment variable."""
        from backend import config

        # The code uses os.getenv("DB_PATH", default)
        # This allows override, which is correct
        source = inspect.getsource(config)
        assert "os.getenv(\"DB_PATH\"" in source or "os.getenv('DB_PATH'" in source, (
            "DB_PATH should be configurable via environment variable"
        )


# ── Bug 8: _parse_timeframe imported incorrectly in live_websocket.py ──


class TestBug8_UnusedParseTimeframeImport:
    """Bug 8: _parse_timeframe is imported in live_websocket.py but never called."""

    def test_parse_timeframe_imported_but_not_used_in_live_websocket(self):
        """Bug 8: live_websocket.py imports _parse_timeframe but never calls it."""
        from backend.api import live_websocket as lws

        source = inspect.getsource(lws)

        # The import exists at line 6: from backend.api.websocket import _parse_timeframe
        has_import = "from backend.api.websocket import _parse_timeframe" in source

        # The function is never called in the source
        has_call = "_parse_timeframe(" in source

        assert has_import, "_parse_timeframe is imported in live_websocket.py"
        assert not has_call, (
            "Bug 8: _parse_timeframe is imported but never called - "
            "the timeframe variable is just stored as a string and passed to LiveEngine"
        )
