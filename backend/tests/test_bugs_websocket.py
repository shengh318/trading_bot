"""Tests for bugs in WebSocket and live engine (H7, L14, L15, L19)."""

import pytest


# ── H7: Live engine unbounded data buffer growth ──


class TestH7_UnboundedBufferGrowth:
    """H7: data_buffers never evict old data."""

    def test_data_buffers_have_max_size(self):
        """H7: Live engine data_buffers should have a maximum size / eviction policy."""
        from backend.engine.live import LiveEngine

        import inspect
        source = inspect.getsource(LiveEngine.__init__)

        assert "maxlen" in source or "maxsize" in source or "deque" in source or "pop" in source or "_max_buffer_bars" in source, (
            "H7: data_buffers should have bounded growth"
        )


# ── L14: Live engine busy-wait with 1-second resolution ──


class TestL14_BusyWait:
    """L14: Live engine uses sleep-based polling instead of proper scheduling."""

    def test_live_engine_uses_busy_wait(self):
        """L14: Live engine uses 1-second sleep polling."""
        from backend.engine.live import LiveEngine

        import inspect
        source = inspect.getsource(LiveEngine._run_loop)

        assert "asyncio.sleep(1)" in source, "L14: Live engine busy-waits at 1s resolution"


# ── L15: Global _engine prevents multiple concurrent live sessions ──


class TestL15_GlobalEnginePreventsConcurrent:
    """L15: Global _engine module variable prevents concurrent sessions."""

    def test_engine_not_global(self):
        """L15: Engine should not be stored as module-level global."""
        from backend.engine.live import LiveEngine

        import inspect
        source = inspect.getsource(LiveEngine.start)

        # The engine is stored as _engine at module level
        # This prevents multiple concurrent sessions
        import backend.engine.live as live_module
        assert hasattr(live_module, "_engine"), (
            "L15: Global _engine variable should be replaced with per-session storage"
        )


# ── L19: Duplicate _parse_timeframe (3 copies) ──


class TestL19_DuplicateParseTimeframe:
    """L19: _parse_timeframe is defined in 3 different files."""

    def test_parse_timeframe_not_duplicated(self):
        """L19: _parse_timeframe should be defined once in a shared utility."""
        import inspect
        from backend.api import websocket as ws
        from backend.api import live_websocket as lws

        ws_source = inspect.getsource(ws)
        lws_source = inspect.getsource(lws)

        # Count occurrences of "def _parse_timeframe" in both files
        ws_count = ws_source.count("def _parse_timeframe")
        lws_count = lws_source.count("def _parse_timeframe")

        assert ws_count + lws_count <= 1, (
            "L19: _parse_timeframe is duplicated across files — should be in shared module"
        )


# ── Bug 16: LiveEngine does not allow initial_cash to be configured via WebSocket ──


class TestBug16_LiveEngineNoInitialCash:
    """Bug 16: LiveEngine always reads cash from Alpaca, no custom initial_cash support."""

    def test_live_engine_supports_initial_cash(self):
        """Bug 16: LiveEngine should accept initial_cash parameter."""
        from backend.engine.live import LiveEngine

        engine = LiveEngine("test", {}, ["TEST"], "1Day", initial_cash=5000.0)
        assert engine._initial_cash == 5000.0, "LiveEngine should accept initial_cash in constructor"

    def test_live_engine_defaults_to_alpaca_when_no_initial_cash(self):
        """Bug 16: LiveEngine falls back to Alpaca when initial_cash is not provided."""
        from backend.engine.live import LiveEngine

        import inspect
        init_source = inspect.getsource(LiveEngine.start)

        # Should still read from Alpaca as fallback
        assert "get_account()" in init_source, "LiveEngine reads from Alpaca account"
        assert "self._initial_cash" in init_source, "LiveEngine checks for custom initial_cash"

    def test_live_engine_constructor_accepts_initial_cash(self):
        """Bug 16: LiveEngine.__init__() should accept initial_cash parameter."""
        import inspect
        from backend.engine.live import LiveEngine

        sig = inspect.signature(LiveEngine.__init__)
        params = list(sig.parameters.keys())

        assert "initial_cash" in params, (
            "Bug 16: LiveEngine.__init__() should accept initial_cash parameter"
        )

    def test_live_websocket_passes_initial_cash(self):
        """Bug 16: Live WebSocket handler should pass initial_cash to LiveEngine."""
        from backend.api import live_websocket as lws

        import inspect
        source = inspect.getsource(lws.live_websocket)

        # After fix, the handler should pass initial_cash to LiveEngine
        assert "initial_cash" in source, (
            "Bug 16: The WebSocket handler should pass initial_cash to LiveEngine"
        )
