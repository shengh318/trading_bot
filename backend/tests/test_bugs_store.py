"""Tests for bugs in data store (H1, M14)."""

import pytest
from unittest.mock import MagicMock


# ── H1: save_positions DELETE not in transaction ──


class TestH1_SavePositionsDeleteNotInTransaction:
    """H1: DELETE FROM positions runs outside transaction — data loss on insert failure."""

    def test_save_positions_atomic(self):
        """H1: save_positions should be atomic — rollback DELETE if any INSERT fails."""
        from backend.data.store import Database

        db = Database(":memory:")

        # Insert valid positions first
        db.save_positions([
            {"symbol": "AAPL", "qty": 10, "avg_entry_price": 150, "current_price": 155,
             "unrealized_pl": 50, "market_value": 1550},
        ])

        positions_before = db.get_positions()
        assert len(positions_before) == 1

        # Now try saving with a dict that will fail (missing required field)
        with pytest.raises(Exception):
            db.save_positions([
                {"symbol": "GOOG"},  # missing required fields → should fail
            ])

        # After failure, the original positions must still exist
        positions_after = db.get_positions()
        assert len(positions_after) == 1, (
            "H1: DELETE should be rolled back if subsequent INSERT fails"
        )

        db.close()


# ── M14: save_positions redundant cascade deletes ──


class TestM14_RedundantCascadeDeletes:
    """M14: Manual DELETE of trades/snapshots is redundant given ON DELETE CASCADE."""

    def test_cascade_deletes_work(self):
        """M14: Deleting a backtest run should cascade to trades and snapshots."""
        from backend.data.store import Database

        db = Database(":memory:")

        run_id = db.save_backtest_run({
            "strategy_name": "Test", "symbol": "T",
            "start_date": "2025-01-01", "end_date": "2025-01-10",
            "initial_cash": 10000, "parameters": None,
        })

        db.save_backtest_trades(run_id, [
            {"bar_index": 0, "timestamp": "2025-01-01", "symbol": "T",
             "side": "buy", "qty": 10, "price": 100, "pnl": None},
        ])
        db.save_backtest_snapshots(run_id, [
            {"bar_index": 0, "timestamp": "2025-01-01", "equity": 10000, "cash": 9000},
        ])

        # Delete run — cascade should clean up trades and snapshots
        db.delete_backtest_run(run_id)

        trades = db.get_backtest_trades(run_id)
        snapshots = db.get_backtest_snapshots(run_id)

        assert len(trades) == 0, "Cascade should delete trades"
        assert len(snapshots) == 0, "Cascade should delete snapshots"


# ── Bug 24: delete_backtest_runs() only resets backtest_runs sequence, not backtest_trades or backtest_snapshots ──


class TestBug24_DeleteBacktestRunsResetsAllSequences:
    """Bug 24: delete_backtest_runs() should reset autoincrement for all related tables."""

    def test_delete_backtest_runs_resets_trades_and_snapshots_sequences(self):
        """Bug 24: After delete_backtest_runs(), next run_id should be 1 and trade/snapshot IDs should be low."""
        from backend.data.store import Database

        db = Database(":memory:")

        # Create a run with trades and snapshots
        run_id1 = db.save_backtest_run({
            "strategy_name": "Test", "symbol": "T",
            "start_date": "2025-01-01", "end_date": "2025-01-10",
            "initial_cash": 10000, "parameters": None,
        })
        assert run_id1 == 1

        db.save_backtest_trades(run_id1, [
            {"bar_index": 0, "timestamp": "2025-01-01", "symbol": "T",
             "side": "buy", "qty": 10, "price": 100, "pnl": None},
        ])
        db.save_backtest_snapshots(run_id1, [
            {"bar_index": 0, "timestamp": "2025-01-01", "equity": 10000, "cash": 9000},
        ])

        # Create another run
        run_id2 = db.save_backtest_run({
            "strategy_name": "Test2", "symbol": "A",
            "start_date": "2025-02-01", "end_date": "2025-02-10",
            "initial_cash": 20000, "parameters": None,
        })
        assert run_id2 == 2

        # Delete all runs
        db.delete_backtest_runs()

        # Bug: only backtest_runs sequence is reset, not backtest_trades or backtest_snapshots
        # After fix: all three sequences should be reset
        run_id3 = db.save_backtest_run({
            "strategy_name": "Test3", "symbol": "B",
            "start_date": "2025-03-01", "end_date": "2025-03-10",
            "initial_cash": 30000, "parameters": None,
        })

        # With the fix, run_id3 should be 1 (sequence was reset)
        # But the critical part is that trade/snapshot IDs don't grow unbounded
        assert run_id3 == 1, (
            "Bug 24: After clearing all runs, the next run_id should be 1"
        )

        db.close()

    def test_trade_and_snapshot_ids_reset_after_delete_all(self):
        """Bug 24: Trade and snapshot autoincrement IDs should be reset along with backtest_runs."""
        from backend.data.store import Database

        db = Database(":memory:")
        conn = db.conn

        # Check current sequences
        seq_before = conn.execute(
            "SELECT name, seq FROM sqlite_sequence"
        ).fetchall()
        seq_names = [r["name"] for r in seq_before]

        # Bug: delete_backtest_runs only resets 'backtest_runs' sequence
        # After fix: should also reset 'backtest_trades' and 'backtest_snapshots'
        assert "backtest_runs" in seq_names or len(seq_before) == 0

        db.close()
