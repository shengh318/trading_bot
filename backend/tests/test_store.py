import pytest


class TestDatabaseInit:
    def test_creates_all_tables(self, db):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in tables]
        assert "portfolio_snapshots" in names
        assert "positions" in names
        assert "orders" in names
        assert "backtest_runs" in names
        assert "backtest_trades" in names
        assert "backtest_snapshots" in names

    def test_empty_db_returns_empty_lists(self, db):
        assert db.get_equity_curve() == []
        assert db.get_positions() == []
        assert db.get_orders() == []
        assert db.get_backtest_runs() == []


class TestPortfolioSnapshots:
    def test_save_and_retrieve(self, db):
        db.save_snapshot("2026-01-01T00:00:00", 10000.0, 5000.0, 15000.0)
        db.save_snapshot("2026-01-02T00:00:00", 10500.0, 4500.0, 16000.0)

        curve = db.get_equity_curve()
        assert len(curve) == 2
        assert curve[0]["total_equity"] == 10000.0
        assert curve[1]["total_equity"] == 10500.0

    def test_limit(self, db):
        for i in range(10):
            db.save_snapshot(f"2026-01-{i+1:02d}T00:00:00", 10000.0 + i, 5000.0, 15000.0)

        curve = db.get_equity_curve(limit=3)
        assert len(curve) == 3

    def test_returns_correct_fields(self, db):
        db.save_snapshot("2026-01-01T00:00:00", 10000.0, 5000.0, 15000.0)
        row = db.get_equity_curve()[0]
        assert set(row.keys()) == {"timestamp", "total_equity", "cash"}


class TestPositions:
    def test_save_and_retrieve(self, db):
        positions = [
            {"symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0,
             "current_price": 155.0, "unrealized_pl": 50.0, "market_value": 1550.0},
            {"symbol": "MSFT", "qty": 5, "avg_entry_price": 400.0,
             "current_price": 420.0, "unrealized_pl": 100.0, "market_value": 2100.0},
        ]
        db.save_positions(positions)
        result = db.get_positions()
        assert len(result) == 2
        assert result[0]["symbol"] == "MSFT"  # highest market_value first
        assert result[1]["symbol"] == "AAPL"

    def test_save_replaces_old(self, db):
        db.save_positions([{"symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0,
                            "current_price": 155.0, "unrealized_pl": 50.0, "market_value": 1550.0}])
        db.save_positions([])
        assert db.get_positions() == []


class TestOrders:
    def test_save_and_retrieve(self, db):
        order = {
            "id": "ord_1", "symbol": "AAPL", "side": "buy", "qty": 10,
            "filled_qty": 10, "filled_avg_price": 150.0, "status": "filled",
            "type": "market", "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
        }
        db.save_order(order)
        orders = db.get_orders()
        assert len(orders) == 1
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["status"] == "filled"

    def test_save_replaces_existing(self, db):
        order = {
            "id": "ord_1", "symbol": "AAPL", "side": "buy", "qty": 10,
            "filled_qty": 5, "filled_avg_price": None, "status": "partial",
            "type": "market", "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
        }
        db.save_order(order)
        order["filled_qty"] = 10
        order["status"] = "filled"
        order["filled_avg_price"] = 150.0
        db.save_order(order)
        orders = db.get_orders()
        assert len(orders) == 1
        assert orders[0]["status"] == "filled"


class TestBacktestRuns:
    def test_save_and_retrieve(self, db):
        run = {
            "strategy_name": "SmaCrossover", "parameters": '{"short": 20, "long": 50}',
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0, "final_equity": 11000.0, "total_return": 10.0,
            "sharpe_ratio": 1.5, "max_drawdown": -5.0, "win_rate": 55.0, "num_trades": 20,
        }
        run_id = db.save_backtest_run(run)
        assert run_id == 1

        runs = db.get_backtest_runs()
        assert len(runs) == 1
        assert runs[0]["strategy_name"] == "SmaCrossover"
        assert runs[0]["final_equity"] == 11000.0

    def test_save_backtest_trades(self, db):
        run = {
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0,
        }
        run_id = db.save_backtest_run(run)
        trades = [
            {"bar_index": 10, "timestamp": "2025-01-15", "symbol": "AAPL",
             "side": "buy", "qty": 10, "price": 150.0, "pnl": None},
            {"bar_index": 50, "timestamp": "2025-02-15", "symbol": "AAPL",
             "side": "sell", "qty": 10, "price": 160.0, "pnl": 100.0},
        ]
        db.save_backtest_trades(run_id, trades)
        result = db.get_backtest_trades(run_id)
        assert len(result) == 2
        assert result[0]["side"] == "buy"

    def test_save_backtest_snapshots(self, db):
        run = {
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0,
        }
        run_id = db.save_backtest_run(run)
        snapshots = [
            {"bar_index": 0, "timestamp": "2025-01-01", "equity": 10000.0, "cash": 10000.0},
            {"bar_index": 1, "timestamp": "2025-01-02", "equity": 10100.0, "cash": 9000.0},
        ]
        db.save_backtest_snapshots(run_id, snapshots)
        result = db.get_backtest_snapshots(run_id)
        assert len(result) == 2
        assert result[0]["equity"] == 10000.0


class TestEdgeCases:
    def test_duplicate_backtest_runs(self, db):
        run = {
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0,
        }
        id1 = db.save_backtest_run(run)
        id2 = db.save_backtest_run(run)
        assert id2 == id1 + 1
        assert len(db.get_backtest_runs()) == 2

    def test_cascade_delete_not_enforced_by_default_sqlite(self, db):
        run = {
            "strategy_name": "SmaCrossover", "parameters": None,
            "symbol": "AAPL", "start_date": "2025-01-01", "end_date": "2025-06-01",
            "initial_cash": 10000.0,
        }
        run_id = db.save_backtest_run(run)
        db.save_backtest_trades(run_id, [
            {"bar_index": 0, "timestamp": "2025-01-01", "symbol": "AAPL",
             "side": "buy", "qty": 1, "price": 100.0, "pnl": None},
        ])
        db.conn.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
        db.conn.commit()
        assert db.get_backtest_trades(run_id) == []
