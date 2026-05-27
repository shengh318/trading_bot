import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.config import DB_PATH


class Database:
    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_equity REAL NOT NULL,
                cash REAL NOT NULL,
                buying_power REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                qty REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pl REAL NOT NULL,
                market_value REAL NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                filled_qty REAL DEFAULT 0,
                filled_avg_price REAL,
                status TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                parameters TEXT,
                symbol TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                initial_cash REAL NOT NULL,
                final_equity REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                num_trades INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                bar_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS backtest_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                bar_index INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL,
                cash REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
            );
        """)
        self.conn.commit()

    # ── Portfolio Snapshots ──────────────────────────────────

    def save_snapshot(self, timestamp: str, total_equity: float, cash: float, buying_power: float) -> None:
        self.conn.execute(
            "INSERT INTO portfolio_snapshots (timestamp, total_equity, cash, buying_power) VALUES (?, ?, ?, ?)",
            (timestamp, total_equity, cash, buying_power),
        )
        self.conn.commit()

    def get_equity_curve(self, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT timestamp, total_equity, cash FROM portfolio_snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Positions ────────────────────────────────────────────

    def save_positions(self, positions: list[dict]) -> None:
        self.conn.execute("DELETE FROM positions")
        for p in positions:
            self.conn.execute(
                """INSERT INTO positions (symbol, qty, avg_entry_price, current_price,
                   unrealized_pl, market_value, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (p["symbol"], p["qty"], p["avg_entry_price"], p["current_price"],
                 p["unrealized_pl"], p["market_value"], datetime.utcnow().isoformat()),
            )
        self.conn.commit()

    def get_positions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM positions ORDER BY market_value DESC").fetchall()
        return [dict(r) for r in rows]

    # ── Orders ───────────────────────────────────────────────

    def save_order(self, order: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO orders
               (id, symbol, side, qty, filled_qty, filled_avg_price, status, type, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order["id"], order["symbol"], order["side"], order["qty"],
             order.get("filled_qty", 0), order.get("filled_avg_price"),
             order["status"], order["type"], order["created_at"], order["updated_at"]),
        )
        self.conn.commit()

    def get_orders(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Backtest Runs ────────────────────────────────────────

    def save_backtest_run(self, run: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO backtest_runs
               (strategy_name, parameters, symbol, start_date, end_date, initial_cash,
                final_equity, total_return, sharpe_ratio, max_drawdown, win_rate, num_trades)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run["strategy_name"], run.get("parameters"), run["symbol"],
             run["start_date"], run["end_date"], run["initial_cash"],
             run.get("final_equity"), run.get("total_return"),
             run.get("sharpe_ratio"), run.get("max_drawdown"),
             run.get("win_rate"), run.get("num_trades")),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_backtest_runs(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_backtest_trades(self, run_id: int, trades: list[dict]) -> None:
        self.conn.executemany(
            """INSERT INTO backtest_trades (run_id, bar_index, timestamp, symbol, side, qty, price, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(run_id, t["bar_index"], t["timestamp"], t["symbol"],
              t["side"], t["qty"], t["price"], t.get("pnl")) for t in trades],
        )
        self.conn.commit()

    def get_backtest_trades(self, run_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY bar_index", (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_backtest_snapshots(self, run_id: int, snapshots: list[dict]) -> None:
        self.conn.executemany(
            """INSERT INTO backtest_snapshots (run_id, bar_index, timestamp, equity, cash)
               VALUES (?, ?, ?, ?, ?)""",
            [(run_id, s["bar_index"], s["timestamp"], s["equity"], s["cash"]) for s in snapshots],
        )
        self.conn.commit()

    def get_backtest_snapshots(self, run_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM backtest_snapshots WHERE run_id = ? ORDER BY bar_index", (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
