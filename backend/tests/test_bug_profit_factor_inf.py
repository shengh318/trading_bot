import json

import pandas as pd
import pytest

from backend.backtest.metrics import calculate_metrics


class TestProfitFactorSerialization:
    def test_profit_factor_all_wins_serializable(self):
        """profit_factor must be JSON-serializable (no float('inf'))."""
        equity = pd.DataFrame({"equity": [10000.0, 10500.0], "cash": [10000.0, 10000.0]})
        trades = pd.DataFrame({
            "bar_index": [1, 3],
            "timestamp": ["2025-01-02", "2025-01-04"],
            "symbol": ["T", "T"],
            "side": ["sell", "sell"],
            "qty": [10, 10],
            "price": [102.0, 103.0],
            "pnl": [200.0, 300.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        # This must not raise TypeError/ValueError
        dumped = json.dumps(metrics)
        loaded = json.loads(dumped)
        assert loaded["profit_factor"] == pytest.approx(999999.0, abs=1.0) or loaded["profit_factor"] is None

    def test_profit_factor_zero_trades(self):
        """profit_factor with no trades must be 0.0 and serializable."""
        equity = pd.DataFrame({"equity": [10000.0], "cash": [10000.0]})
        trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        dumped = json.dumps(metrics)
        loaded = json.loads(dumped)
        assert loaded["profit_factor"] == 0.0

    def test_profit_factor_all_losses(self):
        """profit_factor when all trades lose must be 0.0."""
        equity = pd.DataFrame({"equity": [10000.0, 9500.0], "cash": [10000.0, 10000.0]})
        trades = pd.DataFrame({
            "bar_index": [1, 3],
            "timestamp": ["2025-01-02", "2025-01-04"],
            "symbol": ["T", "T"],
            "side": ["sell", "sell"],
            "qty": [10, 10],
            "price": [98.0, 97.0],
            "pnl": [-200.0, -300.0],
        })
        metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
        dumped = json.dumps(metrics)
        loaded = json.loads(dumped)
        assert loaded["profit_factor"] == 0.0
