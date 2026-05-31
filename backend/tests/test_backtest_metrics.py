import pandas as pd
import pytest

from backend.backtest.metrics import calculate_metrics


def test_total_return_positive():
    equity = pd.DataFrame({"equity": [10000.0, 11000.0], "cash": [10000.0, 10000.0]})
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["total_return_pct"] == 10.0
    assert metrics["final_equity"] == 11000.0


def test_total_return_negative():
    equity = pd.DataFrame({"equity": [10000.0, 9000.0], "cash": [10000.0, 10000.0]})
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["total_return_pct"] == -10.0
    assert metrics["final_equity"] == 9000.0


def test_total_return_zero():
    equity = pd.DataFrame({"equity": [10000.0, 10000.0], "cash": [10000.0, 10000.0]})
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["total_return_pct"] == 0.0


def test_sharpe_ratio_with_flat_returns():
    equity = pd.DataFrame(
        {"equity": [10000.0] * 10, "cash": [10000.0] * 10}
    )
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["sharpe_ratio"] == 0.0


def test_sharpe_ratio_positive():
    equity = pd.DataFrame(
        {"equity": [10000.0, 10100.0, 10200.0, 10300.0, 10400.0],
         "cash": [10000.0] * 5}
    )
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["sharpe_ratio"] > 0


def test_max_drawdown():
    equity = pd.DataFrame(
        {"equity": [10000.0, 11000.0, 9000.0, 10500.0, 9500.0],
         "cash": [10000.0] * 5}
    )
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    # Peak at 11000, trough at 9000 → -18.18%
    assert metrics["max_drawdown_pct"] == pytest.approx(-18.18, abs=0.1)


def test_win_rate_with_mixed_trades():
    equity = pd.DataFrame({"equity": [10000.0, 10500.0], "cash": [10000.0, 10000.0]})
    trades = pd.DataFrame({
        "bar_index": [1, 3, 5, 7],
        "timestamp": ["2025-01-02", "2025-01-04", "2025-01-06", "2025-01-08"],
        "symbol": ["T", "T", "T", "T"],
        "side": ["sell", "sell", "sell", "sell"],
        "qty": [10, 10, 10, 10],
        "price": [101.0, 99.0, 102.0, 98.0],
        "pnl": [100.0, -100.0, 200.0, -200.0],
    })
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["win_rate_pct"] == 50.0
    assert metrics["num_trades"] == 4


def test_profit_factor():
    equity = pd.DataFrame({"equity": [10000.0, 10500.0], "cash": [10000.0, 10000.0]})
    trades = pd.DataFrame({
        "bar_index": [1, 3],
        "timestamp": ["2025-01-02", "2025-01-04"],
        "symbol": ["T", "T"],
        "side": ["sell", "sell"],
        "qty": [10, 10],
        "price": [102.0, 98.0],
        "pnl": [200.0, -200.0],
    })
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["profit_factor"] == 1.0


def test_profit_factor_all_wins():
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
    assert metrics["profit_factor"] == 999999.0


def test_no_trades():
    equity = pd.DataFrame({"equity": [10000.0], "cash": [10000.0]})
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["num_trades"] == 0
    assert metrics["win_rate_pct"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0


def test_single_bar():
    equity = pd.DataFrame({"equity": [10000.0], "cash": [10000.0]})
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=10000.0)
    assert metrics["total_return_pct"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0
