"""Bug: calculate_metrics returns inf for total_return_pct when initial_cash=0.

Line 22 of metrics.py: `total_return_pct = ((final_equity - initial_cash) / initial_cash) * 100`
When initial_cash is 0, this produces inf or NaN instead of a graceful fallback.
"""

import pandas as pd

from backend.backtest.metrics import calculate_metrics


def test_zero_initial_cash_returns_sensible_metrics():
    """With initial_cash=0, metrics should not contain inf or NaN."""
    equity = pd.DataFrame({
        "equity": [0.0, 100.0],
        "cash": [0.0, 100.0],
    })
    trades = pd.DataFrame(columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"])
    metrics = calculate_metrics(equity, trades, initial_cash=0.0)

    # BUG: total_return_pct is inf when initial_cash=0
    # Expected: should return 0.0 or handle gracefully
    assert metrics["total_return_pct"] != float("inf"), (
        "total_return_pct should not be inf when initial_cash=0"
    )
    assert not any(
        v == float("inf") or v != v for v in metrics.values() if isinstance(v, (int, float))
    ), f"Metrics contain inf or NaN: {metrics}"


def test_zero_initial_cash_with_trades():
    """With trades and zero initial cash, profit_factor/wins should be sensible."""
    equity = pd.DataFrame({
        "equity": [0.0, 200.0],
        "cash": [0.0, 100.0],
    })
    trades = pd.DataFrame({
        "bar_index": [1],
        "timestamp": ["2025-01-02"],
        "symbol": ["T"],
        "side": ["sell"],
        "qty": [10],
        "price": [100.0],
        "pnl": [100.0],
    })
    metrics = calculate_metrics(equity, trades, initial_cash=0.0)

    assert isinstance(metrics["total_return_pct"], float), (
        "total_return_pct must be a finite float even with zero initial cash"
    )
    assert metrics["total_return_pct"] != float("inf"), (
        "total_return_pct should not be inf"
    )
