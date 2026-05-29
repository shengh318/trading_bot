import numpy as np
import pandas as pd


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.0,
) -> dict:
    if equity_curve.empty or len(equity_curve) == 0:
        return {
            "total_return_pct": 0.0,
            "final_equity": initial_cash,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "num_trades": 0,
            "profit_factor": 0.0,
        }
    final_equity = equity_curve["equity"].iloc[-1]
    total_return_pct = ((final_equity - initial_cash) / initial_cash) * 100

    num_trades = len(trades) if not trades.empty else 0

    win_rate = 0.0
    profit_factor = 0.0
    if num_trades > 0 and not trades.empty:
        sells = trades[trades["side"] == "sell"].copy()
        winning = sells[sells["pnl"] > 0]
        losing = sells[sells["pnl"] < 0]
        win_rate = (len(winning) / len(sells)) * 100 if len(sells) > 0 else 0.0

        gross_profit = winning["pnl"].sum() if not winning.empty else 0.0
        gross_loss = abs(losing["pnl"].sum()) if not losing.empty else 0.0

        if gross_loss == 0:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

    sharpe_ratio = 0.0
    max_drawdown = 0.0

    if len(equity_curve) > 1:
        equity_curve = equity_curve.copy()
        equity_curve["return"] = equity_curve["equity"].pct_change().fillna(0)
        daily_returns = equity_curve["return"].values
        n_bars = len(daily_returns)
        annual_factor = np.sqrt(252 * n_bars / max(n_bars, 1))

        if daily_returns.std() > 0:
            excess_returns = daily_returns - risk_free_rate / annual_factor
            sharpe_ratio = float(
                (excess_returns.mean() / excess_returns.std()) * annual_factor
            )

        cumulative_max = equity_curve["equity"].cummax()
        drawdowns = (equity_curve["equity"] - cumulative_max) / cumulative_max * 100
        max_drawdown = float(round(drawdowns.min(), 2))

    return {
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final_equity, 2),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown_pct": max_drawdown,
        "win_rate_pct": round(win_rate, 2),
        "num_trades": num_trades,
        "profit_factor": profit_factor if profit_factor == float("inf") else round(profit_factor, 2),
    }
