"""
Phase 7 — Backtest Engine.

Computes comprehensive performance metrics from an equity curve and trade list.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("stats_arb.backtest")


@dataclass
class BacktestResult:
    """Comprehensive backtest performance metrics.

    Attributes
    ----------
    total_return_pct : float
        Total return as a percentage.
    annualised_return_pct : float
        Annualised return percentage.
    sharpe_ratio : float
        Annualised Sharpe ratio.
    sortino_ratio : float
        Annualised Sortino ratio (downside deviation).
    calmar_ratio : float
        Return over max drawdown ratio.
    max_drawdown_pct : float
        Maximum drawdown as a percentage.
    win_rate_pct : float
        Percentage of winning trades.
    num_trades : int
        Total number of closed trades.
    avg_holding_period : float
        Average holding period in days.
    turnover : float
        Total turnover ratio.
    final_equity : float
        Final portfolio value.
    equity_curve : pd.Series
        Full equity curve.
    daily_returns : pd.Series
        Daily return series.
    drawdown_series : pd.Series
        Drawdown time series.
    avg_win_pct : float
        Average winning trade return.
    avg_loss_pct : float
        Average losing trade return.
    profit_factor : float
        Gross profit / gross loss.
    exposure_pct : float
        Percentage of time in the market.
    beta_to_market : float
        Beta relative to the market (0 if no market data).
    alpha : float
        Annualised excess return over benchmark.
    beta : float
        Strategy beta to benchmark.
    information_ratio : float
        Alpha / tracking error.
    tracking_error : float
        Standard deviation of excess returns (annualised).
    """

    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    avg_holding_period: float
    turnover: float
    final_equity: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    drawdown_series: pd.Series
    trades: list
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    exposure_pct: float
    beta_to_market: float
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        def _safe_round(v: float, ndigits: int) -> float:
            if np.isfinite(v):
                return round(v, ndigits)
            return v

        trades_dict = []
        for t in self.trades:
            if hasattr(t, "to_dict"):
                trades_dict.append(t.to_dict())
            elif isinstance(t, dict):
                trades_dict.append({k: _safe_round(v, 4) if isinstance(v, float) else v for k, v in t.items()})
        return {
            "total_return_pct": _safe_round(self.total_return_pct, 2),
            "annualised_return_pct": _safe_round(self.annualised_return_pct, 2),
            "sharpe_ratio": _safe_round(self.sharpe_ratio, 4),
            "sortino_ratio": _safe_round(self.sortino_ratio, 4),
            "calmar_ratio": _safe_round(self.calmar_ratio, 4),
            "max_drawdown_pct": _safe_round(self.max_drawdown_pct, 2),
            "win_rate_pct": _safe_round(self.win_rate_pct, 2),
            "num_trades": self.num_trades,
            "avg_holding_period": _safe_round(self.avg_holding_period, 2),
            "turnover": _safe_round(self.turnover, 4),
            "final_equity": _safe_round(self.final_equity, 2),
            "avg_win_pct": _safe_round(self.avg_win_pct, 2),
            "avg_loss_pct": _safe_round(self.avg_loss_pct, 2),
            "profit_factor": _safe_round(self.profit_factor, 4),
            "exposure_pct": _safe_round(self.exposure_pct, 2),
            "beta_to_market": _safe_round(self.beta_to_market, 4),
            "alpha": _safe_round(self.alpha, 4),
            "beta": _safe_round(self.beta, 4),
            "information_ratio": _safe_round(self.information_ratio, 4),
            "tracking_error": _safe_round(self.tracking_error, 4),
            "num_trades_detail": len(trades_dict),
        }


class BacktestEngine:
    """Computes performance metrics from strategy output.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time.
    trades : list of TradeRecord
        Completed trades.
    initial_capital : float
        Starting capital.
    annual_factor : int
        Number of trading days per year (default 252).
    market_returns : pd.Series, optional
        Market return series for beta calculation.
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: list,
        initial_capital: float,
        annual_factor: int = 252,
        market_returns: pd.Series | None = None,
    ) -> None:
        self.equity = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.annual_factor = annual_factor
        self.market_returns = market_returns

    @property
    def benchmark_returns(self) -> pd.Series | None:
        return self._benchmark_returns

    @benchmark_returns.setter
    def benchmark_returns(self, value: pd.Series | None) -> None:
        self._benchmark_returns = value

    def run(self) -> BacktestResult:
        if len(self.equity) < 2:
            return self._empty_result()

        initial = self.initial_capital
        final = float(self.equity.iloc[-1])

        if initial == 0:
            total_ret = 0.0
        else:
            total_ret = (final - initial) / initial * 100

        years = len(self.equity) / self.annual_factor
        if years > 0 and total_ret > -100 and np.isfinite(total_ret):
            try:
                annual_ret = ((1 + total_ret / 100) ** (1 / years) - 1) * 100
            except (OverflowError, ValueError):
                annual_ret = 1e100 if total_ret > 0 else -100.0
        else:
            annual_ret = total_ret / max(years, 0.01) if years > 0 else 0.0
        if not np.isfinite(annual_ret):
            annual_ret = 1e100 if total_ret > 0 else -100.0

        daily_returns = self.equity.pct_change().dropna()

        sharpe = self._compute_sharpe(daily_returns)
        sortino = self._compute_sortino(daily_returns)
        max_dd, dd_series = self._compute_drawdown(self.equity)
        calmar = abs(annual_ret / max_dd) if max_dd != 0 else 0.0

        num_closed = len(self.trades)
        wins = [t for t in self.trades if self._get_attr(t, "pnl", 0) > 0]
        losses = [t for t in self.trades if self._get_attr(t, "pnl", 0) <= 0]
        win_rate = (len(wins) / num_closed * 100) if num_closed > 0 else 0.0

        avg_hold = self._avg_holding_period()
        turnover = self._compute_turnover()

        avg_win = (
            float(np.mean([self._get_attr(t, "return_pct", 0) for t in wins]))
            if wins else 0.0
        )
        avg_loss = (
            float(np.mean([self._get_attr(t, "return_pct", 0) for t in losses]))
            if losses else 0.0
        )

        gross_profit = sum(
            max(0, self._get_attr(t, "pnl", 0)) for t in self.trades
        )
        gross_loss = abs(
            sum(
                min(0, self._get_attr(t, "pnl", 0)) for t in self.trades
            )
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999999.0 if gross_profit > 0 else 0.0)

        exposure = self._compute_exposure()
        beta = self._compute_beta(daily_returns)

        alpha = 0.0
        information_ratio = 0.0
        tracking_error = 0.0
        benchmark_beta = 0.0

        market_ret = self._compute_market_returns()
        if market_ret is not None and len(daily_returns) > 2:
            aligned = pd.concat([daily_returns, market_ret], axis=1).dropna()
            if len(aligned) > 2:
                strat_ret = aligned.iloc[:, 0]
                bench_ret = aligned.iloc[:, 1]
                excess = strat_ret - bench_ret
                alpha = float(excess.mean() * self.annual_factor)
                tracking_error = float(excess.std() * math.sqrt(self.annual_factor))
                information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
                cov = float(strat_ret.cov(bench_ret))
                var_b = float(bench_ret.var())
                benchmark_beta = cov / var_b if var_b > 0 else 0.0

        return BacktestResult(
            total_return_pct=total_ret,
            annualised_return_pct=annual_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            win_rate_pct=win_rate,
            num_trades=num_closed,
            avg_holding_period=avg_hold,
            turnover=turnover,
            final_equity=final,
            equity_curve=self.equity,
            daily_returns=daily_returns,
            drawdown_series=dd_series,
            trades=self.trades,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            exposure_pct=exposure,
            beta_to_market=beta,
            alpha=alpha,
            beta=benchmark_beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
        )

    def _compute_sharpe(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0 or returns.isnull().all():
            return 0.0
        std = float(returns.std())
        return 0.0 if std == 0 or not np.isfinite(std) else float(returns.mean() / std * math.sqrt(self.annual_factor))

    def _compute_sortino(self, returns: pd.Series) -> float:
        if len(returns) < 2 or returns.isnull().all():
            return 0.0
        mean_return = float(returns.mean())
        downside = returns.copy()
        downside[downside > 0] = 0.0
        downside_dev = np.sqrt(np.mean(downside ** 2))
        if downside_dev == 0 or not np.isfinite(downside_dev):
            return 0.0
        return float(mean_return / downside_dev * math.sqrt(self.annual_factor))

    @staticmethod
    def _compute_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax * 100
        return float(dd.min()), dd

    def _avg_holding_period(self) -> float:
        if not self.trades:
            return 0.0
        periods = [
            self._get_attr(t, "holding_period", 0) for t in self.trades
        ]
        return float(np.mean(periods))

    def _compute_turnover(self) -> float:
        if not self.trades:
            return 0.0
        total_traded = 0.0
        for t in self.trades:
            a_s = abs(self._get_attr(t, "shares_a", 0))
            b_s = abs(self._get_attr(t, "shares_b", 0))
            ep_a = self._get_attr(t, "entry_price_a", 0)
            ep_b = self._get_attr(t, "entry_price_b", 0)
            total_traded += a_s * ep_a + b_s * ep_b
        return float(total_traded / self.initial_capital / len(self.trades))

    def _compute_exposure(self) -> float:
        if not self.trades:
            return 0.0
        total_days = len(self.equity)
        trading_days = sum(
            self._get_attr(t, "holding_period", 0) for t in self.trades
        )
        return min(100.0, float(trading_days / total_days * 100))

    def _compute_market_returns(self) -> pd.Series | None:
        if self.market_returns is None:
            return None
        mr = self.market_returns.dropna()
        if len(mr) < 2:
            return None
        if mr.iloc[0] > 1 and mr.iloc[0] < 1e6:
            mr = mr.pct_change().dropna()
        return mr

    def _compute_beta(self, strategy_returns: pd.Series) -> float:
        market_ret = self._compute_market_returns()
        if market_ret is None:
            return 0.0
        aligned = pd.concat(
            [strategy_returns, market_ret], axis=1
        ).dropna()
        if len(aligned) < 3:
            return 0.0
        cov = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]))
        var = float(aligned.iloc[:, 1].var())
        return cov / var if var > 0 else 0.0

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from an object or dict-like."""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default

    def _empty_result(self) -> BacktestResult:
        empty_eq = pd.Series([self.initial_capital])
        return BacktestResult(
            total_return_pct=0.0,
            annualised_return_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            num_trades=0,
            avg_holding_period=0.0,
            turnover=0.0,
            final_equity=self.initial_capital,
            equity_curve=empty_eq,
            daily_returns=pd.Series(dtype=float),
            drawdown_series=pd.Series(dtype=float),
            trades=[],
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            profit_factor=0.0,
            exposure_pct=0.0,
            beta_to_market=0.0,
            alpha=0.0,
            beta=0.0,
            information_ratio=0.0,
            tracking_error=0.0,
        )
