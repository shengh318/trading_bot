"""
Phase 6 — Pairs Trading Strategy.

Implements spread trading logic with configurable entry/exit rules,
position sizing, and risk controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_Z_ENTRY,
    DEFAULT_Z_EXIT,
    DEFAULT_STOP_LOSS,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_SLIPPAGE,
    DEFAULT_BORROW_FEE,
    DEFAULT_MAX_HOLDING_DAYS,
)

logger = logging.getLogger("stats_arb.strategy")


class PositionSide(str, Enum):
    LONG_SPREAD = "long_spread"
    SHORT_SPREAD = "short_spread"
    FLAT = "flat"


@dataclass
class TradeRecord:
    """Record of a single completed trade.

    Attributes
    ----------
    direction : str
        'long_spread' or 'short_spread'.
    entry_date : str
        Entry date string.
    exit_date : str
        Exit date string.
    entry_zscore : float
        Z-score at entry.
    exit_zscore : float
        Z-score at exit.
    entry_price_a : float
        Price of asset A at entry.
    entry_price_b : float
        Price of asset B at entry.
    exit_price_a : float
        Price of asset A at exit.
    exit_price_b : float
        Price of asset B at exit.
    shares_a : float
        Number of shares of A.
    shares_b : float
        Number of shares of B.
    return_pct : float
        Trade return percentage.
    pnl : float
        Trade P&L in dollars.
    holding_period : int
        Holding period in days.
    exit_reason : str
        Reason for exit (signal/stop_loss/max_hold/break).
    """

    direction: str
    entry_date: str
    exit_date: str
    entry_zscore: float
    exit_zscore: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    shares_a: float
    shares_b: float
    return_pct: float
    pnl: float
    holding_period: int
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_zscore": round(self.entry_zscore, 2),
            "exit_zscore": round(self.exit_zscore, 2),
            "return_pct": round(self.return_pct, 2),
            "pnl": round(self.pnl, 2),
            "holding_period": self.holding_period,
            "exit_reason": self.exit_reason,
        }


class TradingStrategy:
    """Spread mean-reversion trading strategy.

    Entry rules:
        - Long spread (buy A, sell B): z-score < -z_entry
        - Short spread (sell A, buy B): z-score > +z_entry

    Exit rules:
        - Z-score crosses z_exit (default 0)
        - Stop loss: z-score exceeds z_entry * stop_loss_multiple
        - Max holding time exceeded
        - Structural break detected

    Position sizing:
        - Dollar-neutral by default
        - Volatility-adjusted (optional)
        - Beta-neutral (optional)

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with columns [ticker_a, ticker_b].
    hedge_ratio : float
        The hedge ratio (beta) for the pair.
    z_entry : float
        Z-score threshold for entry.
    z_exit : float
        Z-score threshold for exit (cross this to close).
    stop_loss : float
        Stop loss as multiple of z_entry.
    transaction_cost : float
        Transaction cost as fraction per side.
    slippage : float
        Slippage as fraction per side.
    borrow_fee : float
        Daily borrow fee for short leg.
    max_holding_days : int
        Maximum holding period in days.
    initial_capital : float
        Starting capital.
    volatility_adjust : bool
        Whether to volatility-adjust position sizes.
        beta_neutral : bool
        Whether to target beta neutrality.
    market_beta : float
        Beta of the pair to the market (for beta neutrality).
    structural_break_detected : bool
        Force exit all positions if a structural break is flagged.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        hedge_ratio: float,
        z_entry: float = DEFAULT_Z_ENTRY,
        z_exit: float = DEFAULT_Z_EXIT,
        stop_loss: float = DEFAULT_STOP_LOSS,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        slippage: float = DEFAULT_SLIPPAGE,
        borrow_fee: float = DEFAULT_BORROW_FEE,
        max_holding_days: int = DEFAULT_MAX_HOLDING_DAYS,
        initial_capital: float = 100_000.0,
        volatility_adjust: bool = False,
        beta_neutral: bool = False,
        market_beta: float = 1.0,
        structural_break_detected: bool = False,
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.hedge_ratio = hedge_ratio
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss = stop_loss
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.borrow_fee = borrow_fee
        self.max_holding_days = max_holding_days
        self.initial_capital = initial_capital
        self.volatility_adjust = volatility_adjust
        self.beta_neutral = beta_neutral
        self.market_beta = market_beta
        self.structural_break_detected = structural_break_detected

    def execute(self) -> tuple[pd.Series, list[TradeRecord]]:
        """Run the strategy and return (equity_curve, trades)."""
        spread = self.prices[self.cols[0]] - self.hedge_ratio * self.prices[self.cols[1]]
        expanding_mean = spread.expanding().mean()
        expanding_std = spread.expanding().std()
        zscore = (spread - expanding_mean) / expanding_std

        if self.volatility_adjust:
            rolling_vol = spread.rolling(63).std().fillna(expanding_std)
            final_std = float(expanding_std.iloc[-1]) if not expanding_std.empty else 1.0
            vol_scalar = final_std / rolling_vol
        else:
            vol_scalar = pd.Series(1.0, index=spread.index)

        position = PositionSide.FLAT
        cash = float(self.initial_capital)
        a_shares = 0.0
        b_shares = 0.0

        capital_per_leg = self.initial_capital / 2.0

        equity_curve: list[float] = [self.initial_capital]
        trades: list[TradeRecord] = []
        current_trade: Optional[dict] = None
        trade_days = 0

        prices_arr = self.prices.values
        z_arr = zscore.values
        vol_arr = vol_scalar.values

        for i in range(1, len(self.prices)):
            z = z_arr[i]
            a_price = float(prices_arr[i, 0])
            b_price = float(prices_arr[i, 1])
            scale = float(vol_arr[i])

            should_exit = self._check_exit(position, z, trade_days)

            if should_exit and position != PositionSide.FLAT:
                structural_break_exit = self.structural_break_detected
                cash, position, a_shares, b_shares, trade_days, current_trade = (
                    self._close_position(
                        i, cash, a_shares, b_shares, a_price, b_price, z,
                        position, trade_days, current_trade, trades,
                        structural_break_exit=structural_break_exit,
                    )
                )
            elif position == PositionSide.FLAT:
                entry = self._check_entry(
                    i, z, a_price, b_price, cash, capital_per_leg, scale,
                )
                if entry is not None:
                    position, a_shares, b_shares, cash, current_trade = entry
                    trade_days = 0

            if position != PositionSide.FLAT:
                trade_days += 1
                borrow_cost = self._compute_borrow_cost(
                    position, a_shares, b_shares, a_price, b_price,
                )
                cash -= borrow_cost

            portfolio_value = cash + a_shares * a_price + b_shares * b_price
            equity_curve.append(portfolio_value)

        self._close_final_position(
            prices_arr, z_arr, cash, a_shares, b_shares,
            position, current_trade, trades, trade_days,
        )

        equity = pd.Series(equity_curve, index=self.prices.index[: len(equity_curve)])
        return equity, trades

    def _check_exit(
        self, position: PositionSide, z: float, trade_days: int
    ) -> bool:
        if position == PositionSide.FLAT:
            return False
        if self.structural_break_detected:
            return True
        if position == PositionSide.LONG_SPREAD and z >= self.z_exit:
            return True
        if position == PositionSide.SHORT_SPREAD and z <= self.z_exit:
            return True
        if abs(z) >= self.z_entry * self.stop_loss:
            return True
        if trade_days >= self.max_holding_days:
            return True
        return False

    def _check_entry(
        self,
        i: int,
        z: float,
        a_price: float,
        b_price: float,
        cash: float,
        capital_per_leg: float,
        scale: float,
    ) -> Optional[tuple]:
        if z < -self.z_entry:
            direction = PositionSide.LONG_SPREAD
            pos = 1
        elif z > self.z_entry:
            direction = PositionSide.SHORT_SPREAD
            pos = -1
        else:
            return None

        alloc = capital_per_leg * scale
        a_shares = pos * alloc / a_price
        b_shares = -a_shares * self.hedge_ratio if self.hedge_ratio != 0 else 0.0

        if self.beta_neutral and self.hedge_ratio != 0:
            b_shares = -a_shares * self.market_beta

        gross = abs(a_shares * a_price) + abs(b_shares * b_price)
        slippage_cost = gross * self.slippage
        tcost = gross * self.transaction_cost
        total_cost = slippage_cost + tcost
        net_cash_flow = a_shares * a_price + b_shares * b_price

        idx_date = self.prices.index[i]
        if hasattr(idx_date, "date"):
            entry_date_str = str(idx_date.date())
        else:
            entry_date_str = str(idx_date)
        entry = {
            "direction": direction.value,
            "entry_date": entry_date_str,
            "entry_zscore": float(z),
            "entry_price_a": float(a_price),
            "entry_price_b": float(b_price),
            "shares_a": float(a_shares),
            "shares_b": float(b_shares),
            "capital_at_entry": cash,
        }
        return direction, a_shares, b_shares, cash - net_cash_flow - total_cost, entry

    def _close_position(
        self,
        i: int,
        cash: float,
        a_shares: float,
        b_shares: float,
        a_price: float,
        b_price: float,
        z: float,
        position: PositionSide,
        trade_days: int,
        current_trade: Optional[dict],
        trades: list[TradeRecord],
        structural_break_exit: bool = False,
    ) -> tuple:
        a_val = a_shares * a_price
        b_val = b_shares * b_price
        gross = abs(a_val) + abs(b_val)
        cost = gross * (self.transaction_cost + self.slippage)
        net = a_val + b_val - cost
        cash += net

        if current_trade is not None:
            if structural_break_exit:
                exit_reason = "structural_break"
            else:
                exit_reason = self._get_exit_reason(position, z, trade_days)
            ret_pct = (cash - current_trade["capital_at_entry"]) / current_trade["capital_at_entry"] * 100
            idx_date = self.prices.index[i]
            exit_date_str = str(idx_date.date()) if hasattr(idx_date, "date") else str(idx_date)
            record = TradeRecord(
                direction=current_trade["direction"],
                entry_date=current_trade["entry_date"],
                exit_date=exit_date_str,
                entry_zscore=current_trade["entry_zscore"],
                exit_zscore=float(z),
                entry_price_a=current_trade["entry_price_a"],
                entry_price_b=current_trade["entry_price_b"],
                exit_price_a=float(a_price),
                exit_price_b=float(b_price),
                shares_a=current_trade["shares_a"],
                shares_b=current_trade["shares_b"],
                return_pct=round(ret_pct, 2),
                pnl=round(cash - current_trade["capital_at_entry"], 2),
                holding_period=trade_days,
                exit_reason=exit_reason,
            )
            trades.append(record)

        return cash, PositionSide.FLAT, 0.0, 0.0, 0, None

    @staticmethod
    def _get_exit_reason(position: PositionSide, z: float, trade_days: int) -> str:
        if abs(z) >= DEFAULT_Z_ENTRY * DEFAULT_STOP_LOSS:
            return "stop_loss"
        if trade_days >= DEFAULT_MAX_HOLDING_DAYS:
            return "max_hold"
        return "signal"

    @staticmethod
    def _set_exit_reason_structural_break() -> str:
        return "structural_break"

    @staticmethod
    def _compute_borrow_cost(
        position: PositionSide,
        a_shares: float,
        b_shares: float,
        a_price: float,
        b_price: float,
    ) -> float:
        if position == PositionSide.FLAT:
            return 0.0
        short_value = 0.0
        if a_shares < 0:
            short_value += abs(a_shares * a_price)
        if b_shares < 0:
            short_value += abs(b_shares * b_price)
        return short_value * DEFAULT_BORROW_FEE

    def _close_final_position(
        self,
        prices_arr: np.ndarray,
        z_arr: np.ndarray,
        cash: float,
        a_shares: float,
        b_shares: float,
        position: PositionSide,
        current_trade: Optional[dict],
        trades: list[TradeRecord],
        trade_days: int,
    ) -> None:
        if position != PositionSide.FLAT and current_trade is not None:
            a_price = float(prices_arr[-1, 0])
            b_price = float(prices_arr[-1, 1])
            z = float(z_arr[-1])
            self._close_position(
                len(self.prices) - 1,
                cash,
                a_shares,
                b_shares,
                a_price,
                b_price,
                z,
                position,
                trade_days,
                current_trade,
                trades,
            )
