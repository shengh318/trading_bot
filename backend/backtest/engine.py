from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import pandas as pd

from backend.backtest.metrics import calculate_metrics
from backend.strategies.base import Portfolio, Signal, Strategy


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict
    initial_cash: float


class BacktestEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "ASSET",
        initial_cash: float = 10000.0,
        dividends: pd.DataFrame | None = None,
    ):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        self.data = data
        self.strategy = strategy
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.portfolio = Portfolio(initial_cash)

        if dividends is not None and not dividends.empty:
            if "dividend" not in dividends.columns:
                raise ValueError("dividends DataFrame must have a 'dividend' column")
            self._dividends = dividends
        else:
            self._dividends = pd.DataFrame()

    def _apply_dividend(self, timestamp: str) -> float:
        if self._dividends.empty:
            return 0.0
        ts = pd.to_datetime(timestamp)
        match = self._dividends[self._dividends.index == ts]
        if match.empty:
            return 0.0
        shares = self.portfolio.positions.get(self.symbol, 0)
        if shares <= 0:
            return 0.0
        amount = float(match.iloc[0]["dividend"]) * shares
        self.portfolio.cash += amount
        return round(amount, 2)

    def run(self) -> BacktestResult:
        trades: list[dict] = []
        snapshots: list[dict] = []

        self.strategy.init(self.data)

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            timestamp = row.name if isinstance(row.name, str) else str(row.name)
            price = float(row["close"])

            self._apply_dividend(timestamp)

            signal = self.strategy.next(i, self.data, self.portfolio)

            if signal == Signal.BUY and self.portfolio.cash > 0:
                qty = self.portfolio.cash / price
                if qty > 0:
                    cost = qty * price
                    existing_shares = self.portfolio.positions.get(self.symbol, 0)
                    existing_cost = existing_shares * self.portfolio.avg_entry.get(self.symbol, 0)
                    total_shares = existing_shares + qty
                    total_cost = existing_cost + cost
                    self.portfolio.avg_entry[self.symbol] = total_cost / total_shares

                    self.portfolio.cash -= cost
                    self.portfolio.positions[self.symbol] = total_shares

                    trades.append({
                        "bar_index": i,
                        "timestamp": timestamp,
                        "symbol": self.symbol,
                        "side": "buy",
                        "qty": qty,
                        "price": price,
                        "pnl": None,
                    })

            elif signal == Signal.SELL and self.portfolio.positions.get(self.symbol, 0) > 0:
                qty = self.portfolio.positions[self.symbol]
                avg_entry = self.portfolio.avg_entry.get(self.symbol, price)
                proceeds = qty * price
                pnl = round(proceeds - (avg_entry * qty), 2)

                self.portfolio.cash += proceeds
                self.portfolio.positions[self.symbol] = 0
                self.portfolio.avg_entry[self.symbol] = 0.0

                trades.append({
                    "bar_index": i,
                    "timestamp": timestamp,
                    "symbol": self.symbol,
                    "side": "sell",
                    "qty": qty,
                    "price": price,
                    "pnl": pnl,
                })

            equity = self.portfolio.cash + sum(
                pos * float(self.data.iloc[i]["close"])
                for sym, pos in self.portfolio.positions.items()
                if pos > 0
            )
            snapshots.append({
                "bar_index": i,
                "timestamp": timestamp,
                "equity": round(equity, 2),
                "cash": round(self.portfolio.cash, 2),
            })

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["bar_index", "timestamp", "symbol", "side", "qty", "price", "pnl"]
        )
        equity_df = pd.DataFrame(snapshots)
        metrics = calculate_metrics(equity_df, trades_df, self.initial_cash)

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_df,
            metrics=metrics,
            initial_cash=self.initial_cash,
        )

    def stream(self) -> Generator[dict, None, None]:
        self.strategy.init(self.data)

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            timestamp = row.name if isinstance(row.name, str) else str(row.name)
            price = float(row["close"])

            dividend_amount = self._apply_dividend(timestamp)
            event_dividend = None
            if dividend_amount > 0:
                event_dividend = {
                    "bar_index": i,
                    "timestamp": timestamp,
                    "dividend": dividend_amount,
                }

            signal = self.strategy.next(i, self.data, self.portfolio)
            event_trade = None

            if signal == Signal.BUY and self.portfolio.cash > 0:
                qty = self.portfolio.cash / price
                if qty > 0:
                    cost = qty * price
                    existing_shares = self.portfolio.positions.get(self.symbol, 0)
                    existing_cost = existing_shares * self.portfolio.avg_entry.get(self.symbol, 0)
                    total_shares = existing_shares + qty
                    total_cost = existing_cost + cost
                    self.portfolio.avg_entry[self.symbol] = total_cost / total_shares

                    self.portfolio.cash -= cost
                    self.portfolio.positions[self.symbol] = total_shares

                    event_trade = {
                        "bar_index": i,
                        "timestamp": timestamp,
                        "symbol": self.symbol,
                        "side": "buy",
                        "qty": qty,
                        "price": price,
                        "pnl": None,
                    }

            elif signal == Signal.SELL and self.portfolio.positions.get(self.symbol, 0) > 0:
                qty = self.portfolio.positions[self.symbol]
                avg_entry = self.portfolio.avg_entry.get(self.symbol, price)
                proceeds = qty * price
                pnl = round(proceeds - (avg_entry * qty), 2)

                self.portfolio.cash += proceeds
                self.portfolio.positions[self.symbol] = 0
                self.portfolio.avg_entry[self.symbol] = 0.0

                event_trade = {
                    "bar_index": i,
                    "timestamp": timestamp,
                    "symbol": self.symbol,
                    "side": "sell",
                    "qty": qty,
                    "price": price,
                    "pnl": pnl,
                }

            equity = self.portfolio.cash + sum(
                pos * float(self.data.iloc[i]["close"])
                for sym, pos in self.portfolio.positions.items()
                if pos > 0
            )
            snapshot = {
                "bar_index": i,
                "timestamp": timestamp,
                "equity": round(equity, 2),
                "cash": round(self.portfolio.cash, 2),
            }

            yield {"snapshot": snapshot, "trade": event_trade, "signal": signal, "dividend": event_dividend}
