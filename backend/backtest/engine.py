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
        strategy._symbol = symbol
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
        self.portfolio = Portfolio(self.initial_cash)
        trades: list[dict] = []
        snapshots: list[dict] = []

        self.strategy.init(self.data)

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            timestamp = row.name if isinstance(row.name, str) else str(row.name)
            price = float(row["close"])

            self._apply_dividend(timestamp)

            signal = self.strategy.next(i, self.data, self.portfolio)

            if signal == Signal.BUY and self.portfolio.cash > 0 and price > 0:
                strat = self.strategy
                trade_amt = getattr(strat, "buy_size", None)
                buy_cash = min(trade_amt, self.portfolio.cash) if trade_amt is not None else self.portfolio.cash
                qty = buy_cash / price
                if qty > 0:
                    cost = qty * price
                    existing_shares = self.portfolio.positions.get(self.symbol, 0)
                    existing_cost = existing_shares * self.portfolio.avg_entry.get(self.symbol, 0)
                    total_shares = existing_shares + qty
                    total_cost = existing_cost + cost
                    self.portfolio.avg_entry[self.symbol] = total_cost / total_shares

                    self.portfolio.cash -= cost
                    self.portfolio.positions[self.symbol] = total_shares
                    self.strategy.on_trade("buy", self.symbol, qty, price)

                    trades.append({
                        "bar_index": i,
                        "timestamp": timestamp,
                        "symbol": self.symbol,
                        "side": "buy",
                        "qty": qty,
                        "price": price,
                        "pnl": None,
                    })

            elif signal in (Signal.SELL, Signal.EXIT) and self.portfolio.positions.get(self.symbol, 0) > 0:
                strat = self.strategy
                current_pos = self.portfolio.positions[self.symbol]
                avg_entry = self.portfolio.avg_entry.get(self.symbol, price)

                if signal == Signal.EXIT:
                    qty = current_pos
                else:
                    sell_pct = getattr(strat, "sell_portion", None)
                    if sell_pct is not None:
                        qty = current_pos * sell_pct / 100
                    else:
                        qty = current_pos

                if qty > 0:
                    proceeds = qty * price
                    pnl = round(proceeds - (avg_entry * qty), 2)
                    self.portfolio.cash += proceeds
                    new_pos = current_pos - qty
                    self.portfolio.positions[self.symbol] = new_pos

                    if new_pos > 0:
                        pass
                    else:
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
        self.portfolio = Portfolio(self.initial_cash)
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

            if signal == Signal.BUY and self.portfolio.cash > 0 and price > 0:
                strat = self.strategy
                trade_amt = getattr(strat, "buy_size", None)
                buy_cash = min(trade_amt, self.portfolio.cash) if trade_amt is not None else self.portfolio.cash
                qty = buy_cash / price
                if qty > 0:
                    cost = qty * price
                    existing_shares = self.portfolio.positions.get(self.symbol, 0)
                    existing_cost = existing_shares * self.portfolio.avg_entry.get(self.symbol, 0)
                    total_shares = existing_shares + qty
                    total_cost = existing_cost + cost
                    self.portfolio.avg_entry[self.symbol] = total_cost / total_shares

                    self.portfolio.cash -= cost
                    self.portfolio.positions[self.symbol] = total_shares
                    self.strategy.on_trade("buy", self.symbol, qty, price)

                    event_trade = {
                        "bar_index": i,
                        "timestamp": timestamp,
                        "symbol": self.symbol,
                        "side": "buy",
                        "qty": qty,
                        "price": price,
                        "pnl": None,
                    }

            elif signal in (Signal.SELL, Signal.EXIT) and self.portfolio.positions.get(self.symbol, 0) > 0:
                strat = self.strategy
                current_pos = self.portfolio.positions[self.symbol]
                avg_entry = self.portfolio.avg_entry.get(self.symbol, price)

                if signal == Signal.EXIT:
                    qty = current_pos
                else:
                    sell_pct = getattr(strat, "sell_portion", None)
                    if sell_pct is not None:
                        qty = current_pos * sell_pct / 100
                    else:
                        qty = current_pos

                if qty > 0:
                    proceeds = qty * price
                    pnl = round(proceeds - (avg_entry * qty), 2)
                    self.portfolio.cash += proceeds
                    new_pos = current_pos - qty
                    self.portfolio.positions[self.symbol] = new_pos

                    if new_pos > 0:
                        pass
                    else:
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


class MultiSymbolBacktestEngine:
    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        strategy_cls: type[Strategy],
        initial_cash: float = 10000.0,
        dividends: dict[str, pd.DataFrame] | None = None,
        parameters: dict | None = None,
    ):
        for sym, df in data.items():
            required = {"open", "high", "low", "close", "volume"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{sym} data missing required columns: {missing}")

        self.data = data
        self.symbols = sorted(data.keys())
        self.initial_cash = initial_cash
        self.portfolio = Portfolio(initial_cash)
        self._dividends = dividends or {}

        params = parameters or {}
        self.strategies: dict[str, Strategy] = {}
        for sym in self.symbols:
            strategy = strategy_cls(**params)
            strategy._symbol = sym
            strategy.init(data[sym])
            self.strategies[sym] = strategy

    def _get_union_timestamps(self) -> list[pd.Timestamp]:
        all_ts = set()
        for df in self.data.values():
            all_ts.update(df.index)
        return sorted(all_ts)

    def _apply_dividend(self, symbol: str, timestamp: pd.Timestamp) -> float:
        div_df = self._dividends.get(symbol)
        if div_df is None or div_df.empty:
            return 0.0
        ts = pd.to_datetime(timestamp)
        match = div_df[div_df.index == ts]
        if match.empty:
            return 0.0
        shares = self.portfolio.positions.get(symbol, 0)
        if shares <= 0:
            return 0.0
        amount = float(match.iloc[0]["dividend"]) * shares
        self.portfolio.cash += amount
        return round(amount, 2)

    def run(self) -> BacktestResult:
        self.portfolio = Portfolio(self.initial_cash)
        trades: list[dict] = []
        snapshots: list[dict] = []
        timestamps = self._get_union_timestamps()
        local_idx: dict[str, int] = {sym: -1 for sym in self.symbols}

        for global_i, ts in enumerate(timestamps):
            ts = pd.Timestamp(ts)

            for sym in self.symbols:
                if ts not in self.data[sym].index:
                    continue
                local_idx[sym] += 1
                i = local_idx[sym]
                price = float(self.data[sym].loc[ts, "close"])

                self._apply_dividend(sym, ts)
                signal = self.strategies[sym].next(i, self.data[sym], self.portfolio)

                if signal == Signal.BUY and self.portfolio.cash > 0 and price > 0:
                    strat = self.strategies[sym]
                    trade_amt = getattr(strat, "buy_size", None)
                    buy_cash = min(trade_amt, self.portfolio.cash) if trade_amt is not None else self.portfolio.cash
                    qty = buy_cash / price
                    if qty > 0:
                        cost = qty * price
                        existing = self.portfolio.positions.get(sym, 0)
                        existing_cost = existing * self.portfolio.avg_entry.get(sym, 0)
                        total_shares = existing + qty
                        total_cost = existing_cost + cost
                        self.portfolio.avg_entry[sym] = total_cost / total_shares
                        self.portfolio.cash -= cost
                        self.portfolio.positions[sym] = total_shares
                        self.strategies[sym].on_trade("buy", sym, qty, price)

                        trades.append({
                            "bar_index": i,
                            "timestamp": str(ts),
                            "symbol": sym,
                            "side": "buy",
                            "qty": qty,
                            "price": price,
                            "pnl": None,
                        })

                elif signal in (Signal.SELL, Signal.EXIT) and self.portfolio.positions.get(sym, 0) > 0:
                    strat = self.strategies[sym]
                    current_pos = self.portfolio.positions[sym]
                    avg = self.portfolio.avg_entry.get(sym, price)

                    if signal == Signal.EXIT:
                        qty = current_pos
                    else:
                        sell_pct = getattr(strat, "sell_portion", None)
                        if sell_pct is not None:
                            qty = current_pos * sell_pct / 100
                        else:
                            qty = current_pos

                    if qty > 0:
                        proceeds = qty * price
                        pnl = round(proceeds - (avg * qty), 2)
                        self.portfolio.cash += proceeds
                        new_pos = current_pos - qty
                        self.portfolio.positions[sym] = new_pos

                        if new_pos > 0:
                            pass
                        else:
                            self.portfolio.positions[sym] = 0
                            self.portfolio.avg_entry[sym] = 0.0

                        trades.append({
                            "bar_index": i,
                            "timestamp": str(ts),
                            "symbol": sym,
                            "side": "sell",
                            "qty": qty,
                            "price": price,
                            "pnl": pnl,
                        })

            equity = self.portfolio.cash + sum(
                self.portfolio.positions.get(sym, 0) * float(self.data[sym].loc[ts, "close"])
                for sym in self.symbols
                if ts in self.data[sym].index and self.portfolio.positions.get(sym, 0) > 0
            )

            snapshots.append({
                "bar_index": global_i,
                "timestamp": str(ts),
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
        self.portfolio = Portfolio(self.initial_cash)
        timestamps = self._get_union_timestamps()
        local_idx: dict[str, int] = {sym: -1 for sym in self.symbols}

        for global_i, ts in enumerate(timestamps):
            ts = pd.Timestamp(ts)
            bar_trades: list[dict] = []
            bar_signals: list[str] = []
            bar_dividends: list[dict] = []

            for sym in self.symbols:
                if ts not in self.data[sym].index:
                    continue
                local_idx[sym] += 1
                i = local_idx[sym]
                price = float(self.data[sym].loc[ts, "close"])

                div_amount = self._apply_dividend(sym, ts)
                if div_amount > 0:
                    bar_dividends.append({
                        "bar_index": global_i,
                        "timestamp": str(ts),
                        "symbol": sym,
                        "dividend": div_amount,
                    })

                signal = self.strategies[sym].next(i, self.data[sym], self.portfolio)
                bar_signals.append(signal)

                if signal == Signal.BUY and self.portfolio.cash > 0 and price > 0:
                    strat = self.strategies[sym]
                    trade_amt = getattr(strat, "buy_size", None)
                    buy_cash = min(trade_amt, self.portfolio.cash) if trade_amt is not None else self.portfolio.cash
                    qty = buy_cash / price
                    if qty > 0:
                        cost = qty * price
                        existing = self.portfolio.positions.get(sym, 0)
                        existing_cost = existing * self.portfolio.avg_entry.get(sym, 0)
                        total_shares = existing + qty
                        total_cost = existing_cost + cost
                        self.portfolio.avg_entry[sym] = total_cost / total_shares
                        self.portfolio.cash -= cost
                        self.portfolio.positions[sym] = total_shares
                        self.strategies[sym].on_trade("buy", sym, qty, price)

                        bar_trades.append({
                            "bar_index": global_i,
                            "timestamp": str(ts),
                            "symbol": sym,
                            "side": "buy",
                            "qty": qty,
                            "price": price,
                            "pnl": None,
                        })

                elif signal in (Signal.SELL, Signal.EXIT) and self.portfolio.positions.get(sym, 0) > 0:
                    strat = self.strategies[sym]
                    current_pos = self.portfolio.positions[sym]
                    avg = self.portfolio.avg_entry.get(sym, price)

                    if signal == Signal.EXIT:
                        qty = current_pos
                    else:
                        sell_pct = getattr(strat, "sell_portion", None)
                        if sell_pct is not None:
                            qty = current_pos * sell_pct / 100
                        else:
                            qty = current_pos

                    if qty > 0:
                        proceeds = qty * price
                        pnl = round(proceeds - (avg * qty), 2)
                        self.portfolio.cash += proceeds
                        new_pos = current_pos - qty
                        self.portfolio.positions[sym] = new_pos

                        if new_pos > 0:
                            pass
                        else:
                            self.portfolio.positions[sym] = 0
                            self.portfolio.avg_entry[sym] = 0.0

                        bar_trades.append({
                            "bar_index": global_i,
                            "timestamp": str(ts),
                            "symbol": sym,
                            "side": "sell",
                            "qty": qty,
                            "price": price,
                            "pnl": pnl,
                        })

            equity = self.portfolio.cash + sum(
                self.portfolio.positions.get(sym, 0) * float(self.data[sym].loc[ts, "close"])
                for sym in self.symbols
                if ts in self.data[sym].index and self.portfolio.positions.get(sym, 0) > 0
            )

            snapshot = {
                "bar_index": global_i,
                "timestamp": str(ts),
                "equity": round(equity, 2),
                "cash": round(self.portfolio.cash, 2),
            }

            yield {"snapshot": snapshot, "trade": bar_trades, "signal": bar_signals, "dividend": bar_dividends}
