import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from backend.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from backend.strategies.base import Portfolio, Signal
from backend.strategies.registry import get_strategy
from backend.api.websocket import _parse_timeframe


class LiveEngine:
    def __init__(
        self,
        strategy_name: str,
        parameters: dict | None,
        symbols: list[str],
        timeframe_str: str,
    ):
        self.strategy_name = strategy_name
        self.parameters = parameters or {}
        self.symbols = symbols
        self.timeframe_str = timeframe_str

        self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

        self.portfolio = Portfolio(0.0)
        self.strategies: dict[str, object] = {}
        self.data_buffers: dict[str, pd.DataFrame] = {}
        self.running = False
        self._task: asyncio.Task | None = None
        self._callbacks: dict | None = None
        self._last_timestamps: dict[str, pd.Timestamp | None] = {}
        self._bar_count: int = 0

    async def start(self, callbacks: dict) -> None:
        self.running = True
        self._callbacks = callbacks

        account = self.trading_client.get_account()
        self.portfolio = Portfolio(float(account.cash))
        for pos in self.trading_client.get_all_positions():
            self.portfolio.positions[pos.symbol] = float(pos.qty)
            self.portfolio.avg_entry[pos.symbol] = float(pos.avg_entry_price)

        if self._callbacks:
            await self._callbacks["on_initial_equity"](
                round(float(account.equity), 2),
                round(float(account.cash), 2),
                account.timestamp.isoformat() if hasattr(account, "timestamp") and account.timestamp else datetime.now(timezone.utc).isoformat(),
            )

        tf = _parse_timeframe(self.timeframe_str)

        clock = self.trading_client.get_clock()
        if not clock.is_open:
            next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S %Z")
            if self._callbacks:
                await self._callbacks["on_status"]("closed",
                    f"The stock market is closed. You can't trade right now.\n"
                    f"Next open: {next_open}. The engine will auto-start when the market opens."
                )
            self._task = asyncio.create_task(
                self._wait_for_market_open(clock.next_open, tf)
            )
            return

        await self._init_strategies_and_start(tf)

    async def _wait_for_market_open(self, next_open: datetime, tf: TimeFrame) -> None:
        while self.running:
            now = datetime.now(timezone.utc)
            if now >= next_open:
                clock = self.trading_client.get_clock()
                if clock.is_open:
                    if self._callbacks:
                        await self._callbacks["on_status"]("starting", "Market is open! Loading data and starting live trading…")
                    await self._init_strategies_and_start(tf)
                    return
                next_open = clock.next_open
            sleep_secs = min(60, (next_open - datetime.now(timezone.utc)).total_seconds())
            if sleep_secs <= 0:
                sleep_secs = 60
            for _ in range(int(sleep_secs)):
                if not self.running:
                    break
                await asyncio.sleep(1)

        if self._callbacks:
            await self._callbacks["on_status"]("stopped", "Live trading stopped (stopped while waiting for market open)")

    async def _init_strategies_and_start(self, tf: TimeFrame) -> None:
        strategy_cls = get_strategy(self.strategy_name, self.parameters).__class__

        for sym in self.symbols:
            strat = strategy_cls(**self.parameters)
            strat._symbol = sym
            self.strategies[sym] = strat

            buf = self._load_initial_data(sym, tf)
            if buf.empty:
                self.data_buffers[sym] = pd.DataFrame()
                self._last_timestamps[sym] = pd.Timestamp(0, tz="UTC")
                continue
            strat.init(buf)
            self.data_buffers[sym] = buf
            self._last_timestamps[sym] = buf.index[-1]

        if self._callbacks:
            await self._callbacks["on_status"]("running", f"Live started with ${self.portfolio.cash:.2f} cash")

        if tf == TimeFrame.Day:
            poll_seconds = 300
        elif hasattr(tf, "timeframe") and tf.timeframe == TimeFrame.Minute:
            poll_seconds = max(tf.amount * 60, 30)
        elif hasattr(tf, "timeframe") and tf.timeframe == TimeFrame.Hour:
            poll_seconds = max(tf.amount * 60, 60)
        else:
            poll_seconds = 300

        self._task = asyncio.create_task(self._run_loop(tf, poll_seconds))

    def _load_initial_data(self, symbol: str, tf: TimeFrame) -> pd.DataFrame:
        now = datetime.now(timezone.utc)
        if tf == TimeFrame.Day:
            start = now - timedelta(days=120)
        else:
            start = now - timedelta(days=7)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=now,
                timeframe=tf,
            )
            bars = self.data_client.get_stock_bars(request)
            if bars.df.empty:
                return pd.DataFrame()
            df = bars.df.reset_index()
            df = df.drop(columns=["symbol"], errors="ignore")
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    async def _run_loop(self, tf: TimeFrame, poll_seconds: int) -> None:
        while self.running:
            try:
                await self._check_for_new_bars(tf)
            except Exception as e:
                if self._callbacks:
                    await self._callbacks["on_error"](str(e))

            for _ in range(poll_seconds):
                if not self.running:
                    break
                await asyncio.sleep(1)

        if self._callbacks:
            await self._callbacks["on_status"]("stopped", "Live trading stopped")

    async def _check_for_new_bars(self, tf: TimeFrame) -> None:
        now = datetime.now(timezone.utc)
        lookback = now - timedelta(hours=24 if tf == TimeFrame.Day else 2)

        for sym in self.symbols:
            df = self._get_recent_bars(sym, tf, lookback, now)
            if df.empty:
                continue

            last_ts = self._last_timestamps.get(sym)
            new_bars = df[df.index > last_ts] if last_ts is not None else df

            if new_bars.empty:
                continue

            if sym not in self.data_buffers or self.data_buffers[sym].empty:
                self.data_buffers[sym] = new_bars
            else:
                self.data_buffers[sym] = pd.concat([self.data_buffers[sym], new_bars])
            self._last_timestamps[sym] = new_bars.index[-1]

            for idx in new_bars.index:
                i = self.data_buffers[sym].index.get_loc(idx)
                signal = self.strategies[sym].next(i, self.data_buffers[sym], self.portfolio)
                self._bar_count += 1
                await self._execute_signal(sym, signal, float(new_bars.loc[idx, "close"]), idx)

    def _get_recent_bars(self, symbol: str, tf: TimeFrame, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=tf,
            )
            bars = self.data_client.get_stock_bars(request)
            if bars.df.empty:
                return pd.DataFrame()
            df = bars.df.reset_index()
            df = df.drop(columns=["symbol"], errors="ignore")
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return pd.DataFrame()

    async def _execute_signal(self, symbol: str, signal: str, price: float, timestamp: pd.Timestamp) -> None:
        trades: list[dict] = []

        if signal == Signal.BUY:
            trade_amt = getattr(self.strategies[symbol], "buy_size", None)
            buy_cash = min(trade_amt, self.portfolio.cash) if trade_amt is not None else self.portfolio.cash
            if buy_cash <= 0:
                return
            try:
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    notional=round(buy_cash, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_req)
                filled_qty = float(order.filled_qty or (buy_cash / price))
                filled_price = float(order.filled_avg_price or price)
                cost = filled_qty * filled_price
                self.portfolio.cash -= cost
                existing = self.portfolio.positions.get(symbol, 0)
                existing_cost = existing * self.portfolio.avg_entry.get(symbol, 0)
                total = existing + filled_qty
                self.portfolio.avg_entry[symbol] = (existing_cost + cost) / total if total > 0 else 0
                self.portfolio.positions[symbol] = total
                trades.append({
                    "symbol": symbol, "side": "buy", "qty": filled_qty,
                    "price": filled_price, "pnl": None,
                })
            except Exception as e:
                if self._callbacks:
                    await self._callbacks["on_error"](f"Buy {symbol} failed: {e}")

        elif signal in (Signal.SELL, Signal.EXIT):
            current_pos = self.portfolio.positions.get(symbol, 0)
            if current_pos <= 0:
                return
            avg_entry = self.portfolio.avg_entry.get(symbol, price)
            if signal == Signal.EXIT:
                qty = current_pos
            else:
                sell_pct = getattr(self.strategies[symbol], "sell_portion", None)
                qty = current_pos * (sell_pct / 100) if sell_pct is not None else current_pos
            if qty <= 0:
                return
            try:
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=round(qty, 6),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = self.trading_client.submit_order(order_req)
                filled_qty = float(order.filled_qty or qty)
                filled_price = float(order.filled_avg_price or price)
                proceeds = filled_qty * filled_price
                pnl = round(proceeds - (avg_entry * filled_qty), 2)
                self.portfolio.cash += proceeds
                new_pos = current_pos - filled_qty
                self.portfolio.positions[symbol] = max(0, new_pos)
                if new_pos <= 0:
                    self.portfolio.avg_entry[symbol] = 0.0
                trades.append({
                    "symbol": symbol, "side": "sell", "qty": filled_qty,
                    "price": filled_price, "pnl": pnl,
                })
            except Exception as e:
                if self._callbacks:
                    await self._callbacks["on_error"](f"Sell {symbol} failed: {e}")

        if trades and self._callbacks:
            await self._callbacks["on_trades"](trades, str(timestamp))

    def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            self._task = None


_engine: LiveEngine | None = None


def get_live_engine() -> LiveEngine | None:
    global _engine
    return _engine


def set_live_engine(engine: LiveEngine | None) -> None:
    global _engine
    _engine = engine
