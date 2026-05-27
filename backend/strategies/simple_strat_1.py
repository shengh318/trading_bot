import pandas as pd

from backend.strategies.base import Strategy, Signal, Portfolio


class SimpleStrat1(Strategy):
    def __init__(
        self,
        buy_size: float = 100.0,
        entry_drop: float = 1.0,
        profit_target: float = 20.0,
        sell_portion: float = 100.0,
        stop_loss: float = 3.0,
        max_buys: int = 1,
    ):
        self.buy_size = buy_size
        self.entry_drop = entry_drop
        self.profit_target = profit_target
        self.sell_portion = sell_portion
        self.stop_loss = stop_loss
        self.max_buys = max_buys

        self._current_date: str | None = None
        self._day_open: float = 0.0
        self._bought_levels: int = 0
        self._last_buy_avg: float = 0.0

    def init(self, data: pd.DataFrame) -> None:
        self._current_date = None
        self._day_open = 0.0
        self._bought_levels = 0
        self._last_buy_avg = 0.0

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        sym = getattr(self, "_symbol", None)
        if sym is None:
            return Signal.HOLD

        close = float(data["close"].iloc[i])
        position = portfolio.positions.get(sym, 0.0)
        avg_entry = portfolio.avg_entry.get(sym, 0.0)

        ts = data.index[i]
        if isinstance(ts, pd.Timestamp):
            day_str = str(ts.date())
        else:
            day_str = str(pd.Timestamp(ts).date())

        if day_str != self._current_date:
            self._current_date = day_str
            self._day_open = float(data["open"].iloc[i])
            self._bought_levels = 0
            self._last_buy_avg = 0.0

        if position > 0 and avg_entry > 0:
            loss_pct = (avg_entry - close) / avg_entry * 100
            if loss_pct >= self.stop_loss:
                return Signal.EXIT

        if position > 0 and avg_entry > 0:
            gain_pct = (close - avg_entry) / avg_entry * 100
            if gain_pct >= self.profit_target:
                return Signal.SELL

        if avg_entry == 0 or position <= 0:
            drop_from_open = (self._day_open - close) / self._day_open * 100
            if drop_from_open >= self.entry_drop:
                self._bought_levels += 1
                self._last_buy_avg = close
                return Signal.BUY
        else:
            if self._bought_levels < self.max_buys:
                next_dca_price = self._last_buy_avg * (1 - self.entry_drop / 100)
                if close <= next_dca_price:
                    self._bought_levels += 1
                    self._last_buy_avg = avg_entry
                    return Signal.BUY

        return Signal.HOLD
