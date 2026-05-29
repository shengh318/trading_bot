import pandas as pd

from backend.strategies.base import Strategy, Signal, Portfolio


class SmaCrossover(Strategy):
    def __init__(self, short_window: int = 10, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        self._sma_short: pd.Series | None = None
        self._sma_long: pd.Series | None = None

    def init(self, data: pd.DataFrame) -> None:
        close = data["close"]
        self._sma_short = close.rolling(self.short_window).mean()
        self._sma_long = close.rolling(self.long_window).mean()

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if i < self.long_window:
            return Signal.HOLD

        short = self._sma_short.iloc[i]
        long = self._sma_long.iloc[i]
        prev_short = self._sma_short.iloc[i - 1]
        prev_long = self._sma_long.iloc[i - 1]

        if prev_short <= prev_long and short > long:
            return Signal.BUY
        elif prev_short >= prev_long and short < long:
            return Signal.SELL
        return Signal.HOLD
