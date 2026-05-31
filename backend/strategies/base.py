from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class Signal:
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"


class Portfolio:
    def __init__(self, cash: float = 10000.0):
        self.cash = cash
        self.positions: dict[str, float] = {}
        self.avg_entry: dict[str, float] = {}
        self.equity_history: list[float] = []
        self.trades: list[dict] = []


class Strategy(ABC):
    @abstractmethod
    def init(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        pass

    def on_trade(self, side: str, symbol: str, qty: float, price: float) -> None:
        pass
