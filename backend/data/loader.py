import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from backend.config import ALPACA_API_KEY, ALPACA_SECRET_KEY

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_START_PRICE = 100.0


class DataLoader:
    def __init__(self):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY) if ALPACA_API_KEY else None
        CACHE_DIR.mkdir(exist_ok=True)

    def load_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = CACHE_DIR / f"{symbol}_{start.date()}_{end.date()}_{timeframe}.parquet"

        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        if self.client:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                )
                bars = self.client.get_stock_bars(request)

                if not bars.df.empty:
                    df = bars.df.reset_index()
                    df = df.drop(columns=["symbol"], errors="ignore")
                    df = df.set_index("timestamp")
                    df.index = pd.to_datetime(df.index)

                    if use_cache:
                        df.to_parquet(cache_path)

                    return df

                logger.warning("Alpaca returned no data for %s", symbol)
            except Exception as e:
                logger.warning("Alpaca API failed, using synthetic data: %s", e)
        else:
            logger.warning("No Alpaca API keys configured, using synthetic data")

        return self._generate_synthetic_bars(symbol, start, end, timeframe, use_cache, cache_path)

    @staticmethod
    def _generate_synthetic_bars(
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame,
        use_cache: bool,
        cache_path: Path,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed=42)
        delta = end - start
        days = delta.days if timeframe == TimeFrame.Day else delta.days * 390
        num_bars = max(days, 2)

        dates = pd.date_range(start=start, periods=num_bars, freq="D" if timeframe == TimeFrame.Day else "h")
        price = DEFAULT_START_PRICE
        opens, highs, lows, closes, volumes = [], [], [], [], []

        for _ in range(num_bars):
            change = rng.normal(0, price * 0.015)
            close = max(price + change, price * 0.5)
            high = max(price, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(price, close) * (1 - abs(rng.normal(0, 0.005)))
            opens.append(round(price, 2))
            highs.append(round(high, 2))
            lows.append(round(low, 2))
            closes.append(round(close, 2))
            volumes.append(int(rng.integers(1_000_000, 10_000_000)))
            price = close

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes,
        }, index=dates)
        df.index.name = "timestamp"

        if use_cache:
            df.to_parquet(cache_path)

        return df

    def list_cache(self) -> list[str]:
        return [p.name for p in CACHE_DIR.glob("*.parquet")]

    def clear_cache(self) -> None:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
