from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from backend.config import ALPACA_API_KEY, ALPACA_SECRET_KEY


CACHE_DIR = Path(__file__).parent / "cache"


class DataLoader:
    def __init__(self):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
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

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
        )
        bars = self.client.get_stock_bars(request)

        if bars.df.empty:
            raise ValueError(f"No data returned for {symbol} from {start.date()} to {end.date()}")

        df = bars.df.reset_index()
        df = df.drop(columns=["symbol"], errors="ignore")
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

        if use_cache:
            df.to_parquet(cache_path)

        return df

    def list_cache(self) -> list[str]:
        return [p.name for p in CACHE_DIR.glob("*.parquet")]

    def clear_cache(self) -> None:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
