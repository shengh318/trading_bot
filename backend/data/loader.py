from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from backend.config import ALPACA_API_KEY, ALPACA_SECRET_KEY


CACHE_DIR = Path(__file__).parent / "cache"


class DataLoader:
    def __init__(self):
        self._has_alpaca = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)
        if self._has_alpaca:
            self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        CACHE_DIR.mkdir(exist_ok=True)

    def _cache_path(self, symbol: str, start: datetime, end: datetime, timeframe: TimeFrame, tag: str = "") -> Path:
        parts = f"{symbol}_{start.date()}_{end.date()}_{timeframe}"
        if tag:
            parts += f"_{tag}"
        return CACHE_DIR / f"{parts}.parquet"

    def load_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, start, end, timeframe, tag="split")

        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        if self._has_alpaca:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    adjustment="split",
                )
                bars = self.client.get_stock_bars(request)

                if bars.df.empty:
                    raise ValueError(f"No data returned for {symbol} from {start.date()} to {end.date()}")

                df = bars.df.reset_index()
                df = df.drop(columns=["symbol"], errors="ignore")
                df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                if use_cache:
                    df.to_parquet(cache_path)

                return df
            except Exception as e:
                import logging
                logging.getLogger("loader").warning(f"Alpaca load_bars failed for {symbol}: {e}; falling back to yfinance")

        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} from {start.date()} to {end.date()}")
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "timestamp"

        if use_cache:
            df.to_parquet(cache_path)

        return df

    def load_dividends(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, start, end, TimeFrame.Day, tag="dividends")

        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        empty = pd.DataFrame(columns=["dividend"])
        empty.index.name = "ex_date"

        try:
            ticker = yf.Ticker(symbol)
            divs = ticker.dividends
            if divs.empty:
                return empty.copy()
            df = divs.to_frame("dividend")
            df.index.name = "ex_date"
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            start_np = np.datetime64(start)
            end_np = np.datetime64(end)
            mask = (df.index.values >= start_np) & (df.index.values <= end_np)
            df = df.iloc[mask].copy()
            if use_cache:
                df.to_parquet(cache_path)
            return df
        except Exception as e:
            import logging
            logging.getLogger("loader").warning(f"Dividends unavailable for {symbol}: {e}")
            return empty.copy()

    def list_cache(self) -> list[str]:
        return [p.name for p in CACHE_DIR.glob("*.parquet")]

    def clear_cache(self) -> None:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
