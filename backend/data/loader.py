from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
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

        url = "https://data.alpaca.markets/v1beta1/corporate-actions"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        params = {
            "symbols": symbol,
            "types": "cash_dividend",
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "limit": 10000,
        }

        resp = httpx.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        dividends = data.get("corporate_actions", {}).get("cash_dividends", [])
        if not dividends:
            df = pd.DataFrame(columns=["ex_date", "rate"])
        else:
            df = pd.DataFrame(dividends)
            df = df.rename(columns={"rate": "dividend"})
            df["ex_date"] = pd.to_datetime(df["ex_date"])
            df = df.set_index("ex_date")
            df = df[["dividend"]]

        if use_cache:
            df.to_parquet(cache_path)

        return df

    def list_cache(self) -> list[str]:
        return [p.name for p in CACHE_DIR.glob("*.parquet")]

    def clear_cache(self) -> None:
        for p in CACHE_DIR.glob("*.parquet"):
            p.unlink()
