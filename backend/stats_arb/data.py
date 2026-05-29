"""
Phase 1 — Data Pipeline.

Downloads, caches, and prepares adjusted OHLCV data for pairs analysis.
Supports configurable ticker universes, date ranges, and handles
missing data gracefully.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("stats_arb.data")


class DataManager:
    """Downloads and manages price data for the statistical arbitrage framework.

    Features:
        - yfinance download with auto_adjust=True
        - Local pickle cache to avoid redundant downloads
        - Forward-fill + drop-na for missing data handling
        - Returns aligned DataFrame with common index
        - Configurable ticker universe and date range

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols (e.g., ["NVDA", "AMD"]).
    start : str or datetime
        Start date for data download.
    end : str or datetime, optional
        End date (defaults to today).
    cache : bool
        Enable local pickle caching (default True).
    cache_dir : str
        Directory for cached data files.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str | datetime,
        end: str | datetime | None = None,
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end) if end is not None else pd.Timestamp.today()

    def fetch(self) -> pd.DataFrame:
        """Download adjusted close prices for all tickers.

        Returns a DataFrame with one column per ticker, indexed by date.
        Raises ValueError if insufficient data is available.
        """
        logger.info(
            f"Downloading data: {', '.join(self.tickers)} "
            f"[{self.start.date()} → {self.end.date()}]"
        )

        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )

        if self._is_single_ticker():
            prices = self._extract_single(data)
        else:
            prices = self._extract_multi(data)

        prices = prices.ffill().dropna(how="any")
        prices.index = pd.to_datetime(prices.index)

        if len(prices) < 50:
            raise ValueError(
                f"Insufficient common data: only {len(prices)} trading days "
                f"for {', '.join(self.tickers)}"
            )

        logger.info(f"Retrieved {len(prices)} rows x {len(prices.columns)} columns")

        return prices

    def fetch_ohlcv(self) -> dict[str, pd.DataFrame]:
        """Download full OHLCV data for every ticker (not just adjusted close).

        Returns a dict mapping ticker -> DataFrame with columns:
            Open, High, Low, Close, Volume, Adj Close (when available).
        """
        result: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            df = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
            )
            if not df.empty:
                result[ticker] = df
        return result

    def _is_single_ticker(self) -> bool:
        return len(self.tickers) == 1

    def _extract_single(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"].squeeze()
                name = self.tickers[0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close.to_frame(name=name)
        else:
            if "Close" in data.columns:
                close = data["Close"].squeeze()
                return close.to_frame(name=self.tickers[0])
        raise ValueError(f"No data for {self.tickers[0]}")

    def _extract_multi(self, data: pd.DataFrame) -> pd.DataFrame:
        prices: list[pd.Series] = []
        cols = data.columns

        if isinstance(cols, pd.MultiIndex):
            tickers_in_data = cols.get_level_values(1).unique().tolist()
        else:
            tickers_in_data = cols.tolist()

        for t in self.tickers:
            if isinstance(cols, pd.MultiIndex):
                l1 = cols.get_level_values(1)
                found = t in l1.values
            else:
                found = t in cols

            if found:
                if isinstance(cols, pd.MultiIndex):
                    close = data["Close"][t].squeeze()
                else:
                    if "Close" in data:
                        close = data["Close"].squeeze()
                    else:
                        close = data[t].squeeze()
                prices.append(close)
            else:
                logger.warning(f"Ticker {t} not found in download; skipping")

        if not prices:
            raise ValueError("No valid ticker data retrieved")

        result = pd.concat(prices, axis=1)
        result.columns = [t for t in self.tickers if t in tickers_in_data]
        return result

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute daily log returns from price DataFrame."""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def resample(prices: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
        """Resample prices to a lower frequency (e.g. weekly, monthly)."""
        return prices.resample(freq).last().dropna()
