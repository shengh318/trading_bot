"""AutoCointStrategy — auto-discovers best cointegrated pair from a symbol list."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from backend.stats_arb.cointegration import CointegrationTester
from backend.strategies.base import Strategy, Signal, Portfolio

logger = logging.getLogger("auto_coint_strategy")


class AutoCointStrategy(Strategy):
    """Pairs mean-reversion strategy that auto-discovers the best cointegrated
    pair from a comma-separated list of symbols.

    At ``init()`` the strategy:
    1. Downloads price data for all symbols in the list.
    2. Computes pairwise Pearson correlations.
    3. Runs Engle-Granger cointegration on the top correlated pairs.
    4. Picks the best cointegrated pair (lowest p-value).
    5. Pre-computes the spread z-score for the full history.

    On each ``next()`` bar the strategy enters a mean-reversion bet when the
    z-score exceeds the configured thresholds and exits when it reverts, a
    stop-loss is hit, or the holding period expires.

    Parameters
    ----------
    symbols : str
        Comma-separated tickers to search for pairs (e.g. ``"NVDA,AMD,INTC"``).
    min_corr : float
        Minimum Pearson correlation to consider (default 0.7).
    z_entry : float
        Z-score threshold to enter (default 2.0).
    z_exit : float
        Z-score threshold to exit (default 0.0).
    stop_loss : float
        Stop-loss as a multiple of *z_entry* (default 3.0).
    max_holding_days : int
        Maximum bars a position may be held (default 40).
    base_pair_capital : float
        Total capital allocated to the pair; half goes to the primary leg
        (default 10 000).
    min_trading_days : int
        Minimum days of price history required (default 504).
    """

    def __init__(
        self,
        symbols: str = "NVDA,AMD",
        min_corr: float = 0.3,
        z_entry: float = 1.5,
        z_exit: float = 0.0,
        stop_loss: float = 3.0,
        max_holding_days: int = 40,
        base_pair_capital: float = 10000.0,
        min_trading_days: int = 504,
    ) -> None:
        self.symbols_str = symbols
        self.min_corr = min_corr
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss = stop_loss
        self.max_holding_days = max_holding_days
        self.base_pair_capital = base_pair_capital
        self.min_trading_days = min_trading_days

        self.buy_size: float = base_pair_capital / 2
        self.sell_portion: float = 100.0

        self._symbol_b: str = ""
        self._hedge_ratio: float = 1.0
        self._spread: pd.Series = None
        self._zscore: pd.Series = None
        self._in_pair: bool = False
        self._entry_bar: int = -1
        self._entry_z: float = 0.0
        self._consecutive_losses: int = 0
        self._cooldown_until: int = -1
        self._last_trade_pnl: float = 0.0

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

    def init(self, data: pd.DataFrame) -> None:
        all_symbols = [s.strip().upper() for s in self.symbols_str.split(",") if s.strip()]
        if len(all_symbols) < 2:
            logger.error("AutoCointStrategy: need at least 2 symbols. All HOLD.")
            self._create_dummy()
            return

        symbol_a = getattr(self, "_symbol", all_symbols[0])
        if symbol_a not in all_symbols:
            all_symbols.insert(0, symbol_a)

        end_date = data.index[-1]
        lookback = pd.Timestamp(end_date) - pd.Timedelta(days=365 * 12)
        discovery_start = min(pd.Timestamp(data.index[0]), lookback)
        prices = self._download_prices(all_symbols, discovery_start, end_date)
        if prices is None or len(prices.columns) < 2:
            logger.error("AutoCointStrategy: could not download price data. All HOLD.")
            self._create_dummy()
            return

        best = self._find_best_pair(prices, symbol_a)
        if best is None:
            logger.error("AutoCointStrategy: no cointegrated pair found. All HOLD.")
            self._create_dummy()
            return

        symbol_b, hedge_ratio = best
        self._symbol_b = symbol_b
        self._hedge_ratio = hedge_ratio if hedge_ratio != 0 else 1.0

        close_a = data["close"].astype(float)
        if symbol_b in prices.columns:
            close_b = prices[symbol_b].reindex(data.index, method="ffill").astype(float)
        else:
            close_b = close_a.copy()

        clean = close_a.notna() & close_b.notna() & (close_a > 0) & (close_b > 0)
        close_a = close_a[clean]
        close_b = close_b[clean]

        if len(close_a) < 60:
            logger.error("AutoCointStrategy: < 60 valid bars. All HOLD.")
            self._create_dummy()
            return

        spread = close_a - self._hedge_ratio * close_b
        exp_mean = spread.expanding().mean()
        exp_std = spread.expanding().std().replace(0, np.nan)
        zscore = (spread - exp_mean) / exp_std

        self._spread = spread
        self._zscore = zscore
        logger.info(
            "AutoCointStrategy: selected pair %s/%s, hedge_ratio=%.4f, "
            "z-score range [%.2f, %.2f] over %d bars",
            symbol_a, symbol_b, hedge_ratio,
            float(zscore.min()), float(zscore.max()), len(zscore),
        )

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if self._zscore is None or self._zscore.empty:
            return Signal.HOLD
        if i >= len(self._zscore):
            return Signal.HOLD

        z = float(self._zscore.iloc[i])
        price_a = float(data["close"].iloc[i]) if "close" in data.columns else 0.0

        if not np.isfinite(z):
            return Signal.HOLD

        if self._cooldown_until >= i:
            return Signal.HOLD

        if not self._in_pair:
            return self._check_entry(i, z, price_a)

        return self._check_exit(i, z, price_a)

    def on_trade(self, side: str, symbol: str, qty: float, price: float) -> None:
        if side == "sell" and not self._in_pair:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _create_dummy(self) -> None:
        self._spread = pd.Series(dtype=float)
        self._zscore = pd.Series(dtype=float)

    def _download_prices(self, tickers: list[str], start, end) -> pd.DataFrame | None:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            try:
                raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            except Exception:
                return None
        if raw.empty or "Close" not in raw.columns:
            return None
        close = raw["Close"]
        if isinstance(close, pd.Series):
            return None
        if isinstance(close, pd.DataFrame):
            close = close.dropna(how="all", axis=1)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close.ffill()

    def _find_best_pair(self, prices: pd.DataFrame, symbol_a: str) -> tuple[str, float] | None:
        candidates = [c for c in prices.columns if c != symbol_a]
        if not candidates:
            return None

        returns = prices.pct_change().dropna()
        corr_series = returns.corr()[symbol_a].drop(symbol_a).dropna()

        ordered = corr_series.sort_values(ascending=False)
        top_n = min(10, len(ordered))
        candidates_ordered = list(ordered.head(top_n).index)

        best_pair: tuple[str, float] | None = None
        best_p = 1.0

        for sig in (0.05, 0.10):
            for sym_b in candidates_ordered:
                pair_prices = prices[[symbol_a, sym_b]].dropna(how="any")
                if len(pair_prices) < 100:
                    continue
                try:
                    ct = CointegrationTester(pair_prices, significance=sig)
                    result = ct.run()
                    if result.is_cointegrated and result.p_value < best_p:
                        best_p = result.p_value
                        best_pair = (sym_b, result.hedge_ratio)
                except Exception:
                    continue
            if best_pair is not None:
                break

        if best_pair is not None:
            logger.info(
                "AutoCointStrategy: best pair %s/%s p=%.6f hr=%.4f",
                symbol_a, best_pair[0], best_p, best_pair[1],
            )
        else:
            logger.warning(
                "AutoCointStrategy: no cointegrated pair found for %s "
                "(checked %d candidates with %d bars each)",
                symbol_a, len(candidates_ordered), len(prices),
            )
        return best_pair

    def _check_entry(self, i: int, z: float, price_a: float) -> str:
        abs_z = abs(z)
        if abs_z < self.z_entry:
            return Signal.HOLD

        self.buy_size = self.base_pair_capital / 2 if price_a > 0 else self.base_pair_capital / 4
        self.sell_portion = 100.0

        self._in_pair = True
        self._entry_bar = i
        self._entry_z = z
        return Signal.BUY

    def _check_exit(self, i: int, z: float, price_a: float) -> str:
        bars_held = i - self._entry_bar
        abs_z = abs(z)
        side = 1 if self._entry_z < 0 else -1

        if abs_z >= self.z_entry * self.stop_loss:
            self.sell_portion = 100.0
            self._close_trade()
            return Signal.EXIT

        if bars_held >= self.max_holding_days:
            self.sell_portion = 100.0
            self._close_trade()
            return Signal.EXIT

        if (side == 1 and z >= self.z_exit) or (side == -1 and z <= self.z_exit):
            self.sell_portion = 100.0
            self._close_trade()
            return Signal.SELL

        return Signal.HOLD

    def _close_trade(self) -> None:
        self._in_pair = False
        self._entry_bar = -1
        self._entry_z = 0.0
        self.sell_portion = 100.0

    def get_state(self) -> dict[str, Any]:
        return {
            "pair": f"{getattr(self, '_symbol', '?')}/{self._symbol_b}",
            "hedge_ratio": self._hedge_ratio,
            "in_pair": self._in_pair,
            "z_entry": float(self._entry_z) if self._in_pair else None,
            "bars_held": -1,
        }
