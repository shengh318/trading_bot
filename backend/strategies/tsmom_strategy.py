"""
Time-Series Momentum (TSMOM) — Research & Backtesting Framework

Phases:
  1. Data Pipeline (yfinance, caching, multi-symbol)
  2. Momentum Features (21d/63d/126d/252d returns, vol, ATR, drawdown, Sharpe, Sortino)
  3. Trend Filters (price>MA200, MA50>MA200, breakout, vol-adjusted)
  4. Signal Generation (binary / weighted / vol-adjusted)
  5. Position Sizing (equal-weight, vol-target, risk-parity, max caps)
  6. Portfolio Construction (single / multi / top-N monthly rebalance)
  7. Walk-Forward Validation (expanding / rolling windows, leakage-free)
  8. Backtest Engine (t-cost, slippage, full metrics)
  9. Benchmarking (SPY/QQQ/EW, alpha/beta/IR/TE)
 10. Regime Analysis (bull/bear/sideways/high-vol)
 11. Visualizations (matplotlib/seaborn charts)
 12. Parameter Research (lookback/filter/rebalance grids)
 13. ML Extension (XGBoost/RF/LightGBM trend-persistence prediction)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_UNIVERSE = ["SPY", "QQQ", "VOO", "NVDA", "AMD", "META"]
DEFAULT_START = "2005-01-01"
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")
TRADING_DAYS_PER_YEAR = 252

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TSMOMConfig:
    symbols: list[str] = field(default_factory=lambda: DEFAULT_UNIVERSE.copy())
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    momentum_lookbacks: list[int] = field(default_factory=lambda: [21, 63, 126, 252])
    vol_lookback: int = 20
    vol_long_lookback: int = 60
    ma_short: int = 50
    ma_long: int = 200
    breakout_lookback: int = 252
    top_n: int = 3
    rebalance_freq: str = "ME"
    signal_type: str = "binary"
    long_only: bool = True
    sizing_method: str = "equal"
    target_volatility: float = 0.15
    max_position_pct: float = 0.20
    max_leverage: float = 1.0
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    initial_capital: float = 100000.0
    trend_filters: list[str] = field(default_factory=lambda: ["price_above_ma200"])
    risk_free_rate: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMDataLoader:
    """Downloads, caches, and aligns multi-symbol price data via yfinance."""

    CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "tsmom"

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load_prices(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        syms = symbols or self.config.symbols
        start_date = start or self.config.start
        end_date = end or self.config.end

        result: dict[str, pd.DataFrame] = {}
        for sym in syms:
            df = self._load_single(sym, start_date, end_date, use_cache)
            if df is not None and not df.empty:
                result[sym] = df
        return result

    def _load_single(
        self, symbol: str, start: str, end: str, use_cache: bool
    ) -> pd.DataFrame | None:
        cache_path = self.CACHE_DIR / f"{symbol}_{start}_{end}.parquet"
        if use_cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    return df
            except Exception:
                pass

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df.empty:
                print(f"  [TSMOM] WARNING: No data for {symbol}")
                return None
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })
            df.index.name = "date"
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if use_cache:
                df.to_parquet(cache_path)
            return df
        except Exception as e:
            print(f"  [TSMOM] ERROR loading {symbol}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMFeatures:
    """Computes momentum, volatility, and risk metrics for each asset."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def compute_all(self, prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for sym, df in prices.items():
            result[sym] = self.compute_features(df)
        return result

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].copy()
        out = df.copy()

        for lookback in self.config.momentum_lookbacks:
            col = f"momentum_{lookback}d"
            out[col] = close.pct_change(lookback)

        out["vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        out["vol_60d"] = close.pct_change().rolling(60).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr"] = tr.rolling(14).mean()

        rolling_max = close.expanding().max()
        out["drawdown"] = (close - rolling_max) / rolling_max * 100
        out["drawdown_252d"] = close.rolling(252).max()
        out["drawdown_252d"] = (close - out["drawdown_252d"]) / out["drawdown_252d"] * 100

        daily_ret = close.pct_change()
        for lookback in [21, 63, 126, 252]:
            rolling_sharpe = (
                daily_ret.rolling(lookback).mean()
                / daily_ret.rolling(lookback).std().replace(0, np.nan)
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
            out[f"sharpe_{lookback}d"] = rolling_sharpe.fillna(0)

            neg_ret = daily_ret.clip(upper=0)
            downside_std = neg_ret.rolling(lookback).std().replace(0, np.nan)
            rolling_sortino = (
                daily_ret.rolling(lookback).mean() / downside_std
                * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
            out[f"sortino_{lookback}d"] = rolling_sortino.fillna(0)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TREND FILTERS
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMTrendFilter:
    """Applies trend filters to qualify assets for momentum trading."""

    FILTERS = {
        "price_above_ma200": "price > 200-day MA",
        "ma50_above_ma200": "50-day MA > 200-day MA",
        "price_above_ma50": "price > 50-day MA",
        "price_above_breakout": "price > 12-month high",
        "volatility_adjusted": "volatility below threshold",
    }

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()
        self._ma_cache: dict[str, pd.DataFrame] = {}

    def apply_filters(
        self, price_df: pd.DataFrame, features_df: pd.DataFrame, enabled_filters: list[str] | None = None
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=price_df.index)
        out["qualified"] = True

        filters = enabled_filters if enabled_filters is not None else self.config.trend_filters
        if not filters:
            return out

        for fname in filters:
            mask = self._apply_single(fname, price_df, features_df)
            if mask is not None:
                out["qualified"] = out["qualified"] & mask

        return out

    def _apply_single(self, fname: str, price_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series | None:
        if fname == "price_above_ma200":
            ma = price_df["close"].rolling(200).mean()
            return price_df["close"] > ma

        if fname == "ma50_above_ma200":
            ma50 = price_df["close"].rolling(50).mean()
            ma200 = price_df["close"].rolling(200).mean()
            return ma50 > ma200

        if fname == "price_above_ma50":
            ma50 = price_df["close"].rolling(50).mean()
            return price_df["close"] > ma50

        if fname == "price_above_breakout":
            breakout = price_df["close"].rolling(252).max().shift(1)
            return price_df["close"] > breakout

        if fname == "volatility_adjusted":
            if "vol_20d" in features_df.columns:
                median_vol = features_df["vol_20d"].rolling(252).median()
                return features_df["vol_20d"] < median_vol * 1.5
            return None

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMSignalGenerator:
    """Generates momentum signals — binary, weighted, or vol-adjusted."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def generate_signals(
        self,
        features: dict[str, pd.DataFrame],
        trend_filters: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for sym, df in features.items():
            signal_df = pd.DataFrame(index=df.index)
            signal_df["momentum_score"] = self._compute_momentum_score(df)
            signal_df["signal_raw"] = self._raw_signal(signal_df["momentum_score"])
            signal_df["signal_binary"] = signal_df["signal_raw"]
            result[sym] = signal_df
        return result

    def _compute_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        weight_sum = 0.0
        lookbacks = self.config.momentum_lookbacks
        weights = {21: 1.0, 63: 2.0, 126: 3.0, 252: 4.0}
        for lb in lookbacks:
            col = f"momentum_{lb}d"
            if col not in df.columns:
                continue
            w = weights.get(lb, 1.0)
            score = score + df[col].fillna(0) * w
            weight_sum += w
        if weight_sum > 0:
            score = score / weight_sum
        return score

    def _raw_signal(self, momentum_score: pd.Series) -> pd.Series:
        if self.config.signal_type == "binary":
            return momentum_score.apply(lambda x: 1.0 if x > 0 else 0.0)
        if self.config.signal_type == "weighted":
            return momentum_score.clip(-1, 1)
        if self.config.signal_type == "vol_adjusted":
            return momentum_score.clip(-1, 1)
        return momentum_score.apply(lambda x: 1.0 if x > 0 else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMPositionSizer:
    """Computes position weights using various sizing methodologies."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def compute_weights(
        self,
        symbols: list[str],
        features: dict[str, pd.DataFrame],
        signals: dict[str, pd.DataFrame],
        idx: int,
    ) -> dict[str, float]:
        method = self.config.sizing_method

        if method == "equal":
            return self._equal_weight(symbols, signals, idx)
        if method == "vol_target":
            return self._volatility_target(symbols, features, idx)
        if method == "risk_parity":
            return self._risk_parity(symbols, features, idx)
        return self._equal_weight(symbols, signals, idx)

    def _equal_weight(self, symbols: list[str], signals: dict[str, pd.DataFrame], idx: int) -> dict[str, float]:
        active = []
        for sym in symbols:
            if idx < len(signals.get(sym, pd.DataFrame())):
                sig = signals[sym].iloc[idx]["signal_binary"]
                if sig > 0:
                    active.append(sym)
        if not active:
            return {s: 0.0 for s in symbols}
        w = 1.0 / len(active)
        return {s: w if s in active else 0.0 for s in symbols}

    def _volatility_target(self, symbols: list[str], features: dict[str, pd.DataFrame], idx: int) -> dict[str, float]:
        weights: dict[str, float] = {}
        total_inverse_vol = 0.0
        for sym in symbols:
            if idx >= len(features.get(sym, pd.DataFrame())):
                weights[sym] = 0.0
                continue
            row = features[sym].iloc[idx]
            vol = row.get("vol_20d", np.nan)
            if pd.isna(vol) or vol <= 0:
                weights[sym] = 0.0
                continue
            inv_vol = 1.0 / vol
            weights[sym] = inv_vol * self.config.target_volatility
            total_inverse_vol += inv_vol

        if total_inverse_vol > 0:
            for sym in symbols:
                weights[sym] = (
                    weights.get(sym, 0.0) / total_inverse_vol * len(symbols)
                )
        return self._apply_caps(weights, symbols)

    def _risk_parity(self, symbols: list[str], features: dict[str, pd.DataFrame], idx: int) -> dict[str, float]:
        vols: dict[str, float] = {}
        total_inv_vol = 0.0
        for sym in symbols:
            if idx >= len(features.get(sym, pd.DataFrame())):
                vols[sym] = 0.0
                continue
            row = features[sym].iloc[idx]
            v = row.get("vol_20d", np.nan)
            if pd.isna(v) or v <= 0:
                vols[sym] = 0.0
                continue
            iv = 1.0 / v
            vols[sym] = iv
            total_inv_vol += iv
        if total_inv_vol <= 0:
            return {s: 0.0 for s in symbols}
        weights = {s: (vols.get(s, 0.0) / total_inv_vol) for s in symbols}
        return self._apply_caps(weights, symbols)

    def _apply_caps(self, weights: dict[str, float], symbols: list[str]) -> dict[str, float]:
        max_pos = self.config.max_position_pct
        for s in symbols:
            weights[s] = min(weights.get(s, 0.0), max_pos)
        total = sum(weights.values())
        if total > self.config.max_leverage:
            scale = self.config.max_leverage / total
            for s in symbols:
                weights[s] *= scale
        return weights


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TSMOMSignalRecord:
    date: str
    symbol: str
    direction: int
    weight: float
    momentum_score: float
    qualified: bool


class TSMOMPortfolioConstruction:
    """Constructs and manages multi-asset momentum portfolios."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def build_portfolio(
        self,
        features: dict[str, pd.DataFrame],
        signals: dict[str, pd.DataFrame],
        trend_filters: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        syms = list(features.keys())
        rebalance_dates = self._get_rebalance_dates(features)
        all_records: list[dict] = []

        first_feat = next(iter(features.values()))
        for date_idx in rebalance_dates:
            date_str = str(first_feat.index[date_idx].date())
            candidates: list[dict] = []

            for sym in syms:
                feat_df = features[sym]
                sig_df = signals.get(sym)
                if sig_df is None or date_idx >= len(sig_df):
                    continue

                momentum_score = float(sig_df.iloc[date_idx]["momentum_score"])
                signal_binary = float(sig_df.iloc[date_idx]["signal_binary"])

                qualified = True
                if trend_filters and sym in trend_filters:
                    tf = trend_filters[sym]
                    if date_idx < len(tf):
                        qualified = bool(tf.iloc[date_idx]["qualified"])

                if signal_binary <= 0 or not qualified:
                    candidates.append({
                        "date": date_str,
                        "symbol": sym,
                        "momentum_score": momentum_score,
                        "signal": signal_binary,
                        "qualified": qualified,
                    })
                    continue

                candidates.append({
                    "date": date_str,
                    "symbol": sym,
                    "momentum_score": momentum_score,
                    "signal": signal_binary,
                    "qualified": qualified,
                })

            if self.config.long_only:
                candidates = [c for c in candidates if c["momentum_score"] > 0]

            candidates.sort(key=lambda x: x["momentum_score"], reverse=True)
            top = candidates[: self.config.top_n]

            active_symbols = [c["symbol"] for c in top]
            weights = self._compute_portfolio_weights(active_symbols, top, features, date_idx)

            for c in candidates:
                w = weights.get(c["symbol"], 0.0)
                all_records.append({
                    "date": c["date"],
                    "symbol": c["symbol"],
                    "momentum_score": round(c["momentum_score"], 4),
                    "qualified": c["qualified"],
                    "in_portfolio": c["symbol"] in active_symbols,
                    "weight": round(w, 4),
                })

        result = pd.DataFrame(all_records)
        if not result.empty:
            result["date"] = pd.to_datetime(result["date"])
        return result

    def _get_rebalance_dates(self, features: dict[str, pd.DataFrame]) -> list[int]:
        first_df = next(iter(features.values()))
        idx = first_df.index
        freq = self.config.rebalance_freq
        period_freq = {"ME": "M", "QE": "Q", "YE": "Y"}.get(freq, freq)
        if freq == "W":
            groups = idx.isocalendar().week
        else:
            groups = idx.to_period(period_freq)
        rebalance_indices = []
        for period in groups.unique():
            mask = groups == period
            valid = np.where(mask)[0]
            valid = valid[valid >= self.config.ma_long]
            if len(valid) > 0:
                rebalance_indices.append(int(valid[-1]))
        return sorted(rebalance_indices)

    def _compute_portfolio_weights(
        self,
        symbols: list[str],
        candidates: list[dict],
        features: dict[str, pd.DataFrame],
        idx: int,
    ) -> dict[str, float]:
        n = len(symbols)
        if n == 0:
            return {}
        w = 1.0 / n
        weights = {s: w for s in symbols}
        total = sum(weights.values())
        if total > 0:
            scale = min(1.0, self.config.max_leverage / total) if total > 0 else 1.0
            for s in weights:
                weights[s] *= scale
        return weights


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WFPeriod:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class WFResult:
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    periods: list[WFPeriod]


class TSMOMWalkForward:
    """Walk-forward validation with expanding/rolling windows, no lookahead."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def generate_periods(
        self,
        n_observations: int,
        n_train: int = 756,
        n_test: int = 252,
        expanding: bool = True,
    ) -> list[WFPeriod]:
        periods: list[WFPeriod] = []
        train_end = n_train
        while train_end + n_test <= n_observations:
            test_start = train_end
            test_end = min(test_start + n_test, n_observations)
            train_start = 0 if expanding else train_end - n_train
            periods.append(WFPeriod(
                train_start=max(0, train_start),
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            train_end += n_test
        return periods

    def validate(
        self,
        prices: dict[str, pd.DataFrame],
        strategy_fn: Any,
        config_overrides: dict | None = None,
    ) -> WFResult:
        cfg = self.config
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(cfg, k, v)

        first_df = next(iter(prices.values()))
        n = len(first_df)
        periods = self.generate_periods(n)

        all_train_metrics: list[dict[str, float]] = []
        all_test_metrics: list[dict[str, float]] = []

        for period in periods:
            train_prices = {s: df.iloc[period.train_start:period.train_end] for s, df in prices.items()}
            test_prices = {s: df.iloc[period.test_start:period.test_end] for s, df in prices.items()}

            engine = TSMOMBacktestEngine(config=cfg)
            train_result = engine.run(train_prices)
            test_result = engine.run(test_prices)

            if train_result:
                all_train_metrics.append(train_result.summary_metrics)
            if test_result:
                all_test_metrics.append(test_result.summary_metrics)

        def avg_metrics(metrics_list: list[dict]) -> dict[str, float]:
            if not metrics_list:
                return {}
            keys = metrics_list[0].keys()
            return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

        return WFResult(
            train_metrics=avg_metrics(all_train_metrics),
            test_metrics=avg_metrics(all_test_metrics),
            periods=periods,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TSMOMBacktestResult:
    equity_curve: pd.DataFrame
    drawdown_series: pd.DataFrame
    monthly_returns: pd.DataFrame
    trades: list[dict]
    weights_history: pd.DataFrame
    signals_history: pd.DataFrame
    summary_metrics: dict[str, float]
    benchmark_comparison: dict[str, float] | None = None
    regime_analysis: dict[str, dict[str, float]] | None = None


class TSMOMBacktestEngine:
    """Multi-asset backtest with realistic costs, slippage, and full metrics."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def run(self, prices: dict[str, pd.DataFrame]) -> TSMOMBacktestResult | None:
        if not prices:
            return None

        feat_eng = TSMOMFeatures(self.config)
        features = feat_eng.compute_all(prices)

        trend_filter = TSMOMTrendFilter(self.config)
        trend_filters: dict[str, pd.DataFrame] = {}
        for sym, df in prices.items():
            trend_filters[sym] = trend_filter.apply_filters(
                df, features.get(sym, pd.DataFrame())
            )

        sig_gen = TSMOMSignalGenerator(self.config)
        signals = sig_gen.generate_signals(features, trend_filters)

        portfolio_builder = TSMOMPortfolioConstruction(self.config)
        portfolio_df = portfolio_builder.build_portfolio(features, signals, trend_filters)

        equity, trades = self._simulate(prices, portfolio_df)

        metrics = self._compute_metrics(equity, trades)

        return TSMOMBacktestResult(
            equity_curve=equity,
            drawdown_series=self._compute_drawdown(equity),
            monthly_returns=self._monthly_returns(equity),
            trades=trades,
            weights_history=portfolio_df,
            signals_history=portfolio_df,
            summary_metrics=metrics,
        )

    def _simulate(
        self, prices: dict[str, pd.DataFrame], portfolio_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[dict]]:
        all_dates = sorted(set(
            pd.Timestamp(d) for df in prices.values() for d in df.index
        ))

        equity_series: list[dict] = []
        trades: list[dict] = []

        if portfolio_df.empty:
            empty = pd.DataFrame(equity_series)
            if empty.empty:
                empty = pd.DataFrame({"date": [all_dates[0]], "equity": [self.config.initial_capital]})
            return empty, trades

        portfolio_dates = portfolio_df["date"].unique()
        cash = self.config.initial_capital
        positions: dict[str, float] = {}
        last_weights: dict[str, float] = {}
        last_rebalance_date = None

        for current_date in all_dates:
            current_date = pd.Timestamp(current_date)

            close_prices: dict[str, float] = {}
            for sym, df in prices.items():
                recent = df.loc[df.index <= current_date]
                if not recent.empty:
                    close_prices[sym] = float(recent.iloc[-1]["close"])

            if current_date in portfolio_dates:

                row = portfolio_df[portfolio_df["date"] == current_date]
                new_weights: dict[str, float] = {}
                for _, r in row.iterrows():
                    sym = r["symbol"]
                    w = r["weight"]
                    new_weights[sym] = w
                    if sym not in close_prices:
                        close_prices[sym] = self._last_price(prices, sym, current_date)

                if last_rebalance_date is not None:
                    turnover = self._compute_turnover(last_weights, new_weights)
                else:
                    turnover = 1.0

                portfolio_value = cash + sum(
                    positions.get(s, 0.0) * close_prices.get(s, 0.0) for s in close_prices
                )
                total_cost = turnover * portfolio_value * (
                    self.config.transaction_cost_pct + self.config.slippage_pct
                )

                cash = max(0.0, portfolio_value - total_cost)
                positions = {}
                for sym, w in new_weights.items():
                    if w > 0 and sym in close_prices and close_prices[sym] > 0:
                        allocated = cash * w
                        qty = allocated / close_prices[sym]
                        positions[sym] = qty
                    else:
                        positions[sym] = 0.0

                last_weights = new_weights.copy()
                last_rebalance_date = current_date

                if turnover > 0.01:
                    trades.append({
                        "date": str(current_date.date()),
                        "turnover": round(turnover, 4),
                        "cost": round(total_cost, 2),
                    })

            portfolio_value = cash + sum(
                positions.get(s, 0.0) * close_prices.get(s, 0.0) for s in close_prices
            )

            equity_series.append({
                "date": current_date,
                "equity": round(portfolio_value, 2),
                "cash": round(cash, 2),
            })

        result = pd.DataFrame(equity_series)
        return result, trades

    def _last_price(self, prices: dict[str, pd.DataFrame], sym: str, date: pd.Timestamp) -> float:
        df = prices.get(sym)
        if df is None or df.empty:
            return 0.0
        before = df.loc[df.index <= date]
        if before.empty:
            return float(df.iloc[0]["close"])
        return float(before.iloc[-1]["close"])

    def _compute_turnover(self, old_w: dict[str, float], new_w: dict[str, float]) -> float:
        all_syms = set(old_w.keys()) | set(new_w.keys())
        turnover = 0.0
        for s in all_syms:
            turnover += abs(new_w.get(s, 0.0) - old_w.get(s, 0.0))
        return turnover / 2.0

    def _compute_drawdown(self, equity: pd.DataFrame) -> pd.DataFrame:
        if equity.empty or "equity" not in equity.columns:
            return pd.DataFrame()
        dd = equity.copy()
        dd["peak"] = dd["equity"].cummax()
        dd["drawdown"] = (dd["equity"] - dd["peak"]) / dd["peak"] * 100
        return dd[["date", "drawdown"]]

    def _monthly_returns(self, equity: pd.DataFrame) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame()
        monthly = equity.set_index("date").resample("ME")["equity"].last()
        monthly_ret = monthly.pct_change().dropna()
        return monthly_ret.to_frame("monthly_return")

    def _compute_metrics(self, equity: pd.DataFrame, trades: list[dict]) -> dict[str, float]:
        if equity.empty or len(equity) < 2:
            return {
                "total_return_pct": 0.0,
                "annual_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "volatility_pct": 0.0,
                "win_rate_pct": 0.0,
                "num_trades": 0,
                "turnover": 0.0,
            }

        eq = equity.set_index("date")
        eq = eq[~eq.index.duplicated(keep="first")].sort_index()
        daily_ret = eq["equity"].pct_change().dropna()

        total_return = (eq["equity"].iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        n_days = len(daily_ret)
        ann_factor = TRADING_DAYS_PER_YEAR / max(n_days, 1)
        annual_return = (1 + total_return) ** ann_factor - 1

        sharpe = 0.0
        if daily_ret.std() > 1e-10:
            excess = daily_ret - self.config.risk_free_rate / TRADING_DAYS_PER_YEAR
            sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

        neg_ret = daily_ret.clip(upper=0)
        sortino = 0.0
        if neg_ret.std() > 1e-10:
            excess = daily_ret - self.config.risk_free_rate / TRADING_DAYS_PER_YEAR
            sortino = (excess.mean() / neg_ret.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

        peak = eq["equity"].cummax()
        drawdown = (eq["equity"] - peak) / peak
        max_dd = float(drawdown.min())

        calmar = 0.0
        if abs(max_dd) > 1e-10:
            calmar = annual_return / abs(max_dd)

        volatility = float(daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        num_trades = len(trades)
        turnover = sum(t.get("turnover", 0.0) for t in trades) / max(n_days, 1) * TRADING_DAYS_PER_YEAR

        return {
            "total_return_pct": round(float(total_return) * 100, 2),
            "annual_return_pct": round(float(annual_return) * 100, 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "calmar_ratio": round(float(calmar), 4),
            "max_drawdown_pct": round(float(max_dd) * 100, 2),
            "volatility_pct": round(float(volatility) * 100, 2),
            "win_rate_pct": 0.0,
            "num_trades": num_trades,
            "turnover": round(float(turnover), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9 — BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMBenchmark:
    """Compares strategy performance against benchmark assets."""

    BENCHMARKS = ["SPY", "QQQ"]

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def compare(
        self, strategy_result: TSMOMBacktestResult, benchmark_prices: dict[str, pd.DataFrame] | None = None
    ) -> dict[str, dict[str, float]]:
        strat_eq = strategy_result.equity_curve
        if strat_eq.empty:
            return {}

        prices = benchmark_prices
        if prices is None:
            loader = TSMOMDataLoader(self.config)
            prices = loader.load_prices(
                symbols=self.BENCHMARKS,
                start=str(strat_eq["date"].iloc[0].date()),
                end=str(strat_eq["date"].iloc[-1].date()),
            )

        results: dict[str, dict[str, float]] = {}
        strat_returns = self._calculate_returns(strat_eq)

        for bench_sym in self.BENCHMARKS:
            if bench_sym not in prices:
                continue
            bench_df = prices[bench_sym]
            bench_series = strat_eq[["date"]].copy()
            bench_series["equity"] = bench_series["date"].apply(
                lambda d: self._benchmark_value(bench_df, d, self.config.initial_capital)
            )
            bench_returns = self._calculate_returns(bench_series)

            aligned = pd.DataFrame({
                "strat": strat_returns,
                "bench": bench_returns,
            }).dropna()

            if len(aligned) < 2:
                continue

            strat_r = aligned["strat"]
            bench_r = aligned["bench"]
            diff = strat_r - bench_r

            alpha = float(diff.mean() * TRADING_DAYS_PER_YEAR)

            cov = np.cov(strat_r, bench_r)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 1e-10 else 0.0

            te = float(diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            ir = float(diff.mean() / diff.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if diff.std() > 1e-10 else 0.0

            strat_cum = (1 + strat_r).prod() - 1
            bench_cum = (1 + bench_r).prod() - 1

            results[bench_sym] = {
                "alpha": round(alpha * 100, 4),
                "beta": round(beta, 4),
                "information_ratio": round(ir, 4),
                "tracking_error": round(te * 100, 4),
                "strategy_return_pct": round(float(strat_cum) * 100, 2),
                "benchmark_return_pct": round(float(bench_cum) * 100, 2),
                "excess_return_pct": round(float(strat_cum - bench_cum) * 100, 2),
            }

        return results

    def _calculate_returns(self, eq_df: pd.DataFrame) -> pd.Series:
        if eq_df.empty:
            return pd.Series(dtype=float)
        eq = eq_df.set_index("date")["equity"]
        eq = eq[~eq.index.duplicated(keep="first")].sort_index()
        return eq.pct_change().dropna()

    def _benchmark_value(self, bench_df: pd.DataFrame, date: pd.Timestamp, initial_capital: float) -> float:
        date = pd.Timestamp(date)
        before = bench_df.loc[bench_df.index <= date]
        if before.empty:
            return initial_capital
        price = float(before.iloc[-1]["close"])
        first_price = float(bench_df.iloc[0]["close"])
        if first_price <= 0:
            return initial_capital
        return initial_capital * (price / first_price)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10 — REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMRegimeDetector:
    """Classifies market regimes and analyzes strategy performance per regime."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def detect_regimes(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        first_sym = next(iter(prices.keys()))
        df = prices[first_sym]
        close = df["close"]

        regime = pd.DataFrame(index=df.index)
        regime["regime"] = "unknown"

        daily_ret = close.pct_change()
        ma200 = close.rolling(200).mean()
        vol_60 = daily_ret.rolling(60).std()

        above_ma = close > ma200
        vol_high = vol_60 > vol_60.rolling(252).median() * 1.5

        for idx in regime.index:
            loc = regime.index.get_loc(idx)
            if loc < 60:
                continue

            idx_above = above_ma.iloc[loc]
            idx_vol = vol_high.iloc[loc]
            idx_ret_60 = daily_ret.iloc[loc - 60:loc].sum()

            if idx_vol:
                regime.at[idx, "regime"] = "high_volatility"
            elif idx_above and idx_ret_60 > 0.05:
                regime.at[idx, "regime"] = "bull"
            elif not idx_above and idx_ret_60 < -0.05:
                regime.at[idx, "regime"] = "bear"
            else:
                regime.at[idx, "regime"] = "sideways"

        return regime

    def analyze_by_regime(
        self, equity_curve: pd.DataFrame, regime_df: pd.DataFrame
    ) -> dict[str, dict[str, float]]:
        if equity_curve.empty:
            return {}

        merged = equity_curve.copy()
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.set_index("date").join(regime_df["regime"])
        merged["daily_ret"] = merged["equity"].pct_change()

        results: dict[str, dict[str, float]] = {}
        for regime_name in merged["regime"].unique():
            sub = merged[merged["regime"] == regime_name]
            if len(sub) < 5:
                continue
            rets = sub["daily_ret"].dropna()
            cum_ret = (1 + rets).prod() - 1
            results[regime_name] = {
                "days": int(len(sub)),
                "return_pct": round(float(cum_ret) * 100, 2),
                "sharpe": round(float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)), 4) if rets.std() > 1e-10 else 0.0,
                "volatility_pct": round(float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100), 2),
            }

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 11 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMVisualizer:
    """Generates institutional-quality charts for TSMOM analysis."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()
        self._imported = False
        self._import_plots()

    def _import_plots(self) -> None:
        try:
            global plt, mpl, sns
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import seaborn as sns
            self._imported = True
        except ImportError:
            self._imported = False

    def plot_equity_curve(
        self,
        result: TSMOMBacktestResult,
        save_path: str = "tsmom_equity.png",
        show: bool = False,
    ) -> None:
        if not self._imported:
            return
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(pd.to_datetime(result.equity_curve["date"]), result.equity_curve["equity"],
                linewidth=1.5, label="TSMOM Strategy", color="navy")
        ax.set_title("TSMOM Portfolio Equity Curve", fontsize=14, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def plot_drawdown(self, result: TSMOMBacktestResult, save_path: str = "tsmom_drawdown.png", show: bool = False) -> None:
        if not self._imported:
            return
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(pd.to_datetime(result.drawdown_series["date"]), 0,
                         result.drawdown_series["drawdown"], color="crimson", alpha=0.5)
        ax.set_title("TSMOM Drawdown", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def plot_rolling_sharpe(self, features: dict[str, pd.DataFrame], save_path: str = "tsmom_rolling_sharpe.png", show: bool = False) -> None:
        if not self._imported:
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        for sym, df in features.items():
            if "sharpe_126d" in df.columns:
                ax.plot(df.index, df["sharpe_126d"], linewidth=1.0, label=sym, alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axhline(1, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_title("Rolling 126-Day Sharpe Ratio", fontsize=12, fontweight="bold")
        ax.legend(loc="best", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    def plot_allocation(self, weights_history: pd.DataFrame, save_path: str = "tsmom_allocation.png", show: bool = False) -> None:
        if not self._imported or weights_history.empty:
            return
        pivot = weights_history.pivot_table(index="date", columns="symbol", values="weight", aggfunc="first")
        pivot = pivot.fillna(0)
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot.plot.area(ax=ax, stacked=True, linewidth=0.5, alpha=0.8, colormap="tab10")
        ax.set_title("Portfolio Allocation Over Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 12 — PARAMETER RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParameterResearchConfig:
    lookbacks: list[list[int]] = field(default_factory=lambda: [
        [21], [63], [126], [252],
        [21, 63], [63, 126], [126, 252],
        [21, 63, 126, 252],
    ])
    trend_filter_sets: list[list[str]] = field(default_factory=lambda: [
        [],
        ["price_above_ma200"],
        ["ma50_above_ma200"],
        ["price_above_ma200", "ma50_above_ma200"],
        ["price_above_breakout"],
    ])
    rebalance_freqs: list[str] = field(default_factory=lambda: ["W", "ME", "QE"])
    top_n_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    sizing_methods: list[str] = field(default_factory=lambda: ["equal", "vol_target"])


@dataclass
class ResearchResult:
    config: dict[str, Any]
    metrics: dict[str, float]
    benchmark: dict[str, float] | None = None


class TSMOMParameterResearch:
    """Grid search over TSMOM parameters to find optimal configurations."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()

    def run_grid(
        self,
        prices: dict[str, pd.DataFrame],
        research_cfg: ParameterResearchConfig | None = None,
    ) -> list[ResearchResult]:
        rc = research_cfg or ParameterResearchConfig()
        results: list[ResearchResult] = []

        for lookbacks in rc.lookbacks:
            for filters in rc.trend_filter_sets:
                for freq in rc.rebalance_freqs:
                    for top_n in rc.top_n_values:
                        for sizing in rc.sizing_methods:
                            cfg = TSMOMConfig(
                                symbols=self.config.symbols,
                                start=self.config.start,
                                end=self.config.end,
                                momentum_lookbacks=lookbacks,
                                trend_filters=filters,
                                rebalance_freq=freq,
                                top_n=top_n,
                                sizing_method=sizing,
                                initial_capital=self.config.initial_capital,
                                transaction_cost_pct=self.config.transaction_cost_pct,
                            )
                            engine = TSMOMBacktestEngine(cfg)
                            result = engine.run(prices)
                            if result is None:
                                continue

                            benchmark = TSMOMBenchmark(cfg)
                            bench_results = benchmark.compare(result)

                            results.append(ResearchResult(
                                config={
                                    "lookbacks": lookbacks,
                                    "trend_filters": filters,
                                    "rebalance_freq": freq,
                                    "top_n": top_n,
                                    "sizing_method": sizing,
                                },
                                metrics=result.summary_metrics,
                                benchmark=bench_results.get("SPY", None),
                            ))

        return results

    def to_dataframe(self, results: list[ResearchResult]) -> pd.DataFrame:
        rows: list[dict] = []
        for r in results:
            row = {**r.config, **r.metrics}
            if r.benchmark:
                row["alpha"] = r.benchmark.get("alpha")
                row["beta"] = r.benchmark.get("beta")
                row["excess_return"] = r.benchmark.get("excess_return_pct")
            rows.append(row)
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 13 — MACHINE LEARNING EXTENSION
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMForecastModel:
    """ML models predicting trend continuation or failure — enhances, not replaces momentum."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()
        self.model = None
        self.feature_columns: list[str] = []

    def prepare_features(
        self, features: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.Series]:
        frames: list[pd.DataFrame] = []
        targets: list[pd.Series] = []

        for sym, df in features.items():
            close = df["close"]
            fwd_return = close.pct_change(21).shift(-21)
            target = (fwd_return > 0).astype(int)

            feat_cols = [
                "momentum_21d", "momentum_63d", "momentum_126d", "momentum_252d",
                "vol_20d", "vol_60d", "atr", "drawdown",
                "sharpe_21d", "sharpe_63d", "sharpe_126d", "sharpe_252d",
                "sortino_21d", "sortino_63d", "sortino_126d", "sortino_252d",
            ]
            available = [c for c in feat_cols if c in df.columns]
            feat = df[available].copy()
            feat["symbol"] = sym
            feat = feat.dropna()
            if feat.empty:
                continue

            aligned_target = target.loc[feat.index]
            aligned_target = aligned_target.dropna()
            feat = feat.loc[aligned_target.index]
            if feat.empty:
                continue

            frames.append(feat)
            targets.append(aligned_target)

        if not frames:
            return pd.DataFrame(), pd.Series(dtype=float)

        X = pd.concat(frames)
        y = pd.concat(targets)
        self.feature_columns = [c for c in X.columns if c != "symbol"]
        return X[self.feature_columns], y

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Any:
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            self.model.fit(X, y)
            return self.model
        except ImportError:
            print("[TSMOM-ML] xgboost not available")
            return None

    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Any:
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X, y)
            return self.model
        except ImportError:
            print("[TSMOM-ML] sklearn not available")
            return None

    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series) -> Any:
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(X, y)
            return self.model
        except ImportError:
            print("[TSMOM-ML] lightgbm not available")
            return None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def augment_signal(
        self, momentum_score: np.ndarray, ml_prob: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        combined = momentum_score.copy()
        ml_signal = (ml_prob > threshold).astype(float)
        combined = combined * ml_signal
        return combined


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE RESEARCH FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════


class TSMOMResearchFramework:
    """End-to-end TSMOM research platform combining all phases."""

    def __init__(self, config: TSMOMConfig | None = None):
        self.config = config or TSMOMConfig()
        self.data_loader = TSMOMDataLoader(self.config)
        self.feature_engine = TSMOMFeatures(self.config)
        self.trend_filter = TSMOMTrendFilter(self.config)
        self.signal_generator = TSMOMSignalGenerator(self.config)
        self.portfolio_builder = TSMOMPortfolioConstruction(self.config)
        self.backtest_engine = TSMOMBacktestEngine(self.config)
        self.benchmark = TSMOMBenchmark(self.config)
        self.regime_detector = TSMOMRegimeDetector(self.config)
        self.visualizer = TSMOMVisualizer(self.config)
        self.forecast_model = TSMOMForecastModel(self.config)
        self.walk_forward = TSMOMWalkForward(self.config)
        self.parameter_research = TSMOMParameterResearch(self.config)

        self.prices: dict[str, pd.DataFrame] = {}
        self.features: dict[str, pd.DataFrame] = {}
        self.result: TSMOMBacktestResult | None = None

    def load_data(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        self.prices = self.data_loader.load_prices(symbols, start, end, use_cache)
        return self.prices

    def compute_features(self) -> dict[str, pd.DataFrame]:
        self.features = self.feature_engine.compute_all(self.prices)
        return self.features

    def run_backtest(self) -> TSMOMBacktestResult | None:
        if not self.prices:
            print("[TSMOM] No data loaded. Call load_data() first.")
            return None
        self.compute_features()
        self.result = self.backtest_engine.run(self.prices)
        return self.result

    def run_full_analysis(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        print("\n╔════════════════════════════════════════════════╗")
        print("║   TSMOM Research Framework — Running Analysis  ║")
        print("╚════════════════════════════════════════════════╝\n")

        print(f"[1/8] Loading data for {symbols or self.config.symbols}...")
        self.load_data(symbols, start, end)

        print("[2/8] Computing momentum features...")
        self.compute_features()

        print("[3/8] Running backtest...")
        self.result = self.backtest_engine.run(self.prices)
        if self.result is None:
            print("  ERROR: Backtest failed.")
            return {}

        print("\n  ── Summary Metrics ──")
        for k, v in self.result.summary_metrics.items():
            print(f"    {k}: {v}")

        print("\n[4/8] Benchmarking against SPY/QQQ...")
        bench = None
        if self.prices:
            bench = self.benchmark.compare(self.result)
            if bench:
                for bname, bmetrics in bench.items():
                    print(f"  vs {bname}: Alpha={bmetrics.get('alpha', 'N/A')}, "
                          f"Beta={bmetrics.get('beta', 'N/A')}, "
                          f"IR={bmetrics.get('information_ratio', 'N/A')}")

        print("[5/8] Detecting market regimes...")
        regime = None
        if self.prices:
            regime_df = self.regime_detector.detect_regimes(self.prices)
            if self.result:
                regime = self.regime_detector.analyze_by_regime(
                    self.result.equity_curve, regime_df
                )
                if regime:
                    for rname, rmetrics in regime.items():
                        print(f"  {rname}: {rmetrics}")

        print("[6/8] Running walk-forward validation...")
        wf_result = None
        if self.prices:
            wf_result = self.walk_forward.validate(self.prices, None)
            if wf_result:
                print(f"  Train Sharpe: {wf_result.train_metrics.get('sharpe_ratio', 'N/A')}, "
                      f"Test Sharpe: {wf_result.test_metrics.get('sharpe_ratio', 'N/A')}")

        print("[7/8] Training ML forecast model...")
        ml_result = None
        if self.features:
            X, y = self.forecast_model.prepare_features(self.features)
            if not X.empty:
                self.forecast_model.train_xgboost(X, y)
                if self.forecast_model.model is not None:
                    from sklearn.metrics import accuracy_score, roc_auc_score
                    y_pred = self.forecast_model.model.predict(X)
                    try:
                        y_prob = self.forecast_model.model.predict_proba(X)[:, 1]
                        auc = roc_auc_score(y, y_prob)
                    except Exception:
                        auc = 0.0
                    acc = accuracy_score(y, y_pred)
                    ml_result = {"accuracy": round(float(acc), 4), "auc": round(float(auc), 4)}
                    print(f"  ML Model — Accuracy: {ml_result['accuracy']}, AUC: {ml_result['auc']}")

        print("[8/8] Generating parameter research...")
        param_results = None
        if self.prices:
            param_results = self.parameter_research.run_grid(self.prices)
            if param_results:
                df = self.parameter_research.to_dataframe(param_results)
                best = df.loc[df["sharpe_ratio"].idxmax()] if not df.empty else None
                if best is not None:
                    print(f"\n  Best config (by Sharpe): {best.to_dict()}")

        print("\n╔════════════════════════════════════════════════╗")
        print("║   Analysis Complete                           ║")
        print("╚════════════════════════════════════════════════╝\n")

        return {
            "metrics": self.result.summary_metrics if self.result else {},
            "benchmark": bench,
            "regime_analysis": regime,
            "walk_forward": {
                "train": wf_result.train_metrics if wf_result else {},
                "test": wf_result.test_metrics if wf_result else {},
            } if wf_result else {},
            "ml_performance": ml_result,
            "best_params": best.to_dict() if param_results and best is not None else {},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY INTEGRATION for existing Strategy base class
# ═══════════════════════════════════════════════════════════════════════════════

from backend.strategies.base import Strategy, Signal, Portfolio


class TSMOMStrategy(Strategy):
    """Time-Series Momentum strategy compatible with the existing backtest framework.

    Operates on a single symbol using momentum signals with trend filters.
    For multi-asset portfolio construction use TSMOMResearchFramework.
    """

    def __init__(
        self,
        momentum_lookback: int = 126,
        long_only: bool = True,
        use_trend_filter: bool = True,
        trend_filter_type: str = "price_above_ma200",
        volatility_position_sizing: bool = False,
        target_volatility: float = 0.15,
        signal_type: str = "binary",
        max_hold_bars: int = 0,
        stop_loss_pct: float = 0.0,
    ):
        self.momentum_lookback = momentum_lookback
        self.long_only = long_only
        self.use_trend_filter = use_trend_filter
        self.trend_filter_type = trend_filter_type
        self.volatility_position_sizing = volatility_position_sizing
        self.target_volatility = target_volatility
        self.signal_type = signal_type
        self.max_hold_bars = max_hold_bars
        self.stop_loss_pct = stop_loss_pct

        self._mom_series: pd.Series | None = None
        self._ma_long: pd.Series | None = None
        self._ma_short: pd.Series | None = None
        self._vol_series: pd.Series | None = None
        self._entry_bar = -1
        self._peak_price = 0.0
        self._has_position = False
        self.buy_size: float = 1000.0
        self.sell_portion: float = 100.0
        self._symbol = ""

    def init(self, data: pd.DataFrame) -> None:
        close = data["close"]
        lb = self.momentum_lookback
        self._mom_series = close.pct_change(lb)
        self._ma_long = close.rolling(200).mean()
        self._ma_short = close.rolling(50).mean()
        self._vol_series = close.pct_change().rolling(20).std() * np.sqrt(252)

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if i < max(self.momentum_lookback, 200):
            return Signal.HOLD

        close = float(data.iloc[i]["close"])
        momentum = float(self._mom_series.iloc[i])
        symbol = self._symbol or "ASSET"
        has_pos = self._has_position or portfolio.positions.get(symbol, 0) > 0

        if has_pos:
            self._peak_price = max(self._peak_price, close)

            if self.max_hold_bars > 0 and self._entry_bar >= 0 and i - self._entry_bar >= self.max_hold_bars:
                return self._exit(portfolio, symbol, close)

            if self.stop_loss_pct > 0 and self._peak_price > 0:
                dd = (close - self._peak_price) / self._peak_price
                if dd <= -self.stop_loss_pct:
                    return self._exit(portfolio, symbol, close)

        qualified = True
        if self.use_trend_filter:
            qualified = self._check_filter(i, data)

        if not qualified and has_pos:
            return self._exit(portfolio, symbol, close)
        if not qualified:
            return Signal.HOLD

        if self.signal_type == "binary":
            signal_buy = momentum > 0
        elif self.signal_type == "weighted":
            signal_buy = momentum > 0.01
        else:
            signal_buy = momentum > 0

        if self.long_only:
            if signal_buy and not has_pos:
                self._entry_bar = i
                self._peak_price = close
                self._has_position = True
                self.buy_size = self._calc_position_size(portfolio)
                return Signal.BUY
            if not signal_buy and has_pos:
                return self._exit(portfolio, symbol, close)
        else:
            if signal_buy and not has_pos:
                self._entry_bar = i
                self._peak_price = close
                self._has_position = True
                self.buy_size = self._calc_position_size(portfolio)
                return Signal.BUY
            if not signal_buy and not has_pos:
                return Signal.SELL
            if not signal_buy and has_pos:
                return self._exit(portfolio, symbol, close)

        return Signal.HOLD

    def _check_filter(self, i: int, data: pd.DataFrame) -> bool:
        ft = self.trend_filter_type
        close = data.iloc[i]["close"]

        if ft == "price_above_ma200":
            return bool(close > self._ma_long.iloc[i])
        if ft == "ma50_above_ma200":
            return bool(self._ma_short.iloc[i] > self._ma_long.iloc[i])
        if ft == "price_above_breakout":
            close_series = data["close"]
            breakout = close_series.iloc[max(0, i - 252):i].max() if i >= 252 else close_series.iloc[:i].max()
            return bool(close > breakout)
        return True

    def _calc_position_size(self, portfolio: Portfolio) -> float:
        if not self.volatility_position_sizing or self._vol_series is None:
            return portfolio.cash * 0.95
        vol = float(self._vol_series.iloc[-1]) if len(self._vol_series) > 0 else 0.2
        if vol <= 0:
            return portfolio.cash * 0.95
        fraction = min(self.target_volatility / vol, 1.0)
        return portfolio.cash * fraction

    def _exit(self, portfolio: Portfolio, symbol: str, price: float) -> str:
        self._entry_bar = -1
        self._peak_price = 0.0
        self._has_position = False
        return Signal.SELL

    def on_trade(self, side: str, symbol: str, qty: float, price: float) -> None:
        pass
