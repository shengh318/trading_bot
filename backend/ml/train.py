"""CLI training script for multi-symbol ML trading model.

Usage:
    python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20
    python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --walk-forward 3 --beat-baselines
    python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --grid-search --kelly --auto-threshold

Compares against SmaCrossover and SimpleStrat 1 baselines.
With --beat-baselines, only saves model if it outperforms both.
With --grid-search, tries many hyperparameter combinations automatically.
With --walk-forward N, uses N expanding-window folds for validation.
"""

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # ImportError or XGBoostError (missing libomp)
    XGBClassifier = None  # type: ignore
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    LGBMClassifier = None  # type: ignore
    HAS_LGB = False

from backend.ml.features import compute_features, get_feature_columns
from backend.ml.model import save_model, ModelMetadata, MODELS_DIR
from backend.strategies.base import Portfolio, Signal
from backend.strategies.sma_crossover import SmaCrossover
from backend.strategies.simple_strat_1 import SimpleStrat1


@dataclass
class KellyState:
    """Tracks running win/loss statistics for Kelly Criterion sizing."""
    wins: float = 0.0
    losses: float = 0.0
    total_win_pct: float = 0.0
    total_loss_pct: float = 0.0
    count: int = 0

    def update(self, pnl_pct: float) -> None:
        self.count += 1
        if pnl_pct > 0:
            self.wins += 1
            self.total_win_pct += pnl_pct
        else:
            self.losses += 1
            self.total_loss_pct += abs(pnl_pct)

    def kelly_fraction(self, prob_up: float) -> float:
        """Return fraction of capital to risk using Kelly Criterion."""
        if self.count < 3:
            return 0.5 * (2 * prob_up - 1)
        avg_win = self.total_win_pct / max(self.wins, 1)
        avg_loss = self.total_loss_pct / max(self.losses, 1)
        b = avg_win / max(avg_loss, 0.001)
        if b <= 0:
            return 0.0
        p = prob_up
        q = 1 - p
        kelly = max(b * p - q, 0.0) / b
        return float(np.clip(kelly, 0.0, 0.25))


# ── Data ──────────────────────────────────────────────────────────────────

def download_data(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV from Yahoo Finance for each symbol."""
    end = datetime.now()
    start = end - timedelta(days=int(years * 365.25) + 10)

    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        print(f"  Downloading {sym}...")
        ticker = yf.Ticker(sym)
        df = ticker.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            print(f"  \u26a0 No data for {sym}, skipping")
            continue
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        df.index.name = "timestamp"
        data[sym] = df[list(col_map.values())]
        print(f"    {len(df)} bars  ({df.index[0].date()} \u2192 {df.index[-1].date()})")
    return data


def prepare_features(
    data_dict: dict[str, pd.DataFrame],
    labeling: str = "next_bar",
    triple_barrier_pct: float = 0.02,
    triple_barrier_max_bars: int = 10,
    forecast_horizon: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute features for all symbols, combine into a single DataFrame.

    Args:
        data_dict: Maps symbol -> OHLCV DataFrame.
        labeling: ``"next_bar"`` (binary up/down) or ``"triple_barrier"``.
        triple_barrier_pct: Profit target / stop-loss percentage.
        triple_barrier_max_bars: Max holding period for triple barrier.
        forecast_horizon: Number of bars forward for the binary target.
    """
    all_dfs: list[pd.DataFrame] = []
    for sym, df in data_dict.items():
        d = compute_features(df.copy())
        d["symbol"] = sym

        if labeling == "triple_barrier":
            d["target"] = _triple_barrier_label(
                d, pct=triple_barrier_pct, max_bars=triple_barrier_max_bars,
            )
        else:
            d["target"] = (d["close"].shift(-forecast_horizon) > d["close"]).astype(int)

        all_dfs.append(d)

    combined = pd.concat(all_dfs)
    feature_cols = get_feature_columns(combined)
    clean = combined.dropna(subset=feature_cols + ["target"])
    if len(clean) < 100:
        raise ValueError(
            f"Only {len(clean)} clean rows after dropping NaN \u2014 need at least 100"
        )
    clean = clean.sort_index()
    return clean, feature_cols


def _triple_barrier_label(
    df: pd.DataFrame,
    pct: float = 0.02,
    max_bars: int = 10,
) -> pd.Series:
    """Triple-barrier labeling (de Prado).

    1 = profit target hit first,
    0 = stop-loss hit first,
    0 = max holding period expired.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for i in range(n - 1):
        entry = close[i]
        tp = entry * (1 + pct)
        sl = entry * (1 - pct)
        end = min(i + max_bars + 1, n)
        for j in range(i + 1, end):
            if high[j] >= tp:
                labels[i] = 1
                break
            if low[j] <= sl:
                labels[i] = 0
                break
    return pd.Series(labels, index=df.index)


def train_val_split(
    df: pd.DataFrame,
    val_split: float = 0.8,
    cutoff_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation split."""
    if cutoff_date is not None:
        cutoff_dt = pd.Timestamp(cutoff_date)
        if df.index.tz is not None:
            cutoff_dt = cutoff_dt.tz_localize(df.index.tz)
        train = df[df.index < cutoff_dt].reset_index(drop=True)
        val = df[df.index >= cutoff_dt].reset_index(drop=True)
    else:
        split_idx = int(len(df) * val_split)
        cutoff_dt = df.index[split_idx]
        train = df[df.index < cutoff_dt].reset_index(drop=True)
        val = df[df.index >= cutoff_dt].reset_index(drop=True)
    return train, val


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window walk-forward folds.

    Returns list of (train, val) DataFrames.
    """
    symbols = df["symbol"].unique()
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for fold in range(1, n_folds + 1):
        cutoff = fold / n_folds
        all_train: list[pd.DataFrame] = []
        all_val: list[pd.DataFrame] = []
        for sym in symbols:
            sym_df = df[df["symbol"] == sym].sort_index()
            idx = int(len(sym_df) * cutoff)
            min_train = 50
            min_val = 20
            if idx < min_train or len(sym_df) - idx < min_val:
                continue
            all_train.append(sym_df.iloc[:idx].reset_index(drop=True))
            all_val.append(sym_df.iloc[idx:].reset_index(drop=True))
        if all_train and all_val:
            folds.append((
                pd.concat(all_train).reset_index(drop=True),
                pd.concat(all_val).reset_index(drop=True),
            ))
    return folds


# ── Backtest helpers ──────────────────────────────────────────────────────

def backtest_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    confidence_threshold: float,
    base_buy_size: float,
    initial_cash: float = 10000,
    use_kelly: bool = False,
    max_hold_bars: int = 30,
    trailing_stop_pct: float = 0.05,
) -> dict:
    """Backtest using a trained model with shared portfolio across symbols.

    Mirrors ``MultiSymbolBacktestEngine`` — all symbols share one cash pool,
    equity is computed per-timestamp across all open positions.
    """
    symbols = df["symbol"].unique().tolist()

    # Group data by symbol, keep original datetime index
    sym_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        sd = df[df["symbol"] == sym].sort_index()
        if not sd.empty:
            sym_data[sym] = sd

    if not sym_data:
        return _calc_metrics([initial_cash], [], initial_cash)

    portfolio = Portfolio(initial_cash)
    kelly = KellyState() if use_kelly else None

    entry_bar: dict[str, int] = {}
    peak_price: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    local_idx: dict[str, int] = {}
    for sym in sym_data:
        entry_bar[sym] = -1
        peak_price[sym] = 0.0
        cost_basis[sym] = 0.0
        local_idx[sym] = -1

    timestamps = sorted(set.union(*[set(d.index) for d in sym_data.values()]))
    all_trades: list[dict] = []
    snapshots: list[float] = []

    for ts in timestamps:
        for sym in sorted(sym_data.keys()):
            if ts not in sym_data[sym].index:
                continue
            local_idx[sym] += 1
            i = local_idx[sym]
            row = sym_data[sym].iloc[i]
            price = float(row["close"])

            feat_df = pd.DataFrame([row[feature_cols]])
            if feat_df.isna().any(axis=None):
                signal = Signal.HOLD
                prob_up = 0.5
            else:
                proba = model.predict_proba(feat_df)[0]
                if len(proba) >= 2:
                    prob_up = proba[1]
                    if prob_up >= confidence_threshold:
                        signal = Signal.BUY
                    elif prob_up <= (1 - confidence_threshold):
                        signal = Signal.SELL
                    else:
                        signal = Signal.HOLD
                else:
                    signal = Signal.HOLD
                    prob_up = 0.5

            has_pos = portfolio.positions.get(sym, 0) > 0

            # Exits override signal
            if has_pos:
                peak_price[sym] = max(peak_price[sym], price)
                if max_hold_bars > 0 and entry_bar[sym] >= 0 and i - entry_bar[sym] >= max_hold_bars:
                    signal = Signal.EXIT
                elif trailing_stop_pct > 0 and peak_price[sym] > 0:
                    dd = (price - peak_price[sym]) / peak_price[sym]
                    if dd <= -trailing_stop_pct:
                        signal = Signal.EXIT

            if signal == Signal.BUY and portfolio.cash > 0:
                if use_kelly and kelly is not None:
                    fraction = kelly.kelly_fraction(prob_up)
                    amt = min(portfolio.cash * fraction, portfolio.cash)
                else:
                    if prob_up < 0.65:
                        mult = 0.5
                    elif prob_up < 0.75:
                        mult = 1.0
                    elif prob_up < 0.85:
                        mult = 1.5
                    else:
                        mult = 2.0
                    amt = min(base_buy_size * mult, portfolio.cash)

                qty = amt / price
                if qty > 0:
                    cost = qty * price
                    portfolio.cash -= cost
                    cost_basis[sym] += cost
                    portfolio.positions[sym] = portfolio.positions.get(sym, 0) + qty
                    entry_bar[sym] = i
                    peak_price[sym] = price
                    all_trades.append({"side": "buy", "qty": qty, "price": price})

            elif signal in (Signal.SELL, Signal.EXIT) and has_pos:
                qty = portfolio.positions[sym]
                proceeds = qty * price
                portfolio.cash += proceeds
                portfolio.positions[sym] = 0
                pnl = proceeds - cost_basis[sym]
                if use_kelly and kelly is not None and cost_basis[sym] > 0:
                    kelly.update(pnl / cost_basis[sym])
                cost_basis[sym] = 0.0
                entry_bar[sym] = -1
                peak_price[sym] = 0.0
                all_trades.append({"side": "sell", "qty": qty, "price": price, "pnl": pnl})

        equity = portfolio.cash + sum(
            portfolio.positions.get(sym, 0) * float(sym_data[sym].loc[ts, "close"])
            for sym in sym_data
            if ts in sym_data[sym].index and portfolio.positions.get(sym, 0) > 0
        )
        snapshots.append(equity)

    return _calc_metrics(snapshots, all_trades, initial_cash)


def backtest_baseline(
    df: pd.DataFrame,
    strategy_class,
    params: dict,
    initial_cash: float = 10000,
    buy_size: float | None = None,
) -> dict:
    """Simplified backtest for rule-based strategies, run per-symbol."""
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else ["ASSET"]
    n_syms = len(symbols)
    cash_per_sym = initial_cash / n_syms

    all_trades: list[dict] = []
    all_equities: list[list[float]] = []

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].copy().reset_index(drop=True)
        if len(sym_df) == 0:
            continue

        strategy = strategy_class(**params)
        strategy._symbol = sym
        strategy.init(sym_df)

        portfolio = Portfolio(cash_per_sym)
        trades: list[dict] = []
        snapshots: list[float] = []
        cost_basis = 0.0

        for i in range(len(sym_df)):
            row = sym_df.iloc[i]
            price = float(row["close"])
            signal = strategy.next(i, sym_df, portfolio)

            if signal == Signal.BUY and portfolio.cash > 0:
                amt = buy_size if buy_size is not None else getattr(strategy, "buy_size", None)
                buy_cash = min(amt, portfolio.cash) if amt else portfolio.cash
                qty = buy_cash / price
                if qty > 0:
                    cost = qty * price
                    portfolio.cash -= cost
                    cost_basis += cost
                    existing = portfolio.positions.get(sym, 0)
                    portfolio.positions[sym] = existing + qty
                    trades.append({"side": "buy", "qty": qty, "price": price})

            elif signal in (Signal.SELL, Signal.EXIT) and portfolio.positions.get(sym, 0) > 0:
                qty = portfolio.positions[sym]
                proceeds = qty * price
                portfolio.cash += proceeds
                portfolio.positions[sym] = 0
                pnl = proceeds - cost_basis
                cost_basis = 0.0
                trades.append({"side": "sell", "qty": qty, "price": price, "pnl": pnl})

            equity = portfolio.cash + portfolio.positions.get(sym, 0) * price
            snapshots.append(equity)

        all_trades.extend(trades)
        all_equities.append(snapshots)

    min_len = min(len(e) for e in all_equities) if all_equities else 0
    merged_equity: list[float] = (
        [sum(eq[j] for eq in all_equities) for j in range(min_len)]
        if all_equities else [initial_cash]
    )

    return _calc_metrics(merged_equity, all_trades, initial_cash)


def _calc_metrics(
    snapshots: list[float],
    trades: list[dict],
    initial_cash: float,
) -> dict:
    final = snapshots[-1]
    total_return = ((final - initial_cash) / initial_cash) * 100

    returns = pd.Series(snapshots).pct_change().dropna()
    sharpe = round(
        (returns.mean() / returns.std() * np.sqrt(252))
        if returns.std() > 0 else 0, 2)

    peak = snapshots[0]
    max_dd = 0.0
    for eq in snapshots:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)

    sells = [t for t in trades if t["side"] == "sell"]
    wins = sum(1 for t in sells if t.get("pnl", 0) > 0)

    return {
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round((wins / max(len(sells), 1)) * 100, 1),
        "num_trades": len(sells),
        "final_equity": round(final, 2),
    }


# ── Confidence threshold tuning ───────────────────────────────────────────

def tune_confidence_threshold(
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model,
    base_buy_size: float,
    use_kelly: bool = False,
    thresholds: list[float] | None = None,
    max_hold_bars: int = 30,
    trailing_stop_pct: float = 0.05,
) -> tuple[float, dict]:
    """Find the confidence threshold that maximizes Sharpe on validation data."""
    if thresholds is None:
        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]

    best_thresh = thresholds[0]
    best_res: dict | None = None

    for thresh in thresholds:
        res = backtest_model(
            val_df, feature_cols, model, thresh, base_buy_size,
            use_kelly=use_kelly,
            max_hold_bars=max_hold_bars,
            trailing_stop_pct=trailing_stop_pct,
        )
        if best_res is None or res["sharpe_ratio"] > best_res["sharpe_ratio"]:
            best_thresh = thresh
            best_res = res

    return best_thresh, best_res


# ── Comparison ────────────────────────────────────────────────────────────

def print_comparison(
    results: list[tuple[str, dict]],
) -> None:
    """Print the strategy comparison table."""
    print("\n" + "=" * 84)
    print("  STRATEGY COMPARISON  (Validation Period)")
    print("=" * 84)
    print(f"{'Strategy':<28} {'Return':>8} {'Sharpe':>8} {'Win Rate':>9} "
          f"{'Max DD':>8} {'Trades':>7}")
    print("-" * 84)
    for name, r in results:
        print(
            f"{name:<28} {r['total_return_pct']:>7.1f}% "
            f"{r['sharpe_ratio']:>7.2f} "
            f"{r['win_rate_pct']:>7.1f}% "
            f"{r['max_drawdown_pct']:>7.1f}% "
            f"{r['num_trades']:>7}"
        )
    print("=" * 84)


def beats_baselines(ml_res: dict, sma_res: dict, simple_res: dict) -> bool:
    """Check if ML model beats both baselines on return AND Sharpe."""
    beats_ret = (
        ml_res["total_return_pct"] >= sma_res["total_return_pct"]
        and ml_res["total_return_pct"] >= simple_res["total_return_pct"]
    )
    beats_shp = (
        ml_res["sharpe_ratio"] >= sma_res["sharpe_ratio"]
        and ml_res["sharpe_ratio"] >= simple_res["sharpe_ratio"]
    )
    if beats_ret and beats_shp:
        print("\n  \u2705 ML beats BOTH baselines on return AND Sharpe!")
    else:
        if not beats_ret:
            print(
                f"\n  \u274c ML return ({ml_res['total_return_pct']:.1f}%) "
                f"vs SmaCrossover ({sma_res['total_return_pct']:.1f}%) / "
                f"SimpleStrat ({simple_res['total_return_pct']:.1f}%)"
            )
        if not beats_shp:
            print(
                f"  \u274c ML Sharpe ({ml_res['sharpe_ratio']:.2f}) "
                f"vs SmaCrossover ({sma_res['sharpe_ratio']:.2f}) / "
                f"SimpleStrat ({simple_res['sharpe_ratio']:.2f})"
            )
    return beats_ret and beats_shp


# ── Training ──────────────────────────────────────────────────────────────

def train_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    **kwargs,
):
    """Train a single model.

    Supported model types: ``rf``, ``gbt``, ``xgb``, ``lgb``.
    """
    X = train_df[feature_cols]
    y = train_df["target"]

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        model = XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )
    elif model_type == "lgb":
        if not HAS_LGB:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        model = LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=42,
        )
    model.fit(X, y)
    return model


def evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_types: list[str],
    params: dict,
) -> list[tuple[str, object, dict]]:
    """Train + backtest each model type, return (label, model, results) list."""
    base_buy = params.get("base_buy_size", 1000)
    conf = params.get("confidence_threshold", 0.55)
    use_kelly = params.get("use_kelly", False)
    max_hold = params.get("max_hold_bars", 30)
    trail = params.get("trailing_stop_pct", 0.05)
    results: list[tuple[str, object, dict]] = []

    for mt in model_types:
        label = {"rf": "RandomForest", "gbt": "GradientBoosting",
                 "xgb": "XGBoost", "lgb": "LightGBM"}.get(mt, mt.upper())
        print(f"  Training {label} ...")
        model = train_model(train_df, feature_cols, mt, **params)
        print(f"  Backtesting {label} ...")
        res = backtest_model(
            val_df, feature_cols, model, conf, base_buy,
            use_kelly=use_kelly, max_hold_bars=max_hold, trailing_stop_pct=trail,
        )
        results.append((label, model, res))

    return results


def _baseline_results(
    val_df: pd.DataFrame,
) -> dict[str, dict]:
    sma = backtest_baseline(
        val_df, SmaCrossover,
        {"short_window": 10, "long_window": 50},
        buy_size=1000,
    )
    simple = backtest_baseline(
        val_df, SimpleStrat1,
        {"buy_size": 100, "entry_drop": 1.0, "profit_target": 20.0,
         "sell_portion": 100.0, "stop_loss": 3.0, "max_buys": 1},
    )
    return {"SmaCrossover": sma, "SimpleStrat_1": simple}


# ── Grid search ───────────────────────────────────────────────────────────

def _grid_params(model_types: list[str]) -> list[dict]:
    combos: list[dict] = []
    for mt in model_types:
        for n in [100, 200, 300]:
            for d in [5, 8, 12]:
                if mt in ("gbt", "xgb", "lgb"):
                    for lr in [0.05, 0.1, 0.2]:
                        combos.append(dict(model_type=mt, n_estimators=n,
                                           max_depth=d, learning_rate=lr))
                else:
                    combos.append(dict(model_type=mt, n_estimators=n,
                                       max_depth=d))
    return combos


def grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    model_types: list[str],
) -> tuple:
    """Try every hyperparameter combo, return the best model + its results."""
    conf = params.get("confidence_threshold", 0.55)
    base_buy = params.get("base_buy_size", 1000)
    use_kelly = params.get("use_kelly", False)
    max_hold = params.get("max_hold_bars", 30)
    trail = params.get("trailing_stop_pct", 0.05)
    combos = _grid_params(model_types)

    best_model = None
    best_res: dict | None = None
    best_combo: dict | None = None

    for i, c in enumerate(combos):
        mt = c["model_type"]
        mt_label = {"rf": "RF", "gbt": "GBT", "xgb": "XGB", "lgb": "LGB"}.get(mt, mt.upper())
        lr_str = f" lr={c.get('learning_rate', '-')}" if "learning_rate" in c else ""
        print(f"  [{i+1}/{len(combos)}] {mt_label} n={c['n_estimators']} "
              f"depth={c['max_depth']}{lr_str}")

        model = train_model(train_df, feature_cols, **c)
        res = backtest_model(
            val_df, feature_cols, model, conf, base_buy,
            use_kelly=use_kelly,
            max_hold_bars=max_hold,
            trailing_stop_pct=trail,
        )
        print(f"         \u2192 Return={res['total_return_pct']:.1f}%  "
              f"Sharpe={res['sharpe_ratio']:.2f}")

        if best_res is None or res["sharpe_ratio"] > best_res["sharpe_ratio"]:
            best_model = model
            best_res = res
            best_combo = c

    label = {"rf": "RF", "gbt": "GBT", "xgb": "XGB", "lgb": "LGB"}.get(
        best_combo["model_type"], best_combo["model_type"].upper()
    )
    print(f"\n  \u2b50 Grid search winner: {label} n={best_combo['n_estimators']} "
          f"depth={best_combo['max_depth']} "
          f"lr={best_combo.get('learning_rate', '-')}")
    return best_model, best_res, best_combo


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ML trading model and compare against baselines.")
    p.add_argument("--symbols", default="NVDA,AMD,VOO,SPY,META",
                   help="Comma-separated symbols (default: NVDA,AMD,VOO,SPY,META)")
    p.add_argument("--years", type=int, default=20,
                   help="Years of history (default: 20)")
    p.add_argument("--name", default="multi_symbol_model",
                   help="Model name for saving (default: multi_symbol_model)")
    p.add_argument("--model-types", default="rf,gbt",
                   help="Comma-separated model types: rf,gbt,xgb,lgb (default: rf,gbt)")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--confidence-threshold", type=float, default=0.55)
    p.add_argument("--base-buy-size", type=float, default=1000)
    p.add_argument("--cutoff-date", type=str, default=None,
                   help='Explicit train/val boundary (ISO date, e.g. "2023-01-01"). '
                        'Overrides --val-split.')
    p.add_argument("--val-split", type=float, default=0.8,
                   help="Training fraction (default: 0.8, only used without --cutoff-date)")
    p.add_argument("--beat-baselines", action="store_true",
                   help="Only save if ML beats SmaCrossover + SimpleStrat 1")
    p.add_argument("--grid-search", action="store_true",
                   help="Try many hyperparams automatically")
    p.add_argument("--walk-forward", type=int, default=0,
                   help="Use N-fold walk-forward validation (default: 0 = single split)")
    p.add_argument("--kelly", action="store_true",
                   help="Use Kelly Criterion position sizing instead of fixed tiers")
    p.add_argument("--auto-threshold", action="store_true",
                   help="Auto-tune confidence threshold on validation set")
    p.add_argument("--labeling", default="next_bar",
                   choices=["next_bar", "triple_barrier"],
                   help="Labeling method (default: next_bar)")
    p.add_argument("--forecast-horizon", type=int, default=1,
                   help="Number of days forward for target prediction (default: 1)")
    p.add_argument("--max-hold-bars", type=int, default=30,
                   help="Max bars to hold a position before forced exit (default: 30)")
    p.add_argument("--trailing-stop-pct", type=float, default=0.05,
                   help="Trailing stop loss as fraction (default: 0.05 = 5%%)")
    p.add_argument("--model-dir", default=str(MODELS_DIR),
                   help=f"Output dir (default: {MODELS_DIR})")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    model_types = [s.strip() for s in args.model_types.split(",")]

    for mt in model_types:
        if mt not in ("rf", "gbt", "xgb", "lgb"):
            print(f"ERROR: Unknown model type '{mt}'. Choose from: rf, gbt, xgb, lgb")
            sys.exit(1)
        if mt == "xgb" and not HAS_XGB:
            print("ERROR: xgboost not installed. Run: pip install xgboost")
            sys.exit(1)
        if mt == "lgb" and not HAS_LGB:
            print("ERROR: lightgbm not installed. Run: pip install lightgbm")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  ML TRADING MODEL TRAINING")
    print(f"{'='*60}")
    print(f"  Symbols:     {', '.join(symbols)}")
    print(f"  Years:       {args.years}")
    print(f"  Model:       {args.name}")
    print(f"  Types:       {', '.join(model_types)}")
    print(f"  Beat mode:   {'ON' if args.beat_baselines else 'OFF'}")
    print(f"  Grid:        {'ON' if args.grid_search else 'OFF'}")
    print(f"  Walk-forward:{'ON (' + str(args.walk_forward) + ' folds)' if args.walk_forward else 'OFF'}")
    print(f"  Kelly:       {'ON' if args.kelly else 'OFF'}")
    print(f"  Auto-τ:      {'ON' if args.auto_threshold else 'OFF'}")
    print(f"  Labeling:    {args.labeling}")
    print(f"  Horizon:     {args.forecast_horizon}-day")
    print(f"  Max hold:    {args.max_hold_bars} bars")
    print(f"  Trail stop:  {args.trailing_stop_pct*100:.0f}%")
    print(f"{'='*60}\n")

    # ── 1. Download ──
    print("Step 1/5: Downloading data ...")
    data_dict = download_data(symbols, args.years)
    if not data_dict:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    # ── 2. Prepare features ──
    print("\nStep 2/5: Computing features ...")
    full_df, feature_cols = prepare_features(
        data_dict, labeling=args.labeling,
        forecast_horizon=args.forecast_horizon,
    )

    print(f"  Total samples: {len(full_df)}")
    print(f"  Feature columns: {len(feature_cols)}")
    if args.forecast_horizon > 1:
        print(f"  Forecast horizon: {args.forecast_horizon}-day forward return")

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "confidence_threshold": args.confidence_threshold,
        "base_buy_size": args.base_buy_size,
        "use_kelly": args.kelly,
        "max_hold_bars": args.max_hold_bars,
        "trailing_stop_pct": args.trailing_stop_pct,
    }

    # ── 3. Train & backtest ──
    print(f"\nStep 3/5: {'Grid search' if args.grid_search else 'Training'} ...")

    if args.walk_forward:
        folds = walk_forward_folds(full_df, n_folds=args.walk_forward)
        if not folds:
            print("ERROR: No valid walk-forward folds created.")
            sys.exit(1)
        print(f"  Walk-forward: {len(folds)} fold(s)")

        all_ml_labels: list[str] = []
        all_ml_models: list[object] = []
        all_ml_res: list[dict] = []
        all_sma_res: list[dict] = []
        all_simple_res: list[dict] = []

        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            print(f"\n  --- Fold {fold_idx + 1}/{len(folds)} ---")
            print(f"    Train: {len(train_fold)} samples, Val: {len(val_fold)} samples")

            if args.grid_search:
                best_model, best_res, best_combo = grid_search(
                    train_fold, val_fold, feature_cols, params, model_types,
                )
                label = f"GridSearch ({best_combo['model_type']})"
            else:
                results = evaluate_models(
                    train_fold, val_fold, feature_cols, model_types, params,
                )
                best_label, best_model, best_res = max(results, key=lambda x: x[2]["sharpe_ratio"])
                label = best_label

            baselines = _baseline_results(val_fold)
            all_ml_labels.append(label)
            all_ml_models.append(best_model)
            all_ml_res.append(best_res)
            all_sma_res.append(baselines["SmaCrossover"])
            all_simple_res.append(baselines["SimpleStrat_1"])

        # Average results across folds
        def avg_dict(dicts: list[dict]) -> dict:
            keys = dicts[0].keys()
            avg = {}
            for k in keys:
                vals = [d[k] for d in dicts if isinstance(d[k], (int, float))]
                avg[k] = round(sum(vals) / len(vals), 2) if vals else 0
            return avg

        avg_ml = avg_dict(all_ml_res)
        avg_sma = avg_dict(all_sma_res)
        avg_simple = avg_dict(all_simple_res)

        print("\n  Walk-Forward AVERAGE Results:")
        print_comparison([
            (f"ML ({all_ml_labels[0]})", avg_ml),
            ("SmaCrossover", avg_sma),
            ("SimpleStrat 1", avg_simple),
        ])

        # Retrain on ALL data with best model type from last fold
        print("\n  Retraining on full dataset ...")
        final_model_type = all_ml_labels[-1].lower()
        # map label back to model type key
        type_map = {"randomforest": "rf", "gradientboosting": "gbt",
                     "xgboost": "xgb", "lightgbm": "lgb"}
        for k, v in type_map.items():
            if k in final_model_type:
                final_model_type = v
                break
        last_model = train_model(full_df, feature_cols, final_model_type, **params)
        best_model = last_model
        best_res = avg_ml
        model_type = all_ml_labels[-1]

        # Retrain baselines on full val set for comparison (use last fold's val)
        sma_res = avg_sma
        simple_res = avg_simple
        ml_wins = beats_baselines(best_res, sma_res, simple_res)

        # Update params for save
        params["walk_forward_folds"] = args.walk_forward

    else:
        # Single train/val split
        train_df, val_df = train_val_split(
            full_df, args.val_split, cutoff_date=args.cutoff_date,
        )
        print(f"\n  Train: {len(train_df)} samples, Val: {len(val_df)} samples")

        if args.grid_search:
            best_model, best_res, best_combo = grid_search(
                train_df, val_df, feature_cols, params, model_types,
            )
            if args.auto_threshold:
                tuned_t, tuned_res = tune_confidence_threshold(
                    val_df, feature_cols, best_model, args.base_buy_size,
                    use_kelly=args.kelly,
                    max_hold_bars=args.max_hold_bars,
                    trailing_stop_pct=args.trailing_stop_pct,
                )
                print(f"\n  \U0001f3af Auto-tuned confidence threshold: {tuned_t} "
                      f"(Sharpe: {tuned_res['sharpe_ratio']:.2f})")
                params["confidence_threshold"] = tuned_t
                best_res = tuned_res

            baselines = _baseline_results(val_df)
            sma_res = baselines["SmaCrossover"]
            simple_res = baselines["SimpleStrat_1"]

            dummy = {"total_return_pct": 0, "sharpe_ratio": 0,
                     "win_rate_pct": 0, "max_drawdown_pct": 0, "num_trades": 0}
            print_comparison([
                (f"GridSearch ({best_combo['model_type']})", best_res),
            ] + list(baselines.items()))
            ml_wins = beats_baselines(best_res, sma_res, simple_res)
            model_type = f"GridSearch ({best_combo['model_type']})"
        else:
            results = evaluate_models(
                train_df, val_df, feature_cols, model_types, params,
            )
            baselines = _baseline_results(val_df)
            sma_res = baselines["SmaCrossover"]
            simple_res = baselines["SimpleStrat_1"]

            all_rows: list[tuple[str, dict]] = []
            best_model = None
            best_res = None
            best_label = ""
            for label, model, res in results:
                all_rows.append((f"ML \u2014 {label}", res))
                if best_res is None or res["sharpe_ratio"] > best_res["sharpe_ratio"]:
                    best_model = model
                    best_res = res
                    best_label = label

            all_rows.extend(baselines.items())
            print_comparison(all_rows)

            if args.auto_threshold and best_model is not None:
                tuned_t, tuned_res = tune_confidence_threshold(
                    val_df, feature_cols, best_model, args.base_buy_size,
                    use_kelly=args.kelly,
                    max_hold_bars=args.max_hold_bars,
                    trailing_stop_pct=args.trailing_stop_pct,
                )
                print(f"\n  \U0001f3af Auto-tuned confidence threshold: {tuned_t} "
                      f"(Sharpe: {tuned_res['sharpe_ratio']:.2f})")
                params["confidence_threshold"] = tuned_t
                best_res = tuned_res

            model_type = best_label
            ml_wins = beats_baselines(best_res, sma_res, simple_res)

    # ── 4. Save decision ──
    print(f"\nStep 4/5: Save decision ...")
    should_save = not args.beat_baselines or ml_wins

    if should_save and best_model is not None:
        metadata = ModelMetadata(
            feature_columns=feature_cols,
            model_type=model_type,
            params=args.__dict__,
            train_symbols=symbols,
            train_years=args.years,
            num_train_samples=len(full_df),
            num_val_samples=0,
            validation_metrics=best_res,
            baseline_comparison={
                "SmaCrossover": sma_res,
                "SimpleStrat_1": simple_res,
            },
            beat_baselines=ml_wins,
        )
        path = save_model(best_model, args.name, metadata, args.model_dir)
        print(f"  \u2705 Saved to: {path}")
        if ml_wins:
            print(f"  \U0001f3c6 Champion model '{args.name}' beats both baselines!")
    else:
        print(f"  \u274c Model did NOT beat baselines. Nothing saved.")

    # ── 5. Feature importance ──
    print(f"\nStep 5/5: Feature importance (top 10) ...")
    if best_model is not None and hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        indices = np.argsort(imp)[::-1][:10]
        for rank, idx in enumerate(indices, 1):
            print(f"  {rank}. {feature_cols[idx]}: {imp[idx]:.4f}")
    else:
        print("  (not available)")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
