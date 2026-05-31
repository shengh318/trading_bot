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

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from backend.cpp_ext import triple_barrier_label
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

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

from backend.ml.features import compute_features, get_feature_columns, merge_context_features
from backend.ml.model import save_model, ModelMetadata, MODELS_DIR
from backend.ml.stacking import StackedEnsemble, MetaLabeledModel, MultiHorizonEnsemble, RegimeAwareModel
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
            print(f"  [!] No data for {sym}, skipping")
            continue
        col_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        df.index.name = "timestamp"
        data[sym] = df[list(col_map.values())]
        print(f"    {len(df)} bars  ({df.index[0].date()} -> {df.index[-1].date()})")
    return data


def prepare_features(
    data_dict: dict[str, pd.DataFrame],
    labeling: str = "next_bar",
    triple_barrier_pct: float = 0.02,
    triple_barrier_max_bars: int = 10,
    forecast_horizon: int = 1,
    context_data_dict: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute features for all symbols, combine into a single DataFrame.

    Args:
        data_dict: Maps symbol -> OHLCV DataFrame.
        labeling: ``"next_bar"`` (binary up/down) or ``"triple_barrier"``.
        triple_barrier_pct: Profit target / stop-loss percentage.
        triple_barrier_max_bars: Max holding period for triple barrier.
        forecast_horizon: Number of bars forward for the binary target.
        context_data_dict: Optional mapping of context symbol -> OHLCV
            DataFrame.  Features from these symbols are merged in.
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

        # ── Cross-symbol features ──
        if context_data_dict:
            d = merge_context_features(d, context_data_dict)

        all_dfs.append(d)

    combined = pd.concat(all_dfs)
    feature_cols = get_feature_columns(combined)
    clean = combined.dropna(subset=feature_cols + ["target"])
    if len(clean) < 100:
        raise ValueError(
            f"Only {len(clean)} clean rows after dropping NaN - need at least 100"
        )
    clean = clean.sort_index(kind="mergesort")
    return clean, feature_cols


def _triple_barrier_label(
    df: pd.DataFrame,
    pct: float = 0.02,
    max_bars: int = 10,
) -> pd.Series:
    """Triple-barrier labeling (de Prado) — C++ accelerated."""
    labels = triple_barrier_label(
        df["close"].values, df["high"].values, df["low"].values, pct, max_bars,
    )
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
        train = df[df.index < cutoff_dt]
        val = df[df.index >= cutoff_dt]
    else:
        split_idx = int(len(df) * val_split)
        cutoff_dt = df.index[split_idx]
        train = df[df.index < cutoff_dt]
        val = df[df.index >= cutoff_dt]
    return train, val


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 3,
    cutoff_date: str | None = None,
    embargo: int = 5,
    purge_window: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create expanding-window walk-forward folds with purging and embargo.

    Implements de Prado's purged walk-forward to prevent data leakage
    between train and test sets:

    - **Purge**: training rows whose label depends on data overlapping
      with the validation set are removed.
    - **Embargo**: an additional buffer of *embargo* rows after the purge
      point is also removed as a safety margin.

    When *cutoff_date* is provided, data before that date is always
    included in every training fold (base set). Walk-forward folds are
    created from data on or after the cutoff date.

    Args:
        df: Combined DataFrame with a ``symbol`` column and datetime index.
        n_folds: Number of walk-forward folds.
        cutoff_date: Optional ISO date to separate base set from walk set.
        embargo: Number of rows to drop after the purge boundary.
        purge_window: Max label lookahead (``forecast_horizon`` or
            ``triple_barrier_max_bars``).

    Returns:
        List of (train, val) DataFrames with purged/embargoed training sets.
    """
    if cutoff_date is not None:
        cutoff_dt = pd.Timestamp(cutoff_date)
        if df.index.tz is not None:
            cutoff_dt = cutoff_dt.tz_localize(df.index.tz)
        base = df[df.index < cutoff_dt]
        walk = df[df.index >= cutoff_dt]
    else:
        base = None
        walk = df

    symbols = walk["symbol"].unique()
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    min_train = 50
    min_val = 20
    for fold in range(1, n_folds + 1):
        all_train: list[pd.DataFrame] = []
        all_val: list[pd.DataFrame] = []
        for sym in symbols:
            sym_df = walk[walk["symbol"] == sym].sort_index()
            n = len(sym_df)
            pct = fold / (n_folds + 1)
            idx = int(n * pct)
            if idx < min_train:
                continue
            if fold < n_folds:
                next_idx = int(n * ((fold + 1) / (n_folds + 1)))
            else:
                next_idx = n
            if next_idx - idx < min_val:
                continue
            all_train.append(sym_df.iloc[:idx])
            all_val.append(sym_df.iloc[idx:next_idx])
        if all_train and all_val:
            # ── Per-symbol purge: remove rows whose labels overlap validation ──
            if purge_window > 0:
                val_min_ts = pd.concat(all_val).index.min()
                for i in range(len(all_train)):
                    sym_train = all_train[i]
                    unique_ts = sym_train.index.unique().sort_values()
                    pos = int(unique_ts.searchsorted(val_min_ts, side="left"))
                    if pos > 0:
                        cutoff_ts = unique_ts[max(0, pos - purge_window)]
                        all_train[i] = sym_train[sym_train.index < cutoff_ts]

            # ── Per-symbol embargo ──
            if embargo > 0:
                for i in range(len(all_train)):
                    sym_train = all_train[i]
                    if len(sym_train) > embargo:
                        all_train[i] = sym_train.iloc[:-embargo]

            train_parts = [base] + all_train if base is not None else all_train
            train_df = pd.concat(train_parts)

            folds.append((train_df, pd.concat(all_val)))
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

    close_arrs: dict[str, np.ndarray] = {}
    feat_arrs: dict[str, np.ndarray] = {}
    feat_idx: dict[str, np.ndarray] = {}
    ts_sets: dict[str, set] = {}
    for sym, sd in sym_data.items():
        close_arrs[sym] = sd["close"].values
        feat_arrs[sym] = sd[feature_cols].values
        feat_idx[sym] = np.arange(len(sd))
        ts_sets[sym] = set(sd.index)

    timestamps = sorted(set.union(*ts_sets.values()))
    all_trades: list[dict] = []
    snapshots: list[float] = []

    for ts in timestamps:
        for sym in sorted(sym_data.keys()):
            if ts not in ts_sets[sym]:
                continue
            local_idx[sym] += 1
            i = local_idx[sym]
            price = float(close_arrs[sym][i])

            feat_row = feat_arrs[sym][i]
            if np.any(np.isnan(feat_row)):
                signal = Signal.HOLD
                prob_up = 0.5
            else:
                proba = model.predict_proba(feat_row.reshape(1, -1))[0]
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
            portfolio.positions.get(sym, 0) * close_arrs[sym][local_idx[sym]]
            for sym in sym_data
            if local_idx[sym] >= 0 and portfolio.positions.get(sym, 0) > 0
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
    cash_per_sym = initial_cash

    all_trades: list[dict] = []
    all_equities: list[list[float]] = []

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].copy()
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
                    old_shares = portfolio.positions.get(sym, 0)
                    old_avg = portfolio.avg_entry.get(sym, 0.0)
                    portfolio.positions[sym] = old_shares + qty
                    if old_shares > 0 and old_avg > 0:
                        portfolio.avg_entry[sym] = (old_avg * old_shares + price * qty) / (old_shares + qty)
                    else:
                        portfolio.avg_entry[sym] = price
                    trades.append({"side": "buy", "qty": qty, "price": price})

            elif signal in (Signal.SELL, Signal.EXIT) and portfolio.positions.get(sym, 0) > 0:
                qty = portfolio.positions[sym]
                proceeds = qty * price
                portfolio.cash += proceeds
                portfolio.positions[sym] = 0
                portfolio.avg_entry[sym] = 0.0
                pnl = proceeds - cost_basis
                cost_basis = 0.0
                trades.append({"side": "sell", "qty": qty, "price": price, "pnl": pnl})

            equity = portfolio.cash + portfolio.positions.get(sym, 0) * price
            snapshots.append(equity)

        all_trades.extend(trades)
        all_equities.append(snapshots)

    max_len = max(len(e) for e in all_equities) if all_equities else 0
    merged_equity: list[float] = (
        [sum(eq[j] if j < len(eq) else eq[-1] for eq in all_equities) for j in range(max_len)]
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
        print("\n  [OK] ML beats BOTH baselines on return AND Sharpe!")
    else:
        if not beats_ret:
            print(
                f"\n  [X] ML return ({ml_res['total_return_pct']:.1f}%) "
                f"vs SmaCrossover ({sma_res['total_return_pct']:.1f}%) / "
                f"SimpleStrat ({simple_res['total_return_pct']:.1f}%)"
            )
        if not beats_shp:
            print(
                f"  [X] ML Sharpe ({ml_res['sharpe_ratio']:.2f}) "
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
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            max_features=kwargs.get("max_features", "sqrt"),
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
            min_child_weight=kwargs.get("min_child_weight", 1),
            subsample=kwargs.get("subsample", 1.0),
            reg_lambda=kwargs.get("reg_lambda", 1),
            reg_alpha=kwargs.get("reg_alpha", 0),
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
            min_child_samples=kwargs.get("min_child_samples", 20),
            subsample=kwargs.get("subsample", 1.0),
            reg_lambda=kwargs.get("reg_lambda", 0),
            reg_alpha=kwargs.get("reg_alpha", 0),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_type == "sgd":
        model = SGDClassifier(
            loss=kwargs.get("loss", "log_loss"),
            penalty=kwargs.get("penalty", "l2"),
            alpha=kwargs.get("alpha", 0.0001),
            max_iter=kwargs.get("max_iter", 1000),
            tol=kwargs.get("tol", 1e-3),
            learning_rate=kwargs.get("learning_rate_sgd", "optimal"),
            eta0=kwargs.get("eta0", 0.01),
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (128, 64, 32)),
            activation=kwargs.get("activation", "relu"),
            alpha=kwargs.get("alpha_mlp", 0.001),
            batch_size=kwargs.get("batch_size", 32),
            learning_rate_init=kwargs.get("learning_rate_init", 0.001),
            max_iter=kwargs.get("max_iter", 500),
            early_stopping=kwargs.get("early_stopping", True),
            validation_fraction=0.1,
            random_state=42,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.1),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            subsample=kwargs.get("subsample", 1.0),
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
    stacking: bool = False,
    meta_labeling: bool = False,
    multi_horizon: list[int] | None = None,
    regime_aware: bool = False,
) -> list[tuple[str, object, dict]]:
    """Train + backtest each model type, return (label, model, results) list.

    When *stacking* is ``True``, also builds a ``StackedEnsemble`` from all
    trained base models and includes it as an additional result entry.

    When *meta_labeling* is ``True``, wraps each model with a
    ``MetaLabeledModel`` that filters low-conviction predictions.
    """
    base_buy = params.get("base_buy_size", 1000)
    conf = params.get("confidence_threshold", 0.55)
    use_kelly = params.get("use_kelly", False)
    max_hold = params.get("max_hold_bars", 30)
    trail = params.get("trailing_stop_pct", 0.05)
    results: list[tuple[str, object, dict]] = []

    for mt in model_types:
        label = {"rf": "RandomForest", "gbt": "GradientBoosting",
                 "xgb": "XGBoost", "lgb": "LightGBM", "sgd": "SGD",
                 "mlp": "MLP"}.get(mt, mt.upper())
        print(f"  Training {label} ...")
        model = train_model(train_df, feature_cols, mt, **params)

        if meta_labeling:
            print(f"  Fitting meta-labeler for {label} ...")
            wrapper = MetaLabeledModel(model)
            wrapper.fit_meta(train_df[feature_cols], train_df["target"])
            model = wrapper
            label = f"{label}+Meta"

        if regime_aware:
            model = RegimeAwareModel(model)
            label = f"{label}+Regime"

        # ── Overfitting detection ──
        train_acc = model.score(train_df[feature_cols], train_df["target"])
        val_acc = model.score(val_df[feature_cols], val_df["target"])
        gap = train_acc - val_acc
        warning = " << OVERFITTING" if gap > 0.10 else ""
        print(f"    Train acc: {train_acc:.1%}  Val acc: {val_acc:.1%}  Gap: {gap:.1%}{warning}")
        cm = confusion_matrix(val_df["target"], model.predict(val_df[feature_cols]))
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"    Val: prec={prec:.1%} recall={rec:.1%} f1={f1:.1%}")

        print(f"  Backtesting {label} ...")
        res = backtest_model(
            val_df, feature_cols, model, conf, base_buy,
            use_kelly=use_kelly, max_hold_bars=max_hold, trailing_stop_pct=trail,
        )
        results.append((label, model, res))

    # ── Multi-horizon ensemble (C8) ──
    if multi_horizon and len(multi_horizon) >= 2:
        for mt in model_types:
            mt_label = {"rf": "RF", "gbt": "GBT", "xgb": "XGB", "lgb": "LGB", "sgd": "SGD", "mlp": "MLP"}.get(mt, mt.upper())
            lab = f"MH-{mt_label}({','.join(str(h) for h in multi_horizon)})"
            print(f"  Training {lab} ...")
            try:
                ensemble = MultiHorizonEnsemble(
                    horizons=multi_horizon,
                    base_model_type=mt,
                    base_params=params,
                )
                ensemble.fit(train_df, feature_cols=feature_cols)
                res = backtest_model(
                    val_df, feature_cols, ensemble, conf, base_buy,
                    use_kelly=use_kelly, max_hold_bars=max_hold, trailing_stop_pct=trail,
                )
                results.append((lab, ensemble, res))
                print(f"         -> Return={res['total_return_pct']:.1f}%  "
                      f"Sharpe={res['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"    [!] Multi-horizon ensemble failed: {e}")

    # ── Stacking ensemble ──
    if stacking and len(results) >= 2:
        base_models: dict[str, object] = {}
        for label, model, _ in results:
            if isinstance(model, MultiHorizonEnsemble):
                continue
            key = label.split()[-1].lower()
            base_models[key] = model
        print(f"  Building StackingEnsemble ({len(base_models)} base models) ...")
        stacker = StackedEnsemble(base_models)
        pw = max(
            params.get("forecast_horizon", 1),
            params.get("triple_barrier_max_bars", 1) if params.get("labeling") == "triple_barrier" else 1,
        )
        stacker.fit(train_df[feature_cols], train_df["target"], purge_window=pw)
        label = "StackingEnsemble"
        model_out: object = stacker
        if regime_aware:
            model_out = RegimeAwareModel(stacker)
            label = f"{label}+Regime"
        print(f"  Backtesting {label} ...")
        stack_res = backtest_model(
            val_df, feature_cols, model_out, conf, base_buy,
            use_kelly=use_kelly, max_hold_bars=max_hold, trailing_stop_pct=trail,
        )
        results.append((label, model_out, stack_res))

    return results


def _stacking_after_grid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_types: list[str],
    best_combo: dict,
    best_res: dict,
    params: dict,
) -> tuple[object | None, dict | None]:
    """If ``--stacking`` was requested, build a StackingEnsemble using
    all model types trained with the best grid-search combo's params.
    Returns (stacker, stack_res) if the ensemble beats the grid winner
    by Sharpe, else (None, None).
    """
    if len(model_types) < 2:
        return None, None

    conf = params.get("confidence_threshold", 0.55)
    base_buy = params.get("base_buy_size", 1000)
    use_kelly = params.get("use_kelly", False)
    max_hold = params.get("max_hold_bars", 30)
    trail = params.get("trailing_stop_pct", 0.05)

    # Train all model types with best combo params
    base_models: dict[str, object] = {}
    combo_params = {k: v for k, v in best_combo.items() if k != "model_type"}
    for mt in model_types:
        m = train_model(train_df, feature_cols, mt, **combo_params)
        base_models[mt] = m

    print(f"  Building StackingEnsemble from grid winner ...")
    stacker = StackedEnsemble(base_models)
    pw = max(
        params.get("forecast_horizon", 1),
        params.get("triple_barrier_max_bars", 1) if params.get("labeling") == "triple_barrier" else 1,
    )
    stacker.fit(train_df[feature_cols], train_df["target"], purge_window=pw)

    print(f"  Backtesting StackingEnsemble ...")
    stack_res = backtest_model(
        val_df, feature_cols, stacker, conf, base_buy,
        use_kelly=use_kelly, max_hold_bars=max_hold, trailing_stop_pct=trail,
    )
    print(f"         -> Return={stack_res['total_return_pct']:.1f}%  "
          f"Sharpe={stack_res['sharpe_ratio']:.2f}  "
          f"(grid winner: {best_res['sharpe_ratio']:.2f})")

    if stack_res["sharpe_ratio"] > best_res["sharpe_ratio"]:
        print(f"  [>>] StackingEnsemble beats grid winner!")
        return stacker, stack_res
    return None, None


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


def _infer_model_type(model) -> str:
    """Return short model type string (rf/gbt/xgb/lgb) from a model instance."""
    name = type(model).__name__.lower()
    if "randomforest" in name:
        return "rf"
    if "gradientboosting" in name:
        return "gbt"
    if "xgboost" in name or "xgb" in name:
        return "xgb"
    if "lightgbm" in name or "lgbm" in name or "lgb" in name:
        return "lgb"
    if "sgd" in name or "SGD" in name:
        return "sgd"
    if "mlp" in name:
        return "mlp"
    return "rf"


def _try_prune(
    model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    threshold: float,
) -> tuple[list[str], object | None, dict | None]:
    """Drop bottom *threshold* features by importance, retrain, compare.

    Returns ``(new_feature_cols, pruned_model, pruned_results)`` if the
    pruned model improves Sharpe on validation, otherwise returns
    the original ``(feature_cols, None, None)``.
    """
    if threshold <= 0 or not hasattr(model, "feature_importances_"):
        return feature_cols, None, None

    # StackedEnsemble returns meta-coeffs — skip feature-level pruning
    if hasattr(model, "base_models"):
        return feature_cols, None, None

    imp = model.feature_importances_
    if len(imp) != len(feature_cols):
        return feature_cols, None, None

    threshold_val = float(np.quantile(imp, threshold))
    kept = [c for c, i in zip(feature_cols, imp) if i > threshold_val]

    # Don't drop more than 50% of features
    if len(kept) < len(feature_cols) * 0.5:
        return feature_cols, None, None

    dropped = list(set(feature_cols) - set(kept))
    if not dropped:
        return feature_cols, None, None

    print(f"  Pruning: dropping {len(dropped)} features "
          f"({', '.join(sorted(dropped)[:6])}{'...' if len(dropped) > 6 else ''})")

    # Retrain with pruned features
    mt = _infer_model_type(model)
    train_pruned = train_df[kept + ["target", "symbol"]] if "symbol" in train_df.columns else train_df[kept + ["target"]]
    val_pruned = val_df[kept + ["target", "symbol"]] if "symbol" in val_df.columns else val_df[kept + ["target"]]

    pruned_model = train_model(train_pruned, kept, mt, **params)

    conf = params.get("confidence_threshold", 0.55)
    base_buy = params.get("base_buy_size", 1000)
    use_kelly = params.get("use_kelly", False)
    max_hold = params.get("max_hold_bars", 30)
    trail = params.get("trailing_stop_pct", 0.05)

    pruned_res = backtest_model(
        val_pruned, kept, pruned_model, conf, base_buy,
        use_kelly=use_kelly, max_hold_bars=max_hold, trailing_stop_pct=trail,
    )

    # Recompute original result for fair comparison (in case best_res was from grid)
    print(f"    Pruned -> Return={pruned_res['total_return_pct']:.1f}%  "
          f"Sharpe={pruned_res['sharpe_ratio']:.2f}")

    return kept, pruned_model, pruned_res


# ── Grid search ───────────────────────────────────────────────────────────

def _grid_params(model_types: list[str], regularize: bool = False) -> list[dict]:
    combos: list[dict] = []
    for mt in model_types:
        if mt == "sgd":
            for penalty in ["l2", "l1"]:
                for alpha in [0.0001, 0.001, 0.01]:
                    for lr_sgd in ["optimal", "adaptive"]:
                        combos.append(dict(
                            model_type=mt, penalty=penalty,
                            alpha=alpha, learning_rate_sgd=lr_sgd,
                            eta0=0.01,
                        ))
        elif mt == "mlp":
            for hidden in ["(64, 32)", "(128, 64, 32)", "(256, 128, 64)"]:
                for alpha_mlp in [0.0001, 0.001]:
                    for lr_init in [0.001, 0.01]:
                        combos.append(dict(
                            model_type=mt,
                            hidden_layer_sizes=eval(hidden),
                            alpha_mlp=alpha_mlp,
                            learning_rate_init=lr_init,
                        ))
        else:
            for n in [100, 200, 300]:
                for d in [5, 8, 12]:
                    if mt in ("gbt", "xgb", "lgb"):
                        for lr in [0.05, 0.1, 0.2]:
                            base = dict(model_type=mt, n_estimators=n,
                                        max_depth=d, learning_rate=lr)
                            if regularize:
                                _add_reg_params(mt, base, combos)
                            else:
                                combos.append(base)
                    else:
                        base = dict(model_type=mt, n_estimators=n, max_depth=d)
                        if regularize:
                            _add_reg_params(mt, base, combos)
                        else:
                            combos.append(base)
    return combos


def _add_reg_params(mt: str, base: dict, combos: list[dict]) -> None:
    """Add regularization-focused param combos for a given base config."""
    if mt == "rf":
        for msl in [1, 5]:
            for mf in ["sqrt", "log2"]:
                c = dict(base)
                c.update(min_samples_leaf=msl, max_features=mf)
                combos.append(c)
    elif mt == "gbt":
        for msl in [1, 5]:
            for ss in [0.8, 1.0]:
                c = dict(base)
                c.update(min_samples_leaf=msl, subsample=ss)
                combos.append(c)
    elif mt == "xgb":
        for mcw in [1, 5]:
            for ss in [0.8, 1.0]:
                for rl in [0, 1]:
                    c = dict(base)
                    c.update(min_child_weight=mcw, subsample=ss, reg_lambda=rl)
                    combos.append(c)
    elif mt == "lgb":
        for mcs in [5, 20]:
            for ss in [0.8, 1.0]:
                for rl in [0, 1]:
                    c = dict(base)
                    c.update(min_child_samples=mcs, subsample=ss, reg_lambda=rl)
                    combos.append(c)


def grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    model_types: list[str],
    regularize: bool = False,
    meta_labeling: bool = False,
) -> tuple:
    """Try every hyperparameter combo, return the best model + its results."""
    conf = params.get("confidence_threshold", 0.55)
    base_buy = params.get("base_buy_size", 1000)
    use_kelly = params.get("use_kelly", False)
    max_hold = params.get("max_hold_bars", 30)
    trail = params.get("trailing_stop_pct", 0.05)
    combos = _grid_params(model_types, regularize=regularize)

    best_model = None
    best_res: dict | None = None
    best_combo: dict | None = None

    for i, c in enumerate(combos):
        mt = c["model_type"]
        mt_label = {"rf": "RF", "gbt": "GBT", "xgb": "XGB", "lgb": "LGB"}.get(mt, mt.upper())
        lr_str = f" lr={c.get('learning_rate', '-')}" if "learning_rate" in c else ""
        reg_str = ""
        if regularize:
            extra = []
            for k in ("min_samples_leaf", "max_features", "subsample",
                      "min_child_weight", "min_child_samples", "reg_lambda"):
                if k in c:
                    extra.append(f"{k.split('_')[-1]}={c[k]}")
            reg_str = " " + " ".join(extra) if extra else ""
        meta_str = " [M]" if meta_labeling else ""
        n_est_str = f"n={c.get('n_estimators', '-')} "
        depth_str = f"depth={c.get('max_depth', '-')} "
        print(f"  [{i+1}/{len(combos)}] {mt_label} {n_est_str}"
              f"{depth_str}{lr_str}{reg_str}{meta_str}")

        model = train_model(train_df, feature_cols, **c)

        if meta_labeling:
            wrapper = MetaLabeledModel(model)
            wrapper.fit_meta(train_df[feature_cols], train_df["target"])
            model = wrapper

        # ── Overfitting detection ──
        train_acc = model.score(train_df[feature_cols], train_df["target"])
        val_acc = model.score(val_df[feature_cols], val_df["target"])
        gap = train_acc - val_acc
        warning = " << OVERFITTING" if gap > 0.10 else ""

        res = backtest_model(
            val_df, feature_cols, model, conf, base_buy,
            use_kelly=use_kelly,
            max_hold_bars=max_hold,
            trailing_stop_pct=trail,
        )
        print(f"         -> Return={res['total_return_pct']:.1f}%  "
              f"Sharpe={res['sharpe_ratio']:.2f}  "
              f"Acc={train_acc:.1%}/{val_acc:.1%}{warning}")

        if best_res is None or res["sharpe_ratio"] > best_res["sharpe_ratio"]:
            best_model = model
            best_res = res
            best_combo = c

    label = {"rf": "RF", "gbt": "GBT", "xgb": "XGB", "lgb": "LGB"}.get(
        best_combo["model_type"], best_combo["model_type"].upper()
    )
    print(f"\n  [*] Grid search winner: {label} "
          f"n={best_combo.get('n_estimators', '-')} "
          f"depth={best_combo.get('max_depth', '-')} "
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
                   help="Comma-separated model types: rf,gbt,xgb,lgb,sgd,mlp (default: rf,gbt)")
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
    p.add_argument("--triple-barrier-pct", type=float, default=0.02,
                   help="Profit target / stop-loss %% for triple barrier (default: 0.02)")
    p.add_argument("--triple-barrier-max-bars", type=int, default=10,
                   help="Max holding period bars for triple barrier (default: 10)")
    p.add_argument("--forecast-horizon", type=int, default=1,
                   help="Number of days forward for target prediction (default: 1)")
    p.add_argument("--max-hold-bars", type=int, default=30,
                   help="Max bars to hold a position before forced exit (default: 30)")
    p.add_argument("--trailing-stop-pct", type=float, default=0.05,
                   help="Trailing stop loss as fraction (default: 0.05 = 5%%)")
    p.add_argument("--stacking", action="store_true",
                   help="Build a StackingEnsemble from all model types and compare against individuals")
    p.add_argument("--meta-labeling", action="store_true",
                   help="Two-stage: primary predicts direction, meta-model filters false signals")
    p.add_argument("--regime-aware", action="store_true",
                   help="Wrap model with RegimeAwareModel to adjust predictions by market regime")
    p.add_argument("--prune", type=float, default=0.0,
                   help="Drop bottom N%% features by importance, retrain, keep if better (default: 0 = off)")
    p.add_argument("--regularize", action="store_true",
                   help="Add regularization params (min_samples_leaf, subsample, etc.) to grid search")
    p.add_argument("--embargo", type=int, default=5,
                   help="Embargo buffer rows for purged walk-forward (default: 5)")
    p.add_argument("--multi-horizon", default=None,
                   help="Comma-separated forecast horizons for multi-horizon ensemble (e.g. 1,5,21)")
    p.add_argument("--context-symbols", default=None,
                   help="Comma-separated context symbols (e.g. SPY,VOO) for cross-symbol features")
    p.add_argument("--model-dir", default=str(MODELS_DIR),
                   help=f"Output dir (default: {MODELS_DIR})")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    model_types = [s.strip() for s in args.model_types.split(",")]

    for mt in model_types:
        if mt not in ("rf", "gbt", "xgb", "lgb", "sgd", "mlp"):
            print(f"ERROR: Unknown model type '{mt}'. Choose from: rf, gbt, xgb, lgb, sgd, mlp")
            sys.exit(1)
        if mt == "xgb" and not HAS_XGB:
            print("ERROR: xgboost not installed. Run: pip install xgboost")
            sys.exit(1)
        if mt == "lgb" and not HAS_LGB:
            print("ERROR: lightgbm not installed. Run: pip install lightgbm")
            sys.exit(1)

    mh_horizons: list[int] | None = None
    if args.multi_horizon:
        mh_horizons = [int(h) for h in args.multi_horizon.split(",")]
        if len(mh_horizons) < 2:
            print(f"ERROR: --multi-horizon requires at least 2 horizons")
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
    print(f"  Auto-thresh: {'ON' if args.auto_threshold else 'OFF'}")
    print(f"  Stacking:    {'ON' if args.stacking else 'OFF'}")
    print(f"  Meta-label:  {'ON' if args.meta_labeling else 'OFF'}")
    print(f"  Regime-aware:{'ON' if args.regime_aware else 'OFF'}")
    print(f"  Regularize:  {'ON' if args.regularize else 'OFF'}")
    print(f"  Embargo:     {args.embargo}")
    print(f"  Labeling:    {args.labeling}")
    print(f"  Horizon:     {args.forecast_horizon}-day")
    print(f"  Max hold:    {args.max_hold_bars} bars")
    print(f"  Trail stop:  {args.trailing_stop_pct*100:.0f}%")
    print(f"  Prune:       {'ON (' + str(int(args.prune * 100)) + '%)' if args.prune > 0 else 'OFF'}")
    print(f"  Multi-horiz: {args.multi_horizon or 'OFF'}")
    print(f"  Context:     {args.context_symbols or 'OFF'}")
    print(f"{'='*60}\n")

    # ── Context symbols (C3) ──
    context_symbols: list[str] = []
    context_data: dict[str, pd.DataFrame] | None = None
    if args.context_symbols:
        context_symbols = [s.strip().upper() for s in args.context_symbols.split(",")]
        print(f"  Context symbols: {', '.join(context_symbols)}")

    # ── 1. Download ──
    print("Step 1/5: Downloading data ...")
    data_dict = download_data(symbols, args.years)
    if not data_dict:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    if context_symbols:
        print("  Downloading context symbol data ...")
        context_data = download_data(context_symbols, args.years)

    # ── 2. Prepare features ──
    print("\nStep 2/5: Computing features ...")
    full_df, feature_cols = prepare_features(
        data_dict, labeling=args.labeling,
        triple_barrier_pct=args.triple_barrier_pct,
        triple_barrier_max_bars=args.triple_barrier_max_bars,
        forecast_horizon=args.forecast_horizon,
        context_data_dict=context_data,
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
        "forecast_horizon": args.forecast_horizon,
        "triple_barrier_max_bars": args.triple_barrier_max_bars,
        "labeling": args.labeling,
    }

    # ── 3. Train & backtest ──
    print(f"\nStep 3/5: {'Grid search' if args.grid_search else 'Training'} ...")

    if args.walk_forward:
        purge = max(args.forecast_horizon, args.triple_barrier_max_bars if args.labeling == "triple_barrier" else 1)
        folds = walk_forward_folds(
            full_df, n_folds=args.walk_forward, cutoff_date=args.cutoff_date,
            embargo=args.embargo, purge_window=purge,
        )
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
                    regularize=args.regularize,
                    meta_labeling=args.meta_labeling,
                )
                label = f"GridSearch ({best_combo['model_type']})"
                if args.stacking:
                    s_model, s_res = _stacking_after_grid(
                        train_fold, val_fold, feature_cols, model_types,
                        best_combo, best_res, params,
                    )
                    if s_model is not None:
                        best_model = s_model
                        best_res = s_res
                        label = "StackingEnsemble"
            else:
                results = evaluate_models(
                    train_fold, val_fold, feature_cols, model_types, params,
                    stacking=args.stacking,
                    meta_labeling=args.meta_labeling,
                    multi_horizon=mh_horizons,
                    regime_aware=args.regime_aware,
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
        import re
        m = re.search(r'\((\w+)\)', final_model_type)
        if m:
            final_model_type = m.group(1)
        type_map = {"randomforest": "rf", "gradientboosting": "gbt",
                     "xgboost": "xgb", "lightgbm": "lgb",
                     "sgd": "sgd", "mlp": "mlp",
                     "rf": "rf", "gbt": "gbt", "xgb": "xgb", "lgb": "lgb",
                     "sgd": "sgd", "mlp": "mlp"}
        for k, v in type_map.items():
            if k in final_model_type:
                final_model_type = v
                break
        last_model = train_model(full_df, feature_cols, final_model_type, **params)
        best_model = last_model
        full_res = backtest_model(
            full_df, feature_cols, last_model,
            params.get("confidence_threshold", 0.55),
            params.get("base_buy_size", 1000),
            use_kelly=params.get("use_kelly", False),
            max_hold_bars=params.get("max_hold_bars", 30),
            trailing_stop_pct=params.get("trailing_stop_pct", 0.05),
        )
        best_res = full_res
        model_type = all_ml_labels[-1]

        # Retrain baselines on full val set for comparison (use last fold's val)
        sma_res = avg_sma
        simple_res = avg_simple
        ml_wins = beats_baselines(best_res, sma_res, simple_res)

        num_train_samples = len(full_df)
        num_val_samples = int(sum(len(v) for _, v in folds) / len(folds))
        train_params_for_save = dict(params)
        train_params_for_save["walk_forward_folds"] = args.walk_forward

    else:
        # Single train/val split
        train_df, val_df = train_val_split(
            full_df, args.val_split, cutoff_date=args.cutoff_date,
        )
        print(f"\n  Train: {len(train_df)} samples, Val: {len(val_df)} samples")
        num_train_samples = len(train_df)
        num_val_samples = len(val_df)

        if args.grid_search:
            best_model, best_res, best_combo = grid_search(
                train_df, val_df, feature_cols, params, model_types,
                regularize=args.regularize,
                meta_labeling=args.meta_labeling,
            )
            if args.auto_threshold:
                tuned_t, tuned_res = tune_confidence_threshold(
                    val_df, feature_cols, best_model, args.base_buy_size,
                    use_kelly=args.kelly,
                    max_hold_bars=args.max_hold_bars,
                    trailing_stop_pct=args.trailing_stop_pct,
                )
                print(f"\n  [>>] Auto-tuned confidence threshold: {tuned_t} "
                      f"(Sharpe: {tuned_res['sharpe_ratio']:.2f})")
                params["confidence_threshold"] = tuned_t
                best_res = tuned_res

            if args.stacking:
                s_model, s_res = _stacking_after_grid(
                    train_df, val_df, feature_cols, model_types,
                    best_combo, best_res, params,
                )
                if s_model is not None:
                    best_model = s_model
                    best_res = s_res
                    model_type = "StackingEnsemble (grid)"
                else:
                    model_type = f"GridSearch ({best_combo['model_type']})"
            else:
                model_type = f"GridSearch ({best_combo['model_type']})"

            baselines = _baseline_results(val_df)
            sma_res = baselines["SmaCrossover"]
            simple_res = baselines["SimpleStrat_1"]

            print_comparison([
                (model_type, best_res),
            ] + list(baselines.items()))
            ml_wins = beats_baselines(best_res, sma_res, simple_res)
        else:
            results = evaluate_models(
                train_df, val_df, feature_cols, model_types, params,
                stacking=args.stacking,
                meta_labeling=args.meta_labeling,
                multi_horizon=mh_horizons,
                regime_aware=args.regime_aware,
            )
            baselines = _baseline_results(val_df)
            sma_res = baselines["SmaCrossover"]
            simple_res = baselines["SimpleStrat_1"]

            all_rows: list[tuple[str, dict]] = []
            best_model = None
            best_res = None
            best_label = ""
            for label, model, res in results:
                all_rows.append((f"ML - {label}", res))
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
                print(f"\n  [>>] Auto-tuned confidence threshold: {tuned_t} "
                      f"(Sharpe: {tuned_res['sharpe_ratio']:.2f})")
                params["confidence_threshold"] = tuned_t
                best_res = tuned_res

            model_type = best_label
            ml_wins = beats_baselines(best_res, sma_res, simple_res)

    # ── 3.5. Feature pruning ──
    if args.prune > 0 and best_model is not None:
        pruned_cols, pruned_model, pruned_res = _try_prune(
            best_model, full_df if args.walk_forward else train_df,
            full_df if args.walk_forward else val_df,
            feature_cols, params, args.prune,
        )
        if pruned_model is not None and pruned_res is not None:
            if pruned_res["sharpe_ratio"] > best_res["sharpe_ratio"]:
                print(f"    [>>] Pruned model beats original — keeping {len(pruned_cols)} features")
                best_model = pruned_model
                best_res = pruned_res
                feature_cols = pruned_cols
            else:
                print(f"    [-] Pruned model did not beat original — keeping original")

    # ── 4. Save decision ──
    print(f"\nStep 4/5: Save decision ...")
    should_save = not args.beat_baselines or ml_wins

    if should_save and best_model is not None:
        saved_params = train_params_for_save if args.walk_forward else dict(params)
        metadata = ModelMetadata(
            feature_columns=feature_cols,
            model_type=model_type,
            params=saved_params,
            train_symbols=symbols,
            train_years=args.years,
            num_train_samples=num_train_samples,
            num_val_samples=num_val_samples,
            validation_metrics=best_res,
            baseline_comparison={
                "SmaCrossover": sma_res,
                "SimpleStrat_1": simple_res,
            },
            beat_baselines=ml_wins,
            context_symbols=context_symbols,
        )
        path = save_model(best_model, args.name, metadata, args.model_dir)
        print(f"  [OK] Saved to: {path}")
        if ml_wins:
            print(f"  [TROPHY] Champion model '{args.name}' beats both baselines!")
    else:
        print(f"  [X] Model did NOT beat baselines. Nothing saved.")

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
