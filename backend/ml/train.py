"""CLI training script for multi-symbol ML trading model.

Usage:
    python -m backend.ml.train --symbols NVDA,AMD,VOO,SPY,META --years 20

Compares against SmaCrossover and SimpleStrat 1 baselines.
With --beat-baselines, only saves model if it outperforms both.
With --grid-search, tries many hyperparameter combinations automatically.
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from backend.ml.features import compute_features, get_feature_columns
from backend.ml.model import save_model, delete_model, ModelMetadata, MODELS_DIR
from backend.strategies.base import Portfolio, Signal
from backend.strategies.sma_crossover import SmaCrossover
from backend.strategies.simple_strat_1 import SimpleStrat1


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


def prepare_training_data(
    data_dict: dict[str, pd.DataFrame],
    val_split: float,
    cutoff_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute features for all symbols, combine, time-based train/val split.

    Args:
        data_dict: Maps symbol -> OHLCV DataFrame.
        val_split: Fraction of rows used for **training** (e.g. 0.8 = 80%
            train, 20% val).  Only used when ``cutoff_date`` is *not* given.
        cutoff_date: Explicit date boundary (ISO format, e.g. ``"2023-01-01"``).
            All data **before** this date becomes training; all data **on or
            after** becomes validation.  Supersedes ``val_split``.
    """
    all_dfs: list[pd.DataFrame] = []
    for sym, df in data_dict.items():
        d = compute_features(df.copy())
        d["symbol"] = sym
        d["target"] = (d["close"].shift(-1) > d["close"]).astype(int)
        all_dfs.append(d)

    combined = pd.concat(all_dfs)
    feature_cols = get_feature_columns(combined)

    clean = combined.dropna(subset=feature_cols + ["target"])
    if len(clean) < 100:
        raise ValueError(
            f"Only {len(clean)} clean rows after dropping NaN \u2014 need at least 100"
        )

    clean = clean.sort_index()

    if cutoff_date is not None:
        cutoff_dt = pd.Timestamp(cutoff_date)
        if clean.index.tz is not None:
            cutoff_dt = cutoff_dt.tz_localize(clean.index.tz)
        train = clean[clean.index < cutoff_dt].reset_index(drop=True)
        val = clean[clean.index >= cutoff_dt].reset_index(drop=True)
        print(f"\n  Using explicit cutoff: {cutoff_date}")
    else:
        split_idx = int(len(clean) * val_split)
        cutoff_dt = clean.index[split_idx]
        train = clean[clean.index < cutoff_dt].reset_index(drop=True)
        val = clean[clean.index >= cutoff_dt].reset_index(drop=True)
        print(f"\n  Using auto cutoff ({val_split*100:.0f}/{100-val_split*100:.0f} split)")

    print(f"  Training samples:     {len(train)}")
    print(f"  Validation samples:   {len(val)}")
    print(f"  Cutoff date:          {cutoff_dt.date()}")
    print(f"  Feature columns:      {len(feature_cols)}")
    return train, val, feature_cols


# ── Mini-backtest helpers ─────────────────────────────────────────────────

def backtest_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    confidence_threshold: float,
    base_buy_size: float,
    initial_cash: float = 10000,
) -> dict:
    """Simplified backtest using a trained model on DataFrame rows."""
    portfolio = Portfolio(initial_cash)
    trades: list[dict] = []
    snapshots: list[float] = []
    symbol = "ASSET"
    cost_basis = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])

        feat = row[feature_cols].values.reshape(1, -1)
        if pd.isna(feat).any():
            signal = Signal.HOLD
            prob_up = 0.5
        else:
            feat = feat.astype(float)
            proba = model.predict_proba(feat)[0]
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

        if signal == Signal.BUY and portfolio.cash > 0:
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
                cost_basis += cost
                existing = portfolio.positions.get(symbol, 0)
                portfolio.positions[symbol] = existing + qty
                trades.append({"side": "buy", "qty": qty, "price": price})

        elif signal in (Signal.SELL, Signal.EXIT) and portfolio.positions.get(symbol, 0) > 0:
            qty = portfolio.positions[symbol]
            proceeds = qty * price
            portfolio.cash += proceeds
            portfolio.positions[symbol] = 0
            pnl = proceeds - cost_basis
            cost_basis = 0.0
            trades.append({"side": "sell", "qty": qty, "price": price, "pnl": pnl})

        equity = portfolio.cash + portfolio.positions.get(symbol, 0) * price
        snapshots.append(equity)

    return _calc_metrics(snapshots, trades, initial_cash)


def backtest_baseline(
    df: pd.DataFrame,
    strategy_class,
    params: dict,
    initial_cash: float = 10000,
    buy_size: float | None = None,
) -> dict:
    """Simplified backtest for rule-based strategies, run per-symbol.

    Args:
        buy_size: Overrides the strategy's ``buy_size`` attr when set.
            Handy for strategies (e.g. SmaCrossover) that don't declare one.
    """
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

    # Merge equity curves (sum per-symbol equity at each step)
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


# ── Comparison ────────────────────────────────────────────────────────────

def print_comparison(
    rf_res: dict,
    gbt_res: dict,
    sma_res: dict,
    simple_res: dict,
) -> None:
    """Print the strategy comparison table."""
    print("\n" + "=" * 84)
    print("  STRATEGY COMPARISON  (Validation Period)")
    print("=" * 84)
    print(f"{'Strategy':<24} {'Return':>8} {'Sharpe':>8} {'Win Rate':>9} "
          f"{'Max DD':>8} {'Trades':>7}")
    print("-" * 84)
    for name, r in [
        ("ML — RandomForest", rf_res),
        ("ML — GradientBoosting", gbt_res),
        ("SmaCrossover", sma_res),
        ("SimpleStrat 1", simple_res),
    ]:
        print(
            f"{name:<24} {r['total_return_pct']:>7.1f}% "
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
    """Train a single model."""
    X = train_df[feature_cols].values
    y = train_df["target"].values

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 10),
            random_state=42,
            n_jobs=-1,
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
    params: dict,
) -> tuple:
    """Train RF + GBT, backtest all 4 strategies, return results."""
    base_buy = params.get("base_buy_size", 1000)
    conf = params.get("confidence_threshold", 0.55)

    # Train
    print("  Training RandomForest ...")
    rf_model = train_model(train_df, feature_cols, "rf", **params)
    print("  Training GradientBoosting ...")
    gbt_model = train_model(train_df, feature_cols, "gbt", **params)

    # ML backtests
    print("  Backtesting ML models ...")
    rf_res = backtest_model(val_df, feature_cols, rf_model, conf, base_buy)
    gbt_res = backtest_model(val_df, feature_cols, gbt_model, conf, base_buy)

    # Baseline backtests
    print("  Running SmaCrossover ...")
    sma_res = backtest_baseline(
        val_df, SmaCrossover,
        {"short_window": 10, "long_window": 50},
        buy_size=1000,
    )
    print("  Running SimpleStrat 1 ...")
    simple_res = backtest_baseline(
        val_df, SimpleStrat1,
        {"buy_size": 100, "entry_drop": 1.0, "profit_target": 20.0,
         "sell_portion": 100.0, "stop_loss": 3.0, "max_buys": 1},
    )

    return rf_model, gbt_model, rf_res, gbt_res, sma_res, simple_res


# ── Grid search ───────────────────────────────────────────────────────────

def _grid_params() -> list[dict]:
    combos: list[dict] = []
    for mt in ["rf", "gbt"]:
        for n in [100, 200, 300]:
            for d in [5, 8, 12]:
                if mt == "gbt":
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
) -> tuple:
    """Try every hyperparameter combo, return the best model + its results."""
    conf = params.get("confidence_threshold", 0.55)
    base_buy = params.get("base_buy_size", 1000)
    combos = _grid_params()

    best_model = None
    best_res: dict | None = None
    best_combo: dict | None = None

    for i, c in enumerate(combos):
        mt_label = "RF" if c["model_type"] == "rf" else "GBT"
        lr_str = f" lr={c.get('learning_rate', '-')}" if "learning_rate" in c else ""
        print(f"  [{i+1}/{len(combos)}] {mt_label} n={c['n_estimators']} "
              f"depth={c['max_depth']}{lr_str}")

        model = train_model(train_df, feature_cols, **c)
        res = backtest_model(val_df, feature_cols, model, conf, base_buy)
        print(f"         \u2192 Return={res['total_return_pct']:.1f}%  "
              f"Sharpe={res['sharpe_ratio']:.2f}")

        if best_res is None or res["sharpe_ratio"] > best_res["sharpe_ratio"]:
            best_model = model
            best_res = res
            best_combo = c

    print(f"\n  \u2b50 Grid search winner: {best_combo}")
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
    p.add_argument("--model-dir", default=str(MODELS_DIR),
                   help=f"Output dir (default: {MODELS_DIR})")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]

    print(f"\n{'='*60}")
    print("  ML TRADING MODEL TRAINING")
    print(f"{'='*60}")
    print(f"  Symbols:   {', '.join(symbols)}")
    print(f"  Years:     {args.years}")
    print(f"  Model:     {args.name}")
    print(f"  Beat mode: {'ON' if args.beat_baselines else 'OFF'}")
    print(f"  Grid:      {'ON' if args.grid_search else 'OFF'}")
    print(f"{'='*60}\n")

    # ── 1. Download ──
    print("Step 1/5: Downloading data ...")
    data_dict = download_data(symbols, args.years)
    if not data_dict:
        print("ERROR: No data downloaded.")
        sys.exit(1)

    # ── 2. Prepare ──
    print("\nStep 2/5: Computing features ...")
    train_df, val_df, feature_cols = prepare_training_data(
        data_dict, args.val_split, cutoff_date=args.cutoff_date,
    )

    # ── 3. Train & backtest ──
    print(f"\nStep 3/5: {'Grid search' if args.grid_search else 'Training'} ...")

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "confidence_threshold": args.confidence_threshold,
        "base_buy_size": args.base_buy_size,
    }

    if args.grid_search:
        best_model, best_res, best_combo = grid_search(
            train_df, val_df, feature_cols, params)

        # Run baselines for comparison
        print("\n  Running baselines ...")
        sma_res = backtest_baseline(
            val_df, SmaCrossover,
            {"short_window": 10, "long_window": 50},
            buy_size=1000)
        simple_res = backtest_baseline(
            val_df, SimpleStrat1,
            {"buy_size": 100, "entry_drop": 1.0, "profit_target": 20.0,
             "sell_portion": 100.0, "stop_loss": 3.0, "max_buys": 1})

        # Dummy placeholder for the non-selected ML model in the table
        dummy = {"total_return_pct": 0, "sharpe_ratio": 0,
                 "win_rate_pct": 0, "max_drawdown_pct": 0, "num_trades": 0}
        print_comparison(best_res, dummy, sma_res, simple_res)
        ml_wins = beats_baselines(best_res, sma_res, simple_res)
        model_type = f"GridSearch ({best_combo['model_type']})"
    else:
        rf_model, gbt_model, rf_res, gbt_res, sma_res, simple_res = \
            evaluate_models(train_df, val_df, feature_cols, params)

        print_comparison(rf_res, gbt_res, sma_res, simple_res)

        # Pick the one with higher Sharpe
        if gbt_res["sharpe_ratio"] >= rf_res["sharpe_ratio"]:
            best_model, best_res, model_type = gbt_model, gbt_res, "GradientBoosting"
        else:
            best_model, best_res, model_type = rf_model, rf_res, "RandomForest"

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
            num_train_samples=len(train_df),
            num_val_samples=len(val_df),
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
    if hasattr(best_model, "feature_importances_"):
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
