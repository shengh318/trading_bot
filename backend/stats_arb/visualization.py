"""
Visualization module for statistical arbitrage analysis.

Creates publication-quality plots for pair analysis, backtest results,
regime detection, and ML performance.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

logger = logging.getLogger("stats_arb.viz")


class Visualizer:
    """Creates publication-quality plots for statistical arbitrage analysis.

    Parameters
    ----------
    save_dir : str
        Directory to save plots.
    style : str
        Matplotlib style.
    """

    def __init__(
        self,
        save_dir: str = "stats_arb_plots",
        style: str = "seaborn-v0_8-darkgrid",
    ) -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            plt.style.use(style)
        except Exception:
            pass

    def plot_pair_analysis(
        self,
        result: Any,
        prices: pd.DataFrame,
        prefix: str = "",
    ) -> dict[str, str]:
        """Generate all plots for a single pair analysis.

        Returns dict of plot name → file path.
        """
        paths: dict[str, str] = {}
        tickers = f"{result.ticker_a}_{result.ticker_b}"
        pfx = f"{prefix}_{tickers}" if prefix else tickers

        paths["normalized_prices"] = self._plot_normalized_prices(
            prices, result.ticker_a, result.ticker_b, pfx,
        )
        paths["spread"] = self._plot_spread(result, pfx)
        paths["zscore"] = self._plot_zscore(result, pfx)
        paths["rolling_correlation"] = self._plot_rolling_correlation(result, pfx)
        paths["equity_curve"] = self._plot_equity_curve(result, pfx)
        paths["drawdown"] = self._plot_drawdown(result, pfx)
        paths["regime"] = self._plot_regime(result, pfx)
        paths["walk_forward"] = self._plot_walk_forward(result, pfx)

        if result.ml_results:
            paths["ml_feature_importance"] = self._plot_ml_feature_importance(result, pfx)

        return paths

    def _plot_normalized_prices(
        self,
        prices: pd.DataFrame,
        ticker_a: str,
        ticker_b: str,
        pfx: str,
    ) -> str:
        fig, ax = plt.subplots()
        norm = prices / prices.iloc[0]
        norm.plot(ax=ax, title=f"Normalized Prices: {ticker_a} vs {ticker_b}")
        ax.set_ylabel("Normalized Price (base=1)")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        path = os.path.join(self.save_dir, f"{pfx}_prices.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_spread(self, result: Any, pfx: str) -> str:
        fig, ax = plt.subplots()
        spread = result.spread.spread
        spread.plot(ax=ax, title=f"Spread: {result.ticker_a} - β × {result.ticker_b}")
        ax.axhline(spread.mean(), color="r", ls="--", label="Mean")
        ax.axhline(
            spread.mean() + 2 * spread.std(), color="g", ls="--", alpha=0.5, label="±2σ"
        )
        ax.axhline(spread.mean() - 2 * spread.std(), color="g", ls="--", alpha=0.5)
        ax.legend()
        path = os.path.join(self.save_dir, f"{pfx}_spread.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_zscore(self, result: Any, pfx: str) -> str:
        fig, ax = plt.subplots(figsize=(14, 7))
        z = result.spread.zscore_series
        z.plot(ax=ax, title=f"Z-Score: {result.ticker_a} vs {result.ticker_b}", color="purple")
        ax.axhline(0, color="gray", ls="-", alpha=0.5)
        ax.axhline(2, color="r", ls="--", alpha=0.7, label="+2σ Entry")
        ax.axhline(-2, color="g", ls="--", alpha=0.7, label="-2σ Entry")
        ax.axhline(3, color="r", ls=":", alpha=0.4, label="+3σ Stop")
        ax.axhline(-3, color="g", ls=":", alpha=0.4, label="-3σ Stop")
        ax.fill_between(z.index, -2, 2, alpha=0.08, color="gray")

        bt = getattr(result, "in_sample_backtest", None) or getattr(result, "backtest", None)
        if bt and hasattr(bt, "trades"):
            trades = bt.trades
            for t in trades:
                entry_date = self._safe_parse_date(t, "entry_date")
                exit_date = self._safe_parse_date(t, "exit_date")
                entry_z = self._get_attr(t, "entry_zscore", 0)
                if entry_date and entry_date in z.index:
                    ax.scatter(entry_date, entry_z, color="blue", s=30, marker="^", zorder=5)
                if exit_date and exit_date in z.index:
                    exit_z = self._get_attr(t, "exit_zscore", 0)
                    ax.scatter(exit_date, exit_z, color="red", s=30, marker="v", zorder=5)

        ax.legend(loc="best")
        path = os.path.join(self.save_dir, f"{pfx}_zscore.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_rolling_correlation(self, result: Any, pfx: str) -> str:
        fig, ax = plt.subplots()
        corr = result.correlation
        if corr:
            for w, res in corr.items():
                series = res.series if hasattr(res, "series") else res.get("series", pd.Series())
                if not series.empty:
                    series.plot(ax=ax, label=f"{w}d window", alpha=0.8)
        ax.set_title(f"Rolling Correlation: {result.ticker_a} vs {result.ticker_b}")
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        ax.legend()
        path = os.path.join(self.save_dir, f"{pfx}_correlation.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_equity_curve(self, result: Any, pfx: str) -> str:
        bt = getattr(result, "in_sample_backtest", None) or getattr(result, "backtest", None)
        if bt is None or not hasattr(bt, "equity_curve") or bt.equity_curve.empty:
            fig, ax = plt.subplots()
            ax.set_title("No backtest data")
            path = os.path.join(self.save_dir, f"{pfx}_equity.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        fig, ax = plt.subplots()
        eq = bt.equity_curve
        cum_ret = (eq / eq.iloc[0] - 1) * 100
        cum_ret.plot(ax=ax, title=f"Cumulative Return: {result.ticker_a} vs {result.ticker_b}", color="green")
        ax.set_ylabel("Cumulative Return (%)")
        ax.axhline(0, color="gray", ls="--")

        self._mark_trades(ax, bt.trades, cum_ret)
        ax.legend(loc="best")
        path = os.path.join(self.save_dir, f"{pfx}_equity.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_drawdown(self, result: Any, pfx: str) -> str:
        bt = getattr(result, "in_sample_backtest", None) or getattr(result, "backtest", None)
        if bt is None or not hasattr(bt, "drawdown_series") or bt.drawdown_series.empty:
            fig, ax = plt.subplots()
            ax.set_title("No drawdown data")
            path = os.path.join(self.save_dir, f"{pfx}_drawdown.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        fig, ax = plt.subplots()
        bt.drawdown_series.plot(ax=ax, title="Drawdown", color="red", alpha=0.7)
        ax.fill_between(bt.drawdown_series.index, 0, bt.drawdown_series.values, color="red", alpha=0.2)
        ax.set_ylabel("Drawdown (%)")
        ax.axhline(0, color="gray", ls="--")
        path = os.path.join(self.save_dir, f"{pfx}_drawdown.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_regime(self, result: Any, pfx: str) -> str:
        regime = result.regime
        if regime is None:
            fig, ax = plt.subplots()
            ax.set_title("No regime data")
            path = os.path.join(self.save_dir, f"{pfx}_regime.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        rs = regime.regime_series
        colors = {
            "mean_reverting": "green",
            "choppy": "gray",
            "trending": "red",
            "volatile": "orange",
        }
        if not rs.empty:
            for regime_type, color in colors.items():
                mask = (rs == regime_type).values
                axes[0].fill_between(
                    rs.index, 0, 1,
                    where=mask,
                    color=color, alpha=0.3, label=regime_type,
                )
            axes[0].set_title("Regime Classification")
            axes[0].legend(loc="upper right")
            axes[0].set_ylim(0, 1)
            axes[0].set_yticks([])

        rh = regime.rolling_hurst
        if not rh.empty:
            axes[1].plot(rh.index, rh.values, color="purple", label="Hurst")
            axes[1].axhline(0.5, color="gray", ls="--", alpha=0.5, label="Random Walk")
            axes[1].axhline(0.4, color="green", ls=":", alpha=0.3)
            axes[1].axhline(0.6, color="red", ls=":", alpha=0.3)
            axes[1].set_title("Rolling Hurst Exponent")
            axes[1].legend()

        rv = regime.rolling_vol
        if not rv.empty:
            axes[2].plot(rv.index, rv.values, color="blue", label="Spread Volatility")
            axes[2].set_title("Rolling Spread Volatility")
            axes[2].legend()

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{pfx}_regime.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_walk_forward(self, result: Any, pfx: str) -> str:
        wf = result.walk_forward
        if wf is None or not wf.folds:
            fig, ax = plt.subplots()
            ax.set_title("No walk-forward data")
            path = os.path.join(self.save_dir, f"{pfx}_walk_forward.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        folds = wf.folds
        fold_indices = list(range(len(folds)))

        oos_p_vals = [f.oos_p_value for f in folds]
        axes[0].bar(fold_indices, oos_p_vals, color="steelblue")
        axes[0].axhline(0.05, color="r", ls="--", label="α=0.05")
        axes[0].set_ylabel("OOS ADF p-value")
        axes[0].set_title("Walk-Forward: OOS Stationarity (ADF)")
        axes[0].legend()

        hl = [f.half_life if np.isfinite(f.half_life) else 0 for f in folds]
        axes[1].bar(fold_indices, hl, color="orange")
        axes[1].set_ylabel("Half-Life (days)")
        axes[1].set_title("Walk-Forward: Half-Life")

        sharps = [f.spread_sharpe for f in folds]
        colors = ["green" if s > 0 else "red" for s in sharps]
        axes[2].bar(fold_indices, sharps, color=colors)
        axes[2].axhline(0, color="gray", ls="-")
        axes[2].set_ylabel("Spread Sharpe (ann.)")
        axes[2].set_title("Walk-Forward: Out-of-Sample Spread Sharpe")
        axes[2].set_xlabel("Fold")

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{pfx}_walk_forward.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_ml_feature_importance(self, result: Any, pfx: str) -> str:
        ml = result.ml_results
        if not ml:
            fig, ax = plt.subplots()
            ax.set_title("No ML data")
            path = os.path.join(self.save_dir, f"{pfx}_ml_features.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        first_key = list(ml.keys())[0]
        first_result = ml[first_key]
        importances = first_result.feature_importance
        if not importances:
            path = os.path.join(self.save_dir, f"{pfx}_ml_features.png")
            fig, ax = plt.subplots()
            ax.set_title("No feature importance data")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            return path

        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
        names, vals = zip(*sorted_items)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(names)), vals, color="teal")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance")
        ax.set_title(f"ML Feature Importance ({first_key})")
        ax.invert_yaxis()

        plt.tight_layout()
        path = os.path.join(self.save_dir, f"{pfx}_ml_features.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def plot_cointegration_heatmap(
        matrix: pd.DataFrame,
        title: str = "Pairwise Cointegration Score (-log10 p-value)",
        save_path: str = "stats_arb_plots/cointegration_heatmap.png",
    ) -> str:
        """Plot a heatmap of pairwise cointegration scores."""
        try:
            import seaborn as sns
        except ImportError:
            logger.warning("seaborn not available for heatmap")
            return ""

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(
            figsize=(max(8, len(matrix) * 0.6), max(6, len(matrix) * 0.5))
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "-log10(p-value)"},
            mask=matrix.isnull().values if matrix.isnull().values.any() else None,
        )
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return save_path

    @staticmethod
    def plot_ranking_summary(
        ranked_pairs: list[Any],
        save_path: str = "stats_arb_plots/ranking_summary.png",
    ) -> str:
        """Plot a bar chart of top-ranked pairs."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(ranked_pairs) * 0.4)))

        names = [f"{p.ticker_a}/{p.ticker_b}" for p in ranked_pairs]
        scores = [p.composite_score for p in ranked_pairs]
        colors = ["green" if s >= 0.6 else "orange" if s >= 0.3 else "red" for s in scores]

        ax.barh(range(len(names)), scores, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Composite Score")
        ax.set_title("Top Ranked Pairs")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)

        for i, s in enumerate(scores):
            ax.text(s + 0.01, i, f"{s:.3f}", va="center")

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    @staticmethod
    def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default

    @staticmethod
    def _mark_trades(
        ax: plt.Axes, trades: list[Any], cum_ret: pd.Series,
    ) -> None:
        entry_dates: list[pd.Timestamp] = []
        exit_dates: list[pd.Timestamp] = []
        entry_vals: list[float] = []
        exit_vals: list[float] = []

        for t in trades:
            entry = Visualizer._safe_parse_date(t, "entry_date")
            exit_ = Visualizer._safe_parse_date(t, "exit_date")
            if entry and entry in cum_ret.index:
                entry_dates.append(entry)
                entry_vals.append(float(cum_ret.loc[entry]))
            if exit_ and exit_ in cum_ret.index:
                exit_dates.append(exit_)
                exit_vals.append(float(cum_ret.loc[exit_]))

        if entry_dates:
            ax.scatter(
                entry_dates, entry_vals, color="blue",
                s=25, marker="^", label="Entry", zorder=5,
            )
        if exit_dates:
            ax.scatter(
                exit_dates, exit_vals, color="red",
                s=25, marker="v", label="Exit", zorder=5,
            )

    @staticmethod
    def _safe_parse_date(t: Any, key: str) -> Optional[pd.Timestamp]:
        try:
            val = Visualizer._get_attr(t, key, "")
            return pd.Timestamp(val) if val else None
        except (ValueError, TypeError):
            return None
