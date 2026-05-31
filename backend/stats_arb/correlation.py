"""
Phase 2 — Relationship Discovery: Correlation Analysis.

Computes rolling Pearson correlations across multiple windows and
quantifies their stability, drift, and regime changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.cpp_ext import rolling_slope

from .config import CORRELATION_WINDOWS

logger = logging.getLogger("stats_arb.correlation")


@dataclass
class RollingCorrelationResult:
    """Results from a single rolling correlation window.

    Attributes
    ----------
    window : int
        Rolling window length in days.
    current : float
        Most recent correlation value.
    mean : float
        Mean correlation over the full period.
    std : float
        Standard deviation (lower = more stable).
    min_val : float
        Minimum correlation observed.
    max_val : float
        Maximum correlation observed.
    overall_slope : float
        Polyfit slope over full series (replaces drift).
    threshold_crossings : dict[str, int]
        Count of crossings below 0.5, 0.3, and sign flips.
    correlation_collapse : bool
        True if current correlation < 0.3 but historical mean > 0.6.
    max_correlation_drawdown : float
        Maximum drawdown from peak correlation.
    stability_score : float
        Composite stability score in [0, 1].
    rolling_slope_series : pd.Series
        63d window slope time series.
    series : pd.Series
        Full rolling correlation time series.
    """

    window: int
    current: float
    mean: float
    std: float
    min_val: float
    max_val: float
    overall_slope: float
    threshold_crossings: dict[str, int]
    correlation_collapse: bool
    max_correlation_drawdown: float
    stability_score: float
    rolling_slope_series: pd.Series
    series: pd.Series

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "current": round(self.current, 4),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
            "overall_slope": round(self.overall_slope, 6),
            "threshold_crossings": self.threshold_crossings,
            "correlation_collapse": self.correlation_collapse,
            "max_correlation_drawdown": round(self.max_correlation_drawdown, 4),
            "stability_score": round(self.stability_score, 4),
        }


def _rolling_correlation(
    series_a: pd.Series, series_b: pd.Series, window: int
) -> pd.Series:
    """Compute rolling Pearson correlation between two return series."""
    returns_a = series_a.pct_change().dropna()
    returns_b = series_b.pct_change().dropna()
    common_idx = returns_a.index.intersection(returns_b.index)
    return (
        returns_a.loc[common_idx]
        .rolling(window)
        .corr(returns_b.loc[common_idx])
        .dropna()
    )


class CorrelationAnalyzer:
    """Analyzes rolling Pearson correlations between two price series.

    Correlation ≠ cointegration. High correlation does not guarantee a
    tradeable pairs relationship, but *unstable* correlation signals
    that the relationship may be breaking down.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with columns [ticker_a, ticker_b].
    windows : list[int], optional
        Rolling windows to analyze.
    """

    def __init__(
        self, prices: pd.DataFrame, windows: list[int] | None = None
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.windows = windows or CORRELATION_WINDOWS
        self.results: dict[int, RollingCorrelationResult] = {}

    def run(self) -> dict[int, RollingCorrelationResult]:
        """Compute rolling correlations for all configured windows."""
        series_a = self.prices[self.cols[0]]
        series_b = self.prices[self.cols[1]]

        for w in self.windows:
            corr_series = _rolling_correlation(series_a, series_b, w)
            if corr_series.empty:
                continue

            # Overall slope (replaces drift)
            x = np.arange(len(corr_series))
            overall_slope = float(np.polyfit(x, corr_series.values, 1)[0])

            # Rolling slope (63d window) — C++ accelerated (O(1) per step)
            roll_slope_arr = rolling_slope(corr_series.values, 63)
            rolling_slope_series = pd.Series(roll_slope_arr, index=corr_series.index).dropna()

            # Threshold crossings (replaces regime_changes)
            threshold_crossings = {
                "below_0.5": int(((corr_series.shift(1) >= 0.5) & (corr_series < 0.5)).sum()),
                "below_0.3": int(((corr_series.shift(1) >= 0.3) & (corr_series < 0.3)).sum()),
                "sign_flip": int(((corr_series.shift(1) * corr_series) < 0).sum()),
            }

            # Correlation collapse
            correlation_collapse = bool(
                corr_series.iloc[-1] < 0.3 and corr_series.mean() > 0.6
            )

            # Correlation drawdown
            running_max = corr_series.cummax()
            # Avoid division issues when cummax is near zero
            corr_drawdown = 1.0 - (corr_series / running_max.replace(0, float("nan")))
            corr_drawdown = corr_drawdown.fillna(0).clip(0, 1)
            max_correlation_drawdown = float(corr_drawdown.max())

            # Stability score
            vol_of_corr = corr_series.rolling(63).std().mean()
            stability_score = 1.0 / (1.0 + vol_of_corr + abs(overall_slope) + max_correlation_drawdown)

            self.results[w] = RollingCorrelationResult(
                window=w,
                current=float(corr_series.iloc[-1]),
                mean=float(corr_series.mean()),
                std=float(corr_series.std()),
                min_val=float(corr_series.min()),
                max_val=float(corr_series.max()),
                overall_slope=overall_slope,
                threshold_crossings=threshold_crossings,
                correlation_collapse=correlation_collapse,
                max_correlation_drawdown=max_correlation_drawdown,
                stability_score=stability_score,
                rolling_slope_series=rolling_slope_series,
                series=corr_series,
            )

        return self.results
