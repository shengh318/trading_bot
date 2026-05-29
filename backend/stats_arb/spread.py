"""
Phase 3 — Spread Modeling.

Computes spread statistics, z-scores, stationarity tests, Ornstein-Uhlenbeck
half-life, Hurst exponent, mean reversion speed, and spread persistence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf

logger = logging.getLogger("stats_arb.spread")


@dataclass
class SpreadResult:
    """Comprehensive spread analysis result.

    Attributes
    ----------
    mean : float
        Mean of the spread series.
    std : float
        Standard deviation.
    current_zscore : float
        Most recent z-score value.
    zscore_series : pd.Series
        Full z-score time series.
    spread : pd.Series
        The spread series.
    half_life : float
        Estimated half-life of mean reversion (days).
    hurst_exponent : float
        Hurst exponent (H < 0.5 = mean-reverting).
    adf_statistic : float
        Augmented Dickey-Fuller test statistic.
    adf_pvalue : float
        ADF p-value.
    is_stationary : bool
        True if spread is stationary at the 5% level.
    mean_reversion_speed : float
        Theta parameter from OU process (negative = mean-reverting).
    expected_time_to_mean : float
        Expected time (days) to revert to mean from current deviation.
    persistence : float
        First-order autocorrelation of the spread (0 = white noise).
    spread_autocorr_5 : float
        Autocorrelation at 5-day lag.
    variance_ratio : float
        Variance ratio (20d / 5d), > 1 suggests trending.
    variance_explosion_events : int
        Count of times rolling volatility exceeds 3x its expanding mean.
    """

    mean: float
    std: float
    current_zscore: float
    zscore_series: pd.Series
    spread: pd.Series
    half_life: float
    hurst_exponent: float
    adf_statistic: float
    adf_pvalue: float
    is_stationary: bool
    mean_reversion_speed: float
    expected_time_to_mean: float
    persistence: float
    spread_autocorr_5: float
    variance_ratio: float
    variance_explosion_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "current_zscore": round(self.current_zscore, 4),
            "half_life": round(self.half_life, 2),
            "hurst_exponent": round(self.hurst_exponent, 4),
            "adf_statistic": round(self.adf_statistic, 4),
            "adf_pvalue": round(self.adf_pvalue, 6),
            "is_stationary": self.is_stationary,
            "mean_reversion_speed": round(self.mean_reversion_speed, 6),
            "expected_time_to_mean": round(self.expected_time_to_mean, 2),
            "persistence": round(self.persistence, 4),
            "spread_autocorr_5": round(self.spread_autocorr_5, 4),
            "variance_ratio": round(self.variance_ratio, 4),
            "variance_explosion_events": self.variance_explosion_events,
        }


class SpreadAnalyzer:
    """Computes spread statistics and mean-reversion diagnostics.

    The spread is the residual from the cointegrating regression:
        spread = price_A - beta * price_B

    Financial intuition:
        A stationary spread with half-life in the 5–60 day range and
        Hurst < 0.5 indicates a mean-reverting relationship suitable
        for pairs trading.

    Parameters
    ----------
    spread : pd.Series
        The spread series (residual from cointegration).
    """

    def __init__(self, spread: pd.Series) -> None:
        self.spread = spread.dropna()

    def run(self) -> SpreadResult:
        spread = self.spread
        mean = float(spread.mean())
        std = float(spread.std())
        zscore_series = (spread - mean) / std
        current_zscore = float(zscore_series.iloc[-1])

        half_life = self._estimate_half_life(spread.values)
        hurst_exp = self._hurst_exponent(spread.values)

        adf_full = adfuller(spread.values, maxlag=1, autolag="AIC")
        adf_stat, adf_p = float(adf_full[0]), float(adf_full[1])
        is_stationary = bool(adf_p < 0.05)

        speed = self._ou_mean_reversion_speed(spread.values)
        expected_time = self._expected_time_to_mean(current_zscore, speed) if speed < 0 else float("inf")
        persistence = float(spread.autocorr(lag=1)) if len(spread) > 2 else 0.0
        autocorr_5 = float(spread.autocorr(lag=5)) if len(spread) > 5 else 0.0
        var_ratio = self._variance_ratio(spread.values)

        rolling_vol = spread.rolling(63).std()
        expanding_mean_vol = rolling_vol.expanding().mean()
        variance_explosion_events = int(
            (rolling_vol / expanding_mean_vol > 3.0).sum()
        ) if len(spread) > 63 else 0

        return SpreadResult(
            mean=mean,
            std=std,
            current_zscore=current_zscore,
            zscore_series=zscore_series,
            spread=spread,
            half_life=half_life,
            hurst_exponent=hurst_exp,
            adf_statistic=float(adf_stat),
            adf_pvalue=float(adf_p),
            is_stationary=is_stationary,
            mean_reversion_speed=speed,
            expected_time_to_mean=expected_time,
            persistence=persistence,
            spread_autocorr_5=autocorr_5,
            variance_ratio=var_ratio,
            variance_explosion_events=variance_explosion_events,
        )

    @staticmethod
    def _estimate_half_life(spread: np.ndarray) -> float:
        """Estimate half-life of mean reversion via OLS on lagged spread.

        Regress Δspreadₜ = α + θ * spreadₜ₋₁ + εₜ.
        Half-life = -ln(2) / θ  (θ < 0 → mean reversion).
        """
        if len(spread) < 10:
            return float("inf")

        spread_series = pd.Series(spread)
        lagged = spread_series.shift(1).dropna().values
        delta = spread_series.diff().dropna().values

        if len(lagged) < 5:
            return float("inf")

        lagged_with_const = np.column_stack([np.ones_like(lagged), lagged])
        try:
            theta, _, _, _ = np.linalg.lstsq(lagged_with_const, delta, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf")

        theta_val = theta[1]
        if theta_val >= 0:
            return float("inf")

        half_life = -math.log(2) / theta_val
        return float(half_life) if np.isfinite(half_life) else float("inf")

    @staticmethod
    def _hurst_exponent(ts: np.ndarray) -> float:
        """Compute Hurst exponent via rescaled range (R/S) analysis.

        H < 0.5 → mean-reverting (anti-persistent).
        H = 0.5 → random walk.
        H > 0.5 → trending (persistent).
        """
        if len(ts) < 20:
            return 0.5

        ts = np.asarray(ts)
        max_lag = len(ts) // 2
        if max_lag < 3:
            return 0.5
        lags = range(2, max_lag)
        tau: list[float] = []
        for lag in lags:
            chunks = len(ts) // lag
            if chunks < 1:
                continue
            trimmed = ts[: chunks * lag]
            reshaped = trimmed.reshape((chunks, lag))
            mean_adj = reshaped - reshaped.mean(axis=1, keepdims=True)
            cumsum = mean_adj.cumsum(axis=1)
            rs = (cumsum.max(axis=1) - cumsum.min(axis=1)) / reshaped.std(axis=1, ddof=0)
            rs = rs[~np.isnan(rs)]
            if len(rs) > 0:
                tau.append(float(rs.mean()))

        if len(tau) < 3:
            return 0.5

        lags_used = list(lags[: len(tau)])
        if len(lags_used) < 3:
            return 0.5

        reg = np.polyfit(np.log(lags_used), np.log(tau), 1)
        h = float(reg[0])
        return max(0.0, min(1.0, h))

    @staticmethod
    def _ou_mean_reversion_speed(spread: np.ndarray) -> float:
        """Estimate OU mean reversion speed θ.

        Fits Δspreadₜ = θ * (μ - spreadₜ₋₁) * Δt + σ * εₜ.
        Returns θ (negative = mean-reverting).
        """
        if len(spread) < 10:
            return 0.0
        s = pd.Series(spread)
        lagged = s.shift(1).dropna().values
        delta = s.diff().dropna().values
        lagged_const = np.column_stack([np.ones_like(lagged), lagged])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(lagged_const, delta, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        return float(coeffs[1])

    @staticmethod
    def _expected_time_to_mean(current_z: float, speed: float) -> float:
        """Expected time (days) for spread to revert halfway to mean.

        For an OU process, expected first-passage time scales with
        -ln(2) / θ. We scale by the current deviation magnitude.
        """
        if speed >= 0:
            return float("inf")
        hl = -math.log(2) / speed
        scaling = min(max(abs(current_z) / 2.0, 0.5), 3.0)
        return hl * scaling

    @staticmethod
    def _variance_ratio(spread: np.ndarray, period1: int = 5, period2: int = 20) -> float:
        """Variance ratio VR = Var(spread_{t+20} - spread_t) / (20/5 * Var(spread_{t+5} - spread_t)).

        VR > 1  → positive autocorrelation (trending).
        VR < 1  → negative autocorrelation (mean-reverting).
        """
        if len(spread) < period2 + 1:
            return 1.0
        s = pd.Series(spread)
        var_short = s.diff(period1).var()
        var_long = s.diff(period2).var()
        if var_short == 0:
            return 1.0
        return float(var_long / (var_short * (period2 / period1)))
