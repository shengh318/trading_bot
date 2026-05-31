"""
Phase 4 — Regime Detection.

Detects market regimes — trending, mean-reverting, volatile, choppy — using
Hurst exponent, rolling volatility, VIX proxy, spread variance expansion,
and correlation breakdown detection.

Trading rules integrated here: disable during unstable regimes, suppress
signals when cointegration weakens, and flag structural breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import REGIME_VOL_WINDOW, VIX_TICKER, CUSUM_CONFIDENCE, CHOW_SIGNIFICANCE, BAI_PERRON_MAX_BREAKS, BAI_PERRON_MIN_SEGMENT

logger = logging.getLogger("stats_arb.regime")


class RegimeType(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    CHOPPY = "choppy"


@dataclass
class RegimeResult:
    """Regime detection results for a pair.

    Attributes
    ----------
    current_regime : RegimeType
        Current detected market regime.
    regime_series : pd.Series
        Full regime time series.
    hurst_regime : pd.Series
        Hurst-based regime classification.
    volatility_regime : pd.Series
        Volatility-based regime classification.
    rolling_hurst : pd.Series
        Rolling Hurst exponent time series.
    rolling_vol : pd.Series
        Rolling spread volatility.
    vix_level : float
        Most recent VIX level (0 if unavailable).
    correlation_breakdown : bool
        True if a correlation breakdown is currently detected.
    spread_variance_expansion : bool
        True if spread variance is expanding rapidly.
    trading_allowed : bool
        Whether trading is allowed in the current regime.
    signal_suppressed : bool
        Whether signals should be suppressed.
    structural_break : bool
        True if a structural break in the relationship is detected.
    regime_summary : dict[str, float]
        Percentage of time spent in each regime.
    cusum_break_detected : bool
        Whether CUSUM test detected a break.
    cusum_break_indices : list[int]
        Indices where CUSUM crosses the boundary.
    chow_break_detected : bool
        Whether Chow test detected a break.
    chow_break_dates : list[str]
        Dates of Chow break points.
    bai_perron_breaks : list[int]
        Indices of Bai-Perron break points.
    num_structural_breaks : int
        Total number of detected structural breaks.
    """

    current_regime: RegimeType
    regime_series: pd.Series
    hurst_regime: pd.Series
    volatility_regime: pd.Series
    rolling_hurst: pd.Series
    rolling_vol: pd.Series
    vix_level: float
    correlation_breakdown: bool
    spread_variance_expansion: bool
    trading_allowed: bool
    signal_suppressed: bool
    structural_break: bool
    regime_summary: dict[str, float]
    cusum_break_detected: bool = False
    cusum_break_indices: list[int] = field(default_factory=list)
    chow_break_detected: bool = False
    chow_break_dates: list[str] = field(default_factory=list)
    bai_perron_breaks: list[int] = field(default_factory=list)
    num_structural_breaks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_regime": self.current_regime.value,
            "vix_level": round(self.vix_level, 2),
            "correlation_breakdown": self.correlation_breakdown,
            "spread_variance_expansion": self.spread_variance_expansion,
            "trading_allowed": self.trading_allowed,
            "signal_suppressed": self.signal_suppressed,
            "structural_break": self.structural_break,
            "regime_summary": self.regime_summary,
            "cusum_break_detected": self.cusum_break_detected,
            "cusum_break_indices": self.cusum_break_indices,
            "chow_break_detected": self.chow_break_detected,
            "chow_break_dates": self.chow_break_dates,
            "bai_perron_breaks": self.bai_perron_breaks,
            "num_structural_breaks": self.num_structural_breaks,
        }


class RegimeDetector:
    """Detects market regimes relevant for pairs trading.

    Combines multiple signals:
        - Rolling Hurst exponent (trending vs mean-reverting)
        - Rolling spread volatility (normal vs volatile)
        - VIX level (market stress proxy)
        - Spread variance expansion (regime shift detection)
        - Correlation breakdown (relationship stability)

    Parameters
    ----------
    spread : pd.Series
        The spread time series.
    hurst_window : int
        Window for rolling Hurst calculation.
    vol_window : int
        Window for rolling volatility calculation.
    vix_data : pd.Series, optional
        External VIX time series. If None, attempts to use ^VIX ticker.
    """

    def __init__(
        self,
        spread: pd.Series,
        hurst_window: int = 63,
        vol_window: int = REGIME_VOL_WINDOW,
        vix_data: Optional[pd.Series] = None,
    ) -> None:
        self.spread = spread.dropna()
        self.hurst_window = hurst_window
        self.vol_window = vol_window
        self.vix_data = vix_data

    def detect(self) -> RegimeResult:
        """Run all regime detection logic and return comprehensive result."""
        rolling_hurst = self._compute_rolling_hurst()
        rolling_vol = self._compute_rolling_volatility()
        hurst_regime = self._classify_hurst_regime(rolling_hurst)
        vol_regime = self._classify_vol_regime(rolling_vol)

        regime_series = self._combine_regimes(hurst_regime, vol_regime)
        current_regime = self._get_current_regime(regime_series)

        vix = self._get_vix_level()
        corr_breakdown = self._detect_correlation_breakdown()
        var_expansion = self._detect_variance_expansion(rolling_vol)
        structural_break = self._detect_structural_break(rolling_hurst, rolling_vol)

        cusum_detected, cusum_indices, _ = self._detect_cusum_break()

        bai_perron_breaks = self._detect_bai_perron_breaks()

        chow_detected = False
        chow_dates: list[str] = []
        for bp in bai_perron_breaks[:1]:
            if self._detect_chow_break(bp):
                chow_detected = True
                try:
                    chow_dates.append(str(self.spread.index[bp].date()))
                except Exception:
                    chow_dates.append(str(bp))

        num_structural_breaks = len(bai_perron_breaks)
        overall_structural_break = (
            structural_break or cusum_detected or chow_detected or num_structural_breaks >= 2
        )

        trading_allowed = current_regime not in (
            RegimeType.TRENDING,
            RegimeType.VOLATILE,
        )
        signal_suppressed = corr_breakdown or var_expansion

        regime_summary = {
            regime.value: float((regime_series == regime.value).mean())
            for regime in RegimeType
        }

        return RegimeResult(
            current_regime=current_regime,
            regime_series=regime_series,
            hurst_regime=hurst_regime,
            volatility_regime=vol_regime,
            rolling_hurst=rolling_hurst,
            rolling_vol=rolling_vol,
            vix_level=vix,
            correlation_breakdown=corr_breakdown,
            spread_variance_expansion=var_expansion,
            trading_allowed=trading_allowed,
            signal_suppressed=signal_suppressed,
            structural_break=overall_structural_break,
            regime_summary=regime_summary,
            cusum_break_detected=cusum_detected,
            cusum_break_indices=cusum_indices,
            chow_break_detected=chow_detected,
            chow_break_dates=chow_dates,
            bai_perron_breaks=bai_perron_breaks,
            num_structural_breaks=num_structural_breaks,
        )

    def _compute_rolling_hurst(self) -> pd.Series:
        """Compute rolling Hurst exponent."""
        result: list[float] = []
        indices: list[pd.Timestamp] = []
        values = self.spread.values
        idx = self.spread.index

        for i in range(self.hurst_window, len(values) + 1):
            chunk = values[i - self.hurst_window : i]
            h = self._hurst_exponent(chunk)
            result.append(h)
            indices.append(idx[i - 1])

        return pd.Series(result, index=indices)

    def _compute_rolling_volatility(self) -> pd.Series:
        """Compute rolling spread standard deviation."""
        return self.spread.rolling(self.vol_window).std().dropna()

    @staticmethod
    def _hurst_exponent(ts: np.ndarray) -> float:
        """Rescaled range (R/S) Hurst exponent."""
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
            std_vals = reshaped.std(axis=1, ddof=0)
            std_vals = np.where(std_vals > 0, std_vals, np.nan)
            rs = (cumsum.max(axis=1) - cumsum.min(axis=1)) / std_vals
            rs = rs[~np.isnan(rs)]
            if len(rs) > 0:
                tau.append(float(rs.mean()))
        if len(tau) < 3:
            return 0.5
        lags_used = list(lags[: len(tau)])
        if len(lags_used) < 3:
            return 0.5
        reg = np.polyfit(np.log(lags_used), np.log(tau), 1)
        return max(0.0, min(1.0, float(reg[0])))

    @staticmethod
    def _classify_hurst_regime(hurst: pd.Series) -> pd.Series:
        regimes = pd.Series(index=hurst.index, dtype=str)
        regimes.loc[hurst < 0.4] = RegimeType.MEAN_REVERTING.value
        regimes.loc[(hurst >= 0.4) & (hurst <= 0.6)] = RegimeType.CHOPPY.value
        regimes.loc[hurst > 0.6] = RegimeType.TRENDING.value
        return regimes

    @staticmethod
    def _classify_vol_regime(vol: pd.Series) -> pd.Series:
        if vol.empty:
            return pd.Series(dtype=str)
        low = vol.quantile(0.33)
        high = vol.quantile(0.67)
        regimes = pd.Series(index=vol.index, dtype=str)
        regimes.loc[vol <= low] = RegimeType.MEAN_REVERTING.value
        regimes.loc[(vol > low) & (vol < high)] = RegimeType.CHOPPY.value
        regimes.loc[vol >= high] = RegimeType.VOLATILE.value
        return regimes

    @staticmethod
    def _combine_regimes(
        hurst_regime: pd.Series, vol_regime: pd.Series
    ) -> pd.Series:
        """Combine Hurst and volatility regimes.

        If both are available, volatile overrides, trending overrides.
        """
        combined = hurst_regime.copy()
        common_idx = combined.index.intersection(vol_regime.index)
        for idx in common_idx:
            h = combined.loc[idx]
            v = vol_regime.loc[idx]
            if v == RegimeType.VOLATILE.value:
                combined.loc[idx] = RegimeType.VOLATILE.value
            elif v == RegimeType.MEAN_REVERTING.value and h == RegimeType.CHOPPY.value:
                combined.loc[idx] = RegimeType.MEAN_REVERTING.value
        return combined

    @staticmethod
    def _get_current_regime(regime_series: pd.Series) -> RegimeType:
        if regime_series.empty:
            return RegimeType.CHOPPY
        latest = regime_series.iloc[-1]
        try:
            return RegimeType(latest)
        except ValueError:
            return RegimeType.CHOPPY

    def _get_vix_level(self) -> float:
        """Get VIX level as a market stress proxy."""
        if self.vix_data is not None and not self.vix_data.empty:
            return float(self.vix_data.iloc[-1])
        try:
            import yfinance as yf
            vix = yf.download(VIX_TICKER, period="5d", progress=False)
            if not vix.empty:
                close = vix["Close"].squeeze()
                return float(close.iloc[-1])
        except Exception:
            pass
        return 0.0

    def _detect_correlation_breakdown(self) -> bool:
        """Detect if the relationship is breaking down.

        Uses spread variance changes as a proxy for correlation breakdown.
        """
        if len(self.spread) < 40:
            return False
        recent = self.spread.iloc[-20:]
        earlier = self.spread.iloc[-40:-20]
        if earlier.std() == 0:
            return False
        vol_ratio = recent.std() / earlier.std()
        return bool(vol_ratio > 2.5)

    def _detect_variance_expansion(self, rolling_vol: pd.Series) -> bool:
        """Detect rapid variance expansion (regime shift signal)."""
        if len(rolling_vol) < 20:
            return False
        recent = rolling_vol.iloc[-5:].mean()
        earlier = rolling_vol.iloc[-20:-5].mean()
        if earlier == 0:
            return False
        return bool(recent / earlier > 2.0)

    def _detect_cusum_break(
        self, confidence: float = CUSUM_CONFIDENCE,
    ) -> tuple[bool, list[int], np.ndarray]:
        """CUSUM test via recursive residuals.

        Fits OLS of spread ~ constant, computes recursive residuals,
        and checks if cumulative sum exceeds confidence bounds.
        """
        n = len(self.spread)
        if n < 30:
            return False, [], np.array([])

        y = self.spread.values
        x = np.ones((n, 1))
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        recursive_residuals: list[float] = []

        for t in range(2, n + 1):
            y_t = y[:t]
            x_t = np.ones((t, 1))
            beta_t, _, _, _ = np.linalg.lstsq(x_t, y_t, rcond=None)
            pred = beta_t[0]
            err = y[t - 1] - pred
            denom = np.sqrt(1.0 + 1.0 / t)
            recursive_residuals.append(err / denom)

        rr = np.array(recursive_residuals)
        sigma = np.std(rr)
        if sigma == 0:
            return False, [], np.array([])

        std_rr = rr / sigma
        cusum = np.cumsum(std_rr)

        z = {0.90: 0.850, 0.95: 0.948, 0.99: 1.143}.get(confidence, 0.948)
        bound = z * np.sqrt(n - 1)
        breaks = np.where(np.abs(cusum) > bound)[0].tolist()
        has_break = len(breaks) > 0

        return has_break, breaks, cusum

    def _detect_chow_break(
        self, candidate_break: int, significance: float = CHOW_SIGNIFICANCE,
    ) -> bool:
        """Chow breakpoint F-test.

        Tests restricted (single regression) vs unrestricted (split at point).
        F = ((RSS_pooled - RSS_split) / k) / (RSS_split / (n - 2k))
        where k = number of parameters, n = total observations.
        """
        n = len(self.spread)
        if candidate_break < 10 or candidate_break > n - 10:
            return False

        y = self.spread.values
        x = np.ones((n, 1))
        beta_pooled, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        rss_pooled = float(np.sum((y - x @ beta_pooled) ** 2))

        y1, y2 = y[:candidate_break], y[candidate_break:]
        x1, x2 = np.ones((len(y1), 1)), np.ones((len(y2), 1))

        beta1, _, _, _ = np.linalg.lstsq(x1, y1, rcond=None)
        beta2, _, _, _ = np.linalg.lstsq(x2, y2, rcond=None)
        rss1 = float(np.sum((y1 - x1 @ beta1) ** 2))
        rss2 = float(np.sum((y2 - x2 @ beta2) ** 2))
        rss_split = rss1 + rss2

        k = 1
        f_stat = ((rss_pooled - rss_split) / k) / (rss_split / (n - 2 * k))

        from scipy import stats as scipy_stats
        p_value = 1.0 - scipy_stats.f.cdf(f_stat, k, n - 2 * k)
        return bool(p_value < significance)

    def _detect_bai_perron_breaks(
        self, max_breaks: int = BAI_PERRON_MAX_BREAKS, min_segment: int = BAI_PERRON_MIN_SEGMENT,
    ) -> list[int]:
        """Sequential breakpoint detection (simplified Bai-Perron).

        1. Find single break via argmin RSS over all valid positions
        2. Split at break, recurse on each segment
        3. Stop when: BIC doesn't improve, max_breaks reached, or segment too small
        """
        def _find_single_break(y: np.ndarray) -> tuple[int, float, float]:
            n = len(y)
            best_bic = float("inf")
            best_pos = -1
            x = np.ones((n, 1))
            beta_pooled, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            rss_pooled = float(np.sum((y - x @ beta_pooled) ** 2))
            pooled_bic = n * np.log(rss_pooled / n) + 2 * np.log(n)

            for pos in range(min_segment, n - min_segment):
                y1, y2 = y[:pos], y[pos:]
                x1, x2 = np.ones((len(y1), 1)), np.ones((len(y2), 1))
                beta1, _, _, _ = np.linalg.lstsq(x1, y1, rcond=None)
                beta2, _, _, _ = np.linalg.lstsq(x2, y2, rcond=None)
                rss1 = float(np.sum((y1 - x1 @ beta1) ** 2))
                rss2 = float(np.sum((y2 - x2 @ beta2) ** 2))
                rss = rss1 + rss2
                k = 4
                bic = n * np.log(rss / n) + k * np.log(n)
                if bic < best_bic:
                    best_bic = bic
                    best_pos = pos

            return best_pos, best_bic, pooled_bic

        breaks: list[int] = []
        segments = [(0, len(self.spread.values))]
        y_full = self.spread.values

        while len(breaks) < max_breaks:
            best_seg_idx = -1
            best_break = -1
            best_bic_improvement = 0.0
            best_pos_local = -1

            for seg_idx, (start, end) in enumerate(segments):
                y_seg = y_full[start:end]
                if len(y_seg) < 2 * min_segment:
                    continue
                pos, bic, pooled_bic = _find_single_break(y_seg)
                if pos < 0:
                    continue
                improvement = pooled_bic - bic
                if improvement > best_bic_improvement:
                    best_bic_improvement = improvement
                    best_seg_idx = seg_idx
                    best_break = start + pos
                    best_pos_local = pos

            if best_break < 0 or best_bic_improvement <= 0:
                break

            breaks.append(best_break)
            seg_start, seg_end = segments.pop(best_seg_idx)
            segments.append((seg_start, best_break))
            segments.append((best_break, seg_end))
            segments.sort()

        return sorted(breaks)

    def _detect_structural_break(
        self, rolling_hurst: pd.Series, rolling_vol: pd.Series
    ) -> bool:
        """Detect structural break using simultaneous Hurst + vol change."""
        if len(rolling_hurst) < 20 or len(rolling_vol) < 20:
            return False
        h_recent = rolling_hurst.iloc[-10:].mean()
        h_earlier = rolling_hurst.iloc[-20:-10].mean()
        v_recent = rolling_vol.iloc[-10:].mean()
        v_earlier = rolling_vol.iloc[-20:-10].mean()

        h_shift = abs(h_recent - h_earlier) > 0.2
        if v_earlier == 0:
            return bool(h_shift)
        v_shift = v_recent / v_earlier > 2.0
        return bool(h_shift and v_shift)
