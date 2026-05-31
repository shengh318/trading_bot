"""
Phase 8 — Pair Ranking System.

Scores and ranks all tested pairs using a composite of statistical,
risk-adjusted, and out-of-sample metrics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("stats_arb.ranking")


def bonferroni_correct(p_values: list[float], n_tests: int) -> list[float]:
    """Bonferroni correction: p_adj = min(p * n, 1.0)"""
    return [min(p * n_tests, 1.0) for p in p_values]


def benjamini_hochberg_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Sort p-values ascending, rank them, compute:
        p_adj = p * n / rank
    then enforce monotonicity by taking cumulative minimum
    from the largest p-value downward.
    """
    n = len(p_values)
    if n == 0:
        return []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result.tolist()


@dataclass
class RankedPair:
    """A ranked pair with scoring breakdown.

    Attributes
    ----------
    ticker_a : str
        First ticker.
    ticker_b : str
        Second ticker.
    composite_score : float
        Overall composite score (0–1).
    confidence : str
        'high', 'medium', or 'low'.
    risk : str
        'low', 'medium', or 'high'.
    score_components : dict[str, float]
        Individual component scores.
    metrics : dict[str, Any]
        Raw metrics from the pair analysis.
    raw_p_value : float
        Unadjusted cointegration p-value.
    adjusted_p_value : float
        Multiple-comparison adjusted p-value.
    """

    ticker_a: str
    ticker_b: str
    composite_score: float
    confidence: str
    risk: str
    score_components: dict[str, float]
    metrics: dict[str, Any]
    raw_p_value: float = 1.0
    adjusted_p_value: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker_a": self.ticker_a,
            "ticker_b": self.ticker_b,
            "composite_score": round(self.composite_score, 4),
            "confidence": self.confidence,
            "risk": self.risk,
            "score_components": {k: round(v, 4) for k, v in self.score_components.items()},
            "raw_p_value": round(self.raw_p_value, 6),
            "adjusted_p_value": round(self.adjusted_p_value, 6),
        }


class PairRanker:
    """Ranks stock pairs by a composite score for statistical arbitrage suitability.

    Scoring dimensions (configurable weights):
        coint_p                : Cointegration significance (lower p → higher score)
        coint_persistence      : Walk-forward cointegration % (higher → better)
        spread_sharpe          : Out-of-sample spread Sharpe
        half_life_score        : Half-life in ideal 5–60d range
        max_drawdown           : Lower drawdown → higher score
        hurst                  : H < 0.5 → higher score
        oos_robustness         : Out-of-sample stationarity %
        regime_consistency     : % of time in mean-reverting regime
        correlation_stability  : Lower correlation std → higher score

    Parameters
    ----------
    weights : dict, optional
        Custom weights for scoring dimensions.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "coint_p": 0.20,
        "coint_persistence": 0.15,
        "spread_sharpe": 0.15,
        "half_life_score": 0.07,
        "max_drawdown": 0.10,
        "hurst": 0.05,
        "oos_robustness": 0.10,
        "regime_consistency": 0.10,
        "correlation_stability": 0.05,
        "structural_break_frequency": 0.03,
    }

    def __init__(
        self, weights: Optional[dict[str, float]] = None,
        multiple_comparison: str = "bh",
    ) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.multiple_comparison = multiple_comparison

    def rank(
        self,
        pair_results: list[Any],
        top_n: int = 10,
        multiple_comparison: Optional[str] = None,
    ) -> list[RankedPair]:
        """Score and rank all pairs, returning top N."""
        if not pair_results:
            return []

        mc = multiple_comparison or self.multiple_comparison

        p_values: list[float] = []
        for pair in pair_results:
            wf = getattr(pair, "walk_forward", None) or {}
            coint = getattr(pair, "cointegration", None) or {}
            p = (
                self._safe_get(wf, "avg_oos_p_value", None)
                or self._safe_get(coint, "p_value", 1.0)
            )
            p_values.append(p)

        if mc == "bonferroni":
            adjusted_p = bonferroni_correct(p_values, len(p_values))
        elif mc == "bh":
            adjusted_p = benjamini_hochberg_correct(p_values)
        else:
            adjusted_p = p_values

        scored: list[RankedPair] = []
        for pair, raw_p, adj_p in zip(pair_results, p_values, adjusted_p):
            rp = self._score_pair(pair, adjusted_p_value=adj_p)
            rp.raw_p_value = float(raw_p)
            rp.adjusted_p_value = float(adj_p)
            scored.append(rp)

        scored.sort(key=lambda x: x.composite_score, reverse=True)
        return scored[:top_n]

    def _score_pair(self, pair: Any, adjusted_p_value: Optional[float] = None) -> RankedPair:
        components: dict[str, float] = {}

        coint = getattr(pair, "cointegration", None) or {}
        spread = getattr(pair, "spread", None) or {}
        wf = getattr(pair, "walk_forward", None) or {}
        bt = getattr(pair, "in_sample_backtest", getattr(pair, "backtest", None)) or {}
        corr = getattr(pair, "correlation", None) or {}
        regime = getattr(pair, "regime", None) or {}

        if adjusted_p_value is not None:
            coint_p = adjusted_p_value
        else:
            coint_p = self._safe_get(wf, "avg_oos_p_value", self._safe_get(coint, "p_value", 1.0))
        components["coint_p"] = self._inverse_score(coint_p)

        coint_persistence = self._safe_get(wf, "cointegration_percentage", 0.0)
        components["coint_persistence"] = coint_persistence / 100.0

        spread_sharpe = self._safe_get(wf, "avg_spread_sharpe", 0.0)
        components["spread_sharpe"] = self._direct_score(max(spread_sharpe, 0.0))

        half_life = self._safe_get(spread, "half_life", float("inf"))
        components["half_life_score"] = self._half_life_score(half_life)

        dd = abs(self._safe_get(bt, "max_drawdown_pct", 100.0))
        components["max_drawdown"] = self._inverse_score(dd / 100.0)

        hurst = self._safe_get(spread, "hurst_exponent", 0.5)
        components["hurst"] = max(0.0, 1.0 - hurst)

        oos_robustness = self._safe_get(wf, "oos_stationarity_pct", 0.0)
        components["oos_robustness"] = oos_robustness / 100.0

        regime_consistency = 0.0
        if regime:
            summary = getattr(regime, "regime_summary", None) or {}
            if isinstance(summary, dict):
                regime_consistency = summary.get("mean_reverting", 0.0)
        components["regime_consistency"] = regime_consistency

        num_breaks = self._safe_get(regime, "num_structural_breaks", 0)
        components["structural_break_frequency"] = max(0.0, 1.0 - num_breaks / 3.0)

        corr_stability_score = 1.0
        if corr:
            if hasattr(corr, "values"):
                corr_results_list = list(corr.values()) if isinstance(corr, dict) else []
                scores = [
                    getattr(r, "stability_score", 1.0)
                    for r in corr_results_list if hasattr(r, "stability_score")
                ]
                if scores:
                    corr_stability_score = float(np.mean(scores))
            elif isinstance(corr, dict):
                scores = [
                    v.get("stability_score", 1.0) if isinstance(v, dict) else getattr(v, "stability_score", 1.0)
                    for v in corr.values()
                ]
                if scores:
                    corr_stability_score = float(np.mean(scores))
        components["correlation_stability"] = min(1.0, max(0.0, corr_stability_score))

        composite = sum(
            components.get(k, 0.0) * w for k, w in self.weights.items()
        )

        confidence = self._classify_confidence(composite)
        risk = self._classify_risk(components)

        ticker_a = self._safe_get(pair, "ticker_a", "?")
        ticker_b = self._safe_get(pair, "ticker_b", "?")

        metrics = {
            "coint_p": coint_p,
            "half_life": half_life,
            "hurst": hurst,
            "spread_sharpe": spread_sharpe,
            "coint_persistence": coint_persistence,
            "oos_robustness": oos_robustness,
        }

        return RankedPair(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            composite_score=composite,
            confidence=confidence,
            risk=risk,
            score_components=components,
            metrics=metrics,
        )

    @staticmethod
    def _inverse_score(x: float, eps: float = 1e-10) -> float:
        """Higher when x is lower. 1 / (1 + x)."""
        if not np.isfinite(x):
            return 0.0
        return 1.0 / (1.0 + max(x, 0.0))

    @staticmethod
    def _direct_score(x: float) -> float:
        """Higher when x is higher, sigmoid-like bounded to [0, 1]."""
        if not np.isfinite(x):
            return 0.0
        return 1.0 / (1.0 + math.exp(-x / 2.0))

    @staticmethod
    def _half_life_score(hl: float) -> float:
        """Score half-life: ideal range 5–60 days.

        Too fast (< 5d) → noise, not mean reversion.
        Too slow (> 60d) → too slow to trade.
        """
        if not np.isfinite(hl) or hl <= 0:
            return 0.0
        if 5 <= hl <= 60:
            return 1.0
        if hl < 5:
            return max(0.0, hl / 5.0)
        return max(0.0, 1.0 - (hl - 60) / 200.0)

    @staticmethod
    def _classify_confidence(score: float) -> str:
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _classify_risk(components: dict[str, float]) -> str:
        hurst = components.get("hurst", 0.5)
        dd = components.get("max_drawdown", 0.5)
        badness = (1.0 - hurst) * 0.5 + (1.0 - dd) * 0.5
        if badness >= 0.6:
            return "high"
        if badness >= 0.3:
            return "medium"
        return "low"

    @staticmethod
    def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from object or key from dict."""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default
