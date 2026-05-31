"""
Phase 2 — Relationship Discovery: Cointegration.

Engle-Granger and Johansen cointegration tests for identifying
mean-reverting spread relationships between asset pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from backend.cpp_ext import eg_coint_test

from .config import DEFAULT_SIGNIFICANCE
from .hedge_ratio import estimate_ols

logger = logging.getLogger("stats_arb.cointegration")


@dataclass
class CointegrationResult:
    """Engle-Granger cointegration test result.

    Attributes
    ----------
    p_value : float
        p-value of the test (lower → stronger cointegration).
    test_statistic : float
        ADF test statistic on the spread.
    critical_values : dict[str, float]
        Critical values at 1%, 5%, 10% significance.
    hedge_ratio : float
        OLS beta from A = beta * B + intercept.
    is_cointegrated : bool
        True if p_value < significance threshold.
    spread : pd.Series
        The residual series (A - beta * B).
    method : str
        Test method identifier.
    """

    p_value: float
    test_statistic: float
    critical_values: dict[str, float]
    hedge_ratio: float
    is_cointegrated: bool
    spread: pd.Series
    method: str = "engle-granger"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "p_value": round(self.p_value, 6),
            "test_statistic": round(self.test_statistic, 4),
            "critical_values": {
                k: round(v, 4) for k, v in self.critical_values.items()
            },
            "hedge_ratio": round(self.hedge_ratio, 6),
            "is_cointegrated": self.is_cointegrated,
        }


@dataclass
class JohansenResult:
    """Johansen cointegration test result.

    Attributes
    ----------
    trace_statistic : float
        Trace test statistic.
    trace_critical_values : dict[str, float]
        Critical values for the trace test.
    eigenvalue_statistic : float
        Max eigenvalue test statistic.
    eigenvalue_critical_values : dict[str, float]
        Critical values for the eigenvalue test.
    is_cointegrated : bool
        True if trace_stat > 95% critical value (r >= 1).
    r : int
        Number of cointegrating relationships detected.
    eigenvectors : np.ndarray
        Estimated eigenvectors (cointegrating vectors).
    """

    trace_statistic: float
    trace_critical_values: dict[str, float]
    eigenvalue_statistic: float
    eigenvalue_critical_values: dict[str, float]
    is_cointegrated: bool
    r: int
    eigenvectors: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "johansen",
            "trace_statistic": round(self.trace_statistic, 4),
            "trace_critical_values": {
                k: round(v, 4) for k, v in self.trace_critical_values.items()
            },
            "eigenvalue_statistic": round(self.eigenvalue_statistic, 4),
            "eigenvalue_critical_values": {
                k: round(v, 4) for k, v in self.eigenvalue_critical_values.items()
            },
            "is_cointegrated": self.is_cointegrated,
        }


class CointegrationTester:
    """Engle-Granger cointegration test with OLS hedge ratio estimation.

    The EG test checks whether the spread series A - βB is stationary (I(0)).
    If so, the two series are cointegrated — deviations from equilibrium
    are mean-reverting and potentially tradeable.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with columns [ticker_a, ticker_b].
    significance : float
        p-value threshold for cointegration.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        significance: float = DEFAULT_SIGNIFICANCE,
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.significance = significance

    def run(self) -> CointegrationResult:
        a = self.prices[self.cols[0]].values
        b = self.prices[self.cols[1]].values

        hedge_ratio = estimate_ols(a, b)
        spread = pd.Series(a - hedge_ratio * b, index=self.prices.index)

        test_stat, p_value, _ = eg_coint_test(a, b, maxlag=1, autolag=True)
        crit_values = {"1%": 0.0, "5%": 0.0, "10%": 0.0}

        return CointegrationResult(
            p_value=float(p_value),
            test_statistic=float(test_stat),
            critical_values=crit_values,
            hedge_ratio=hedge_ratio,
            is_cointegrated=bool(p_value < self.significance),
            spread=spread,
            method="engle-granger",
        )

    @staticmethod
    def test_on_window(
        a: np.ndarray, b: np.ndarray, significance: float = DEFAULT_SIGNIFICANCE
    ) -> tuple[float, float, bool]:
        """Quick cointegration test on raw arrays (for walk-forward use) — C++ accelerated."""
        hr = estimate_ols(a, b)
        try:
            _, p_val, _ = eg_coint_test(a, b, maxlag=1, autolag=True)
        except Exception:
            p_val = 1.0
        return hr, float(p_val), bool(p_val < significance)


class JohansenTester:
    """Johansen cointegration test (trace and max eigenvalue).

    Unlike EG (which assumes a single cointegrating vector), Johansen
    can detect multiple cointegrating relationships. For pairs, r=1
    indicates a single cointegrating relationship (the spread).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with columns [ticker_a, ticker_b].
    significance : float
        Significance level for critical value comparison.
    det_order : int
        Deterministic term order (-1 = none, 0 = constant, 1 = trend).
    k_ar_diff : int
        Number of lags in the VAR (default 1).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        significance: float = DEFAULT_SIGNIFICANCE,
        det_order: int = 0,
        k_ar_diff: int = 1,
    ) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()
        self.significance = significance
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff

    def run(self) -> Optional[JohansenResult]:
        """Execute Johansen test and return result or None if it fails."""
        data = self.prices[self.cols].values
        try:
            result = coint_johansen(data, self.det_order, self.k_ar_diff)
        except Exception as e:
            logger.warning(f"Johansen test failed: {e}")
            return None

        trace_stat = float(result.lr1[0])
        eig_stat = float(result.lr2[0])

        trace_cv = {
            "90%": float(result.cvt[0, 0]),
            "95%": float(result.cvt[0, 1]),
            "99%": float(result.cvt[0, 2]),
        }
        eig_cv = {
            "90%": float(result.cvm[0, 0]),
            "95%": float(result.cvm[0, 1]),
            "99%": float(result.cvm[0, 2]),
        }

        is_cointegrated = trace_stat > trace_cv["95%"]

        return JohansenResult(
            trace_statistic=trace_stat,
            trace_critical_values=trace_cv,
            eigenvalue_statistic=eig_stat,
            eigenvalue_critical_values=eig_cv,
            is_cointegrated=is_cointegrated,
            r=1 if is_cointegrated else 0,
            eigenvectors=result.evec,
        )
