"""
Phase 2 — Hedge Ratio Estimation.

Multiple methods for estimating the hedge ratio beta in:
    spread = price_A - beta * price_B

Supports OLS, rolling OLS, and Kalman filter / RLS dynamic estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("stats_arb.hedge_ratio")


def estimate_ols(
    a: np.ndarray,
    b: np.ndarray,
    add_const: bool = True,
    robust: bool = False,
) -> float:
    """Single source of truth for OLS hedge ratio estimation.

    Parameters
    ----------
    a : np.ndarray
        Dependent variable (price A).
    b : np.ndarray
        Independent variable (price B).
    add_const : bool
        Whether to include an intercept term.
    robust : bool
        If True, use HuberRegressor instead of OLS.

    Returns
    -------
    float
        The hedge ratio (slope coefficient).
    """
    if robust:
        from sklearn.linear_model import HuberRegressor
        b_reshaped = b.reshape(-1, 1)
        if add_const:
            model = HuberRegressor(fit_intercept=True)
        else:
            model = HuberRegressor(fit_intercept=False)
        model.fit(b_reshaped, a)
        return float(model.coef_[0])
    if add_const:
        b_with_const = np.column_stack([np.ones_like(b), b])
        beta, _, _, _ = np.linalg.lstsq(b_with_const, a, rcond=None)
        return float(beta[1])
    beta, _, _, _ = np.linalg.lstsq(b.reshape(-1, 1), a, rcond=None)
    return float(beta[0])


@dataclass
class HedgeRatioResult:
    """Hedge ratio estimation result.

    Attributes
    ----------
    beta : float or np.ndarray
        Static beta or time series of betas (for rolling/Kalman).
    intercept : float
        Regression intercept.
    beta_series : pd.Series, optional
        Time series of betas if dynamic method used.
    method : str
        Estimation method name.
    """

    beta: float | np.ndarray
    intercept: float
    beta_series: Optional[pd.Series] = None
    method: str = "ols"

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "beta": round(float(self.beta), 6) if isinstance(self.beta, (int, float)) else float(self.beta[-1]),
            "intercept": round(self.intercept, 6),
        }


class HedgeRatioEstimator:
    """Estimates hedge ratio between two price series using multiple methods.

    Methods
    -------
    ols :
        Static OLS regression over the full sample.
    rolling_ols :
        Rolling-window OLS producing a time series of betas.
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        self.prices = prices
        self.cols = prices.columns.tolist()

    def ols(self) -> HedgeRatioResult:
        a = self.prices[self.cols[0]].values
        b = self.prices[self.cols[1]].values
        b_with_const = np.column_stack([np.ones_like(b), b])
        beta, residuals, _, _ = np.linalg.lstsq(b_with_const, a, rcond=None)
        return HedgeRatioResult(
            beta=float(beta[1]),
            intercept=float(beta[0]),
            method="ols",
        )

    def rolling_ols(self, window: int = 63) -> HedgeRatioResult:
        a = self.prices[self.cols[0]]
        b = self.prices[self.cols[1]]
        betas: list[float] = []
        intercepts: list[float] = []

        for i in range(window, len(self.prices) + 1):
            chunk_a = a.iloc[i - window : i].values
            chunk_b = b.iloc[i - window : i].values
            b_with_const = np.column_stack([np.ones_like(chunk_b), chunk_b])
            beta, _, _, _ = np.linalg.lstsq(b_with_const, chunk_a, rcond=None)
            intercepts.append(float(beta[0]))
            betas.append(float(beta[1]))

        beta_series = pd.Series(
            betas, index=self.prices.index[window - 1 :]
        )
        return HedgeRatioResult(
            beta=np.array(betas),
            intercept=float(np.mean(intercepts)),
            beta_series=beta_series,
            method="rolling_ols",
        )


class KalmanFilterHedge:
    """Dynamic hedge ratio via Recursive Least Squares (RLS) with forgetting factor.

    RLS is numerically more stable than a plain Kalman filter for hedge ratio
    estimation. The forgetting factor λ ∈ (0, 1] controls how quickly past
    observations are discounted — lower λ = more adaptive to regime changes.

    A standard choice is λ = 0.99 for slowly varying relationships.

    Parameters
    ----------
    lambda_ : float
        Forgetting factor (default 0.99).
    delta : float
        Regularisation term to prevent covariance collapse.
    """

    def __init__(self, lambda_: float = 0.99, delta: float = 1e-4) -> None:
        self.lambda_ = lambda_
        self.delta = delta

    def fit(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Estimate time-varying hedge ratio (beta).

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (price A).
        x : np.ndarray
            Independent variable (price B).

        Returns
        -------
        np.ndarray
            Time series of hedge ratio estimates.
        """
        n = len(y)
        if n < 10:
            return np.full(n, float("nan"))

        y = y.astype(np.float64)
        x = x.astype(np.float64)

        theta = np.array([0.0, 0.0])  # [intercept, slope]
        P = np.eye(2) * 100.0
        betas = np.zeros(n)

        for t in range(n):
            phi = np.array([1.0, x[t]])
            y_pred = theta @ phi
            innovation = y[t] - y_pred

            g = P @ phi / (self.lambda_ + phi @ P @ phi)
            theta = theta + g * innovation
            P = (P - np.outer(g, phi @ P)) / self.lambda_
            P += np.eye(2) * self.delta

            betas[t] = float(theta[1])

        return betas

    def fit_dataframe(
        self, prices: pd.DataFrame
    ) -> HedgeRatioResult:
        a = prices[prices.columns[0]].values
        b = prices[prices.columns[1]].values
        betas = self.fit(a, b)
        return HedgeRatioResult(
            beta=betas,
            intercept=0.0,
            beta_series=pd.Series(betas, index=prices.index),
            method="kalman_rls",
        )
