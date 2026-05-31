"""Fast pairs cointegration engine — C++ backend with pure-Python fallback."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("cpp_engine")

_HAS_CPP = False
_cpp_module = None

try:
    import pairs_engine as _cpp_module
    _HAS_CPP = True
    logger.info("Loaded native pairs_engine (C++ pybind11)")
except ImportError:
    logger.info("pairs_engine not available; using pure-Python fallback")


def test_coint_pairs(
    prices: pd.DataFrame,
    significance: float = 0.05,
) -> list[dict[str, Any]]:
    """Test all pairs in a price DataFrame for cointegration.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns are tickers, index is datetime, values are close prices.
    significance : float
        p-value threshold.

    Returns
    -------
    list of dict
        Each dict has keys: ticker_a, ticker_b, p_value, hedge_ratio, is_cointegrated.
    """
    tickers = prices.columns.tolist()
    n = len(tickers)

    if n < 2:
        return []

    if _HAS_CPP:
        return _test_coint_cpp(prices, tickers, significance)
    else:
        return _test_coint_py(prices, tickers, significance)


def _test_coint_cpp(
    prices: pd.DataFrame,
    tickers: list[str],
    significance: float,
) -> list[dict[str, Any]]:
    """C++ implementation."""
    from .stats_arb.hedge_ratio import estimate_ols

    arr = prices.values.astype(np.float64).T  # shape (n_tickers, n_days)
    arr = np.nan_to_num(arr, nan=0.0)

    result = _cpp_module.test_coint(arr, significance)
    pairs: list[dict[str, Any]] = []

    for i in range(result.shape[0]):
        a_idx = int(result[i, 0])
        b_idx = int(result[i, 1])
        p_val = float(result[i, 2])
        hr = float(result[i, 3])
        coint = bool(result[i, 4])

        pairs.append({
            "ticker_a": tickers[a_idx],
            "ticker_b": tickers[b_idx],
            "p_value": p_val,
            "hedge_ratio": hr,
            "is_cointegrated": coint,
        })

    return pairs


def _test_coint_py(
    prices: pd.DataFrame,
    tickers: list[str],
    significance: float,
) -> list[dict[str, Any]]:
    """Pure-Python sequential implementation (fallback)."""
    from .stats_arb.cointegration import CointegrationTester

    pairs: list[dict[str, Any]] = []
    n = len(tickers)
    total = n * (n - 1) // 2

    for idx, i in enumerate(range(n)):
        for j in range(i + 1, n):
            a, b = tickers[i], tickers[j]
            try:
                pair_prices = prices[[a, b]].ffill().dropna(how="any")
                if len(pair_prices) < 50:
                    continue
                ct = CointegrationTester(pair_prices, significance)
                res = ct.run()
                pairs.append({
                    "ticker_a": a,
                    "ticker_b": b,
                    "p_value": res.p_value,
                    "hedge_ratio": res.hedge_ratio,
                    "is_cointegrated": res.is_cointegrated,
                })
            except Exception:
                continue

            if (idx * n + j) % 500 == 0:
                logger.info(f"  EG test progress: {idx * n + j}/{total}")

    return pairs
