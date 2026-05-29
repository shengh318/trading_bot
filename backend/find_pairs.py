"""
Automated stock pairs discovery script — 3-phase pipeline.

Phase 1: Bulk download -> pairwise correlation filter
Phase 2: EG cointegration on correlated pairs only
Phase 3: Full pipeline on top candidates -> ranked report

Usage:
    # S&P 500 (default) — saves report to pairs_report_<timestamp>.md
    python -m backend.find_pairs

    # NASDAQ-100
    python -m backend.find_pairs nasdaq100

    # Custom universe
    python -m backend.find_pairs NVDA,AMD,KO,PEP,JPM,GS,MSFT

    # Strict filters
    python -m backend.find_pairs sp500 --min-sharpe 1.5 --min-corr 0.7 --output best_pairs.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

import yfinance as yf

from .stats_arb.cointegration import CointegrationTester
from .stats_arb.pipeline import PairAnalyzer
from .stats_arb.ranking import PairRanker
from .stats_arb.config import (
    AUTO_DISCOVER_TOP_CANDIDATES,
    AUTO_DISCOVER_MIN_SHARPE,
    AUTO_DISCOVER_MIN_RETURN_PCT,
    AUTO_DISCOVER_MAX_DRAWDOWN_PCT,
    DEFAULT_SIGNIFICANCE,
    MULTIPLE_COMPARISON_METHOD,
)
from .stats_arb.discover import resolve_universe, _apply_filters, _format_results, _print_results, _save_markdown

logger = logging.getLogger("find_pairs")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


_UNIVERSE_PRESETS = {"sp500", "nasdaq100", "dow30"}


def _compute_correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlation on daily returns."""
    returns = prices.pct_change().dropna()
    return returns.corr()


def _get_highly_correlated_pairs(
    tickers: list[str],
    corr_matrix: pd.DataFrame,
    min_correlation: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Return pairs with correlation >= min_correlation and < 0.99.
    
    Pairs with correlation >= 0.99 are near-identical (e.g. SPY/VOO) and
    cause collinearity warnings + LAPACK errors in cointegration tests.
    """
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            corr = corr_matrix.loc[a, b]
            if min_correlation <= corr < 0.99 and np.isfinite(corr):
                pairs.append((a, b, corr))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _run_coint_silent(prices: pd.DataFrame, significance: float):
    """Run CointegrationTester with warning suppression."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ct = CointegrationTester(prices, significance)
        return ct.run()


def _test_coint(args: tuple) -> tuple[str, str, float, float]:
    """Quick EG cointegration test on a single pair using cached parquet data."""
    a, b, data_path, significance = args
    try:
        prices = pd.read_parquet(data_path, columns=[a, b])
        if len(prices) < 50:
            return (a, b, 1.0, 0.0)
        corr_val = prices[a].corr(prices[b])
        if corr_val >= 0.99 or not np.isfinite(corr_val):
            return (a, b, 1.0, 0.0)
        # Skip pairs with near-constant price ratio (triggers LAPACK errors)
        ratio = prices[a] / prices[b]
        ratio_cv = ratio.std() / ratio.mean()
        if not np.isfinite(ratio_cv) or ratio_cv < 0.001:
            return (a, b, 1.0, 0.0)
        res = _run_coint_silent(prices, significance)
        if res.is_cointegrated:
            return (a, b, res.p_value, -np.log10(max(res.p_value, 1e-15)))
        return (a, b, 1.0, 0.0)
    except Exception as e:
        logger.debug(f"EG test failed {a}/{b}: {e}")
        return (a, b, 1.0, 0.0)


def _analyze_full(args: tuple):
    """Full pipeline for a single pair."""
    a, b, start, end, significance, capital = args
    try:
        analyzer = PairAnalyzer(
            a, b, start, end,
            significance=significance,
            capital=capital,
        )
        return analyzer.analyze()
    except Exception as e:
        logger.error(f"Full analysis failed {a}/{b}: {e}")
        return None


def find_pairs(
    universe: str | list[str] | None = None,
    universe_file: str | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    min_correlation: float = 0.5,
    top_candidates: int = AUTO_DISCOVER_TOP_CANDIDATES,
    min_sharpe: float = AUTO_DISCOVER_MIN_SHARPE,
    min_return_pct: float = AUTO_DISCOVER_MIN_RETURN_PCT,
    max_drawdown_pct: float = AUTO_DISCOVER_MAX_DRAWDOWN_PCT,
    require_mean_reverting: bool = False,
    require_no_breaks: bool = False,
    capital: float = 100_000.0,
    output_md: str | None = None,
    output_json: bool = False,
) -> list[dict]:
    """Run 3-phase pairs discovery.

    Phase 1: Bulk download -> pairwise correlation matrix -> keep correlated pairs
    Phase 2: EG cointegration on correlated pairs (parallel)
    Phase 3: Full pipeline on top N candidates -> rank -> filter -> report
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    tickers = resolve_universe(universe, universe_file)
    n = len(tickers)
    logger.info(f"Universe: {n} tickers")
    logger.info(f"Period: {start} -> {end}")

    # ── Phase 1: Batch download + correlation filter ──
    # Download in batches of 50 to avoid yfinance date-range truncation
    # that occurs when requesting 500 tickers in a single call.
    logger.info(f"Phase 1: Downloading {n} tickers in batches ...")
    batch_size = 50
    all_prices: pd.DataFrame | None = None
    failed_tickers: list[str] = []
    for batch_start in range(0, n, batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        try:
            raw = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                batch_prices = raw["Close"].copy()
            else:
                batch_prices = raw.copy()
        except Exception as e:
            logger.warning(f"Batch download failed for {batch}: {e}")
            failed_tickers.extend(batch)
            continue
        if all_prices is None:
            all_prices = batch_prices
        else:
            all_prices = all_prices.join(batch_prices, how="outer")

    if all_prices is None or all_prices.empty:
        logger.error("No data downloaded for any ticker")
        return []

    # Drop tickers with insufficient data
    min_rows = 252
    valid = [c for c in all_prices.columns if all_prices[c].notna().sum() >= min_rows]
    dropped = set(all_prices.columns) - set(valid) | set(failed_tickers)
    if dropped:
        logger.warning(f"Dropped {len(dropped)} tickers with insufficient data")
        all_prices = all_prices[valid]
    all_prices = all_prices.ffill()
    if len(all_prices.columns) < 2:
        logger.error("Fewer than 2 valid tickers remaining after filtering")
        return []
    tickers = valid
    n = len(tickers)
    logger.info(f"  Got {len(all_prices)} rows x {len(all_prices.columns)} columns ({n} valid tickers)")

    corr_matrix = _compute_correlation_matrix(all_prices)
    correlated_pairs = _get_highly_correlated_pairs(tickers, corr_matrix, min_correlation)
    logger.info(f"Phase 1: {len(correlated_pairs)} pairs with correlation >= {min_correlation} (from {n*(n-1)//2:,} total)")

    if not correlated_pairs:
        logger.warning(f"No pairs found with correlation >= {min_correlation}. Try lowering --min-corr.")
        return []

    # ── Phase 2: EG cointegration on filtered pairs ──
    logger.info(f"Phase 2: Testing {len(correlated_pairs)} pairs for cointegration ...")

    # Save prices to temp parquet so workers can read efficiently
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    all_prices.to_parquet(tmp_path)

    coint_args = [
        (a, b, tmp_path, significance)
        for a, b, _ in correlated_pairs
    ]

    import os
    try:
        cointegrated: list[tuple[str, str, float, float]] = []
        max_workers = min(8, len(coint_args))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_test_coint, args): args for args in coint_args}
            for future in as_completed(futures):
                try:
                    cointegrated.append(future.result())
                except Exception as e:
                    logger.error(f"EG worker failed: {e}")

        # Filter to only cointegrated, sort by -log10(p)
        cointegrated = [(a, b, p, s) for a, b, p, s in cointegrated if s > 0]
        cointegrated.sort(key=lambda x: x[3], reverse=True)

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    logger.info(f"Phase 2: Found {len(cointegrated)} cointegrated pairs (p < {significance})")

    if not cointegrated:
        logger.warning("No cointegrated pairs found.")
        return []

    # ── Phase 3: Full pipeline on top N candidates ──
    candidates = cointegrated[:top_candidates]
    logger.info(f"Phase 3: Running full pipeline on top {len(candidates)} candidates ...")

    pair_args = [
        (a, b, start, end, significance, capital)
        for a, b, _, _ in candidates
    ]

    all_results: list = []
    max_workers = min(8, len(pair_args))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_analyze_full, args): args for args in pair_args}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Full pipeline worker failed: {e}")

    # Fallback to sequential if parallel produced no results
    if not all_results:
        logger.warning("Parallel analysis produced no results; trying sequentially ...")
        for args in pair_args:
            result = _analyze_full(args)
            if result is not None:
                all_results.append(result)

    if not all_results:
        logger.warning("No full analyses succeeded.")
        return []

    # Rank and filter
    ranker = PairRanker(multiple_comparison=MULTIPLE_COMPARISON_METHOD)
    ranked = ranker.rank(all_results, top_n=len(all_results))
    filtered = _apply_filters(
        ranked, all_results, min_sharpe, min_return_pct,
        max_drawdown_pct, require_mean_reverting, require_no_breaks,
    )

    output = _format_results(filtered, all_results)
    _print_results(output)

    if output_json:
        import json
        print(json.dumps(output, indent=2))

    if output_md and output:
        _save_markdown(output, output_md, start, end)
        logger.info(f"Markdown report saved: {output_md}")

    return output


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Find good stock pairs using correlation + cointegration analysis (3-phase pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.find_pairs
  python -m backend.find_pairs nasdaq100
  python -m backend.find_pairs dow30
  python -m backend.find_pairs NVDA,AMD,KO,PEP,JPM,GS,MSFT
  python -m backend.find_pairs sp500 --min-sharpe 1.5 --min-corr 0.7 --output best_pairs.md
        """,
    )

    parser.add_argument(
        "universe", type=str, nargs="?",
        default="sp500",
        help='Universe: "sp500", "nasdaq100", "dow30", or comma-separated tickers (default: sp500)',
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output Markdown file (default: pairs_report_<timestamp>.md)",
    )
    parser.add_argument(
        "--min-corr", type=float, default=0.5,
        help="Minimum Pearson correlation for Phase 1 filter (default: 0.5)",
    )
    parser.add_argument(
        "--min-sharpe", type=float, default=0.0,
        help="Minimum Sharpe ratio (default: 0.0)",
    )
    parser.add_argument(
        "--min-return", type=float, default=-1000.0,
        help="Minimum total return %% (default: -1000.0)",
    )
    parser.add_argument(
        "--max-drawdown", type=float, default=-100.0,
        help="Maximum drawdown %% (default: -100.0)",
    )
    parser.add_argument(
        "--top-candidates", type=int, default=50,
        help="Number of top cointegrated pairs to fully analyze (default: 50)",
    )
    parser.add_argument(
        "--require-mean-reverting", action="store_true",
        help="Only keep pairs currently in mean-reverting regime",
    )
    parser.add_argument(
        "--require-no-breaks", action="store_true",
        help="Exclude pairs with structural breaks",
    )
    parser.add_argument(
        "--start", type=str, default="2015-01-01",
        help="Start date (default: 2015-01-01)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Initial capital for backtest (default: 100000)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also output JSON to stdout",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.getLogger("find_pairs").setLevel(getattr(logging, args.log_level.upper()))

    output_file = args.output or f"pairs_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    print(f"\n{'='*60}")
    print(f"Stock Pairs Discovery — 3-Phase Pipeline")
    print(f"{'='*60}")
    print(f"Universe:     {args.universe}")
    print(f"Period:       {args.start} -> today")
    print(f"Min corr:     {args.min_corr}")
    print(f"Filters:      Sharpe >= {args.min_sharpe}, Return >= {args.min_return}%, DD >= {args.max_drawdown}%")
    print(f"Top N:        {args.top_candidates}")
    print(f"Output:       {output_file}")
    print(f"{'='*60}\n")

    find_pairs(
        universe=args.universe,
        start=args.start,
        significance=DEFAULT_SIGNIFICANCE,
        min_correlation=args.min_corr,
        top_candidates=args.top_candidates,
        min_sharpe=args.min_sharpe,
        min_return_pct=args.min_return,
        max_drawdown_pct=args.max_drawdown,
        require_mean_reverting=args.require_mean_reverting,
        require_no_breaks=args.require_no_breaks,
        capital=args.capital,
        output_md=output_file,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
