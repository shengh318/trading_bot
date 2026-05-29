"""
Phase 10 — CLI Entry Point.

Configurable command-line interface for the statistical arbitrage framework.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import (
    AUTO_DISCOVER_MAX_DRAWDOWN_PCT,
    AUTO_DISCOVER_MIN_RETURN_PCT,
    AUTO_DISCOVER_MIN_SHARPE,
    AUTO_DISCOVER_TOP_CANDIDATES,
    DEFAULT_SIGNIFICANCE,
    DEFAULT_Z_ENTRY,
    DEFAULT_WALK_FORWARD_TRAIN,
    DEFAULT_WALK_FORWARD_TEST,
)
from .data import DataManager
from .cointegration import CointegrationTester
from .discover import auto_discover
from .pipeline import PairAnalyzer, PairAnalysisResult
from .ranking import PairRanker
from .visualization import Visualizer

logger = logging.getLogger("stats_arb")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


def analyze_pair_cli(
    ticker_a: str,
    ticker_b: str,
    start: str = "2015-01-01",
    end: str | None = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    use_purged_wfv: bool = True,
    run_johansen: bool = True,
    run_kalman: bool = False,
    run_ml: bool = False,
    save_plots: bool = True,
    output_json: bool = False,
    capital: float = 100_000.0,
) -> PairAnalysisResult:
    """Run full pipeline for a single pair and log results."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    logger.info(f"{'='*60}")
    logger.info(f"Analyzing pair: {ticker_a} / {ticker_b}")
    logger.info(f"Period: {start} → {end}")
    logger.info(f"{'='*60}")

    analyzer = PairAnalyzer(
        ticker_a, ticker_b, start, end,
        significance=significance,
        use_purged_walk_forward=use_purged_wfv,
        run_johansen=run_johansen,
        run_kalman=run_kalman,
        run_ml=run_ml,
        capital=capital,
    )
    result = analyzer.analyze()

    logger.info(f"  Cointegration: p={result.cointegration.p_value:.6f}, "
                f"hr={result.cointegration.hedge_ratio:.4f}, "
                f"ok={result.cointegration.is_cointegrated}")
    logger.info(f"  Spread: hl={result.spread.half_life:.1f}d, "
                f"H={result.spread.hurst_exponent:.3f}, "
                f"z={result.spread.current_zscore:.2f}, "
                f"stationary={result.spread.is_stationary}")
    logger.info(f"  Walk-Forward: avg_oos_p={result.walk_forward.avg_oos_p_value:.6f}, "
                f"coint%={result.walk_forward.cointegration_percentage:.1f}%")
    bt = getattr(result, "in_sample_backtest", None) or getattr(result, "backtest", None)
    if bt:
        logger.info(f"  In-Sample Backtest: ret={bt.total_return_pct:.1f}%, "
                    f"sharpe={bt.sharpe_ratio:.3f}, "
                    f"sortino={bt.sortino_ratio:.3f}, "
                    f"dd={bt.max_drawdown_pct:.1f}%, "
                    f"trades={bt.num_trades}, "
                    f"win%={bt.win_rate_pct:.1f}%")
    logger.info(f"  Regime: {result.regime.current_regime.value}, "
                f"tradeable={result.regime.trading_allowed}, "
                f"VIX={result.regime.vix_level:.1f}")

    if save_plots:
        dm = DataManager([ticker_a, ticker_b], start, end)
        prices = dm.fetch()
        vis = Visualizer(save_dir="stats_arb_plots")
        paths = vis.plot_pair_analysis(result, prices)
        logger.info(f"Plots saved ({len(paths)} files)")

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))

    return result


def analyze_multiple_cli(
    pairs: list[tuple[str, str]],
    start: str = "2015-01-01",
    end: str | None = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    top_n: int = 5,
    parallel: bool = True,
    use_purged_wfv: bool = True,
    run_ml: bool = False,
    save_plots: bool = False,
    capital: float = 100_000.0,
) -> list[PairAnalysisResult]:
    """Analyze and rank multiple pairs."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    all_results: list[PairAnalysisResult] = []

    if parallel and len(pairs) > 2:
        logger.info(f"Analyzing {len(pairs)} pairs in parallel...")
        pair_args = [
            (a, b, start, end, significance, use_purged_wfv, run_ml, capital)
            for a, b in pairs
        ]

        with ProcessPoolExecutor(max_workers=min(8, len(pairs))) as pool:
            futures = {
                pool.submit(_analyze_pair_static, args): args
                for args in pair_args
            }
            for future in as_completed(futures):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    logger.error(f"Error: {e}")
    else:
        for a, b in pairs:
            try:
                result = analyze_pair_cli(
                    a, b, start, end, significance,
                    use_purged_wfv=use_purged_wfv,
                    run_ml=run_ml,
                    save_plots=save_plots,
                    capital=capital,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed {a}/{b}: {e}")

    from .config import MULTIPLE_COMPARISON_METHOD
    ranker = PairRanker(multiple_comparison=MULTIPLE_COMPARISON_METHOD)
    ranked = ranker.rank(all_results, top_n=top_n)

    logger.info(f"\n{'='*60}")
    logger.info(f"Top {top_n} Ranked Pairs")
    logger.info(f"{'='*60}")
    for i, p in enumerate(ranked):
        logger.info(
            f"{i+1}. {p.ticker_a}/{p.ticker_b}: "
            f"score={p.composite_score:.4f}, "
            f"confidence={p.confidence}, "
            f"risk={p.risk}, "
            f"raw_p={p.raw_p_value:.6f}, "
            f"adj_p={p.adjusted_p_value:.6f}"
        )

    if save_plots:
        Visualizer.plot_ranking_summary(
            ranked, save_path="stats_arb_plots/ranking_summary.png"
        )

    return all_results


def _analyze_pair_static(args: tuple) -> PairAnalysisResult:
    """Helper for parallel execution."""
    a, b, start, end, sig, purged, run_ml, capital = args
    logger.info(f"Analyzing {a}/{b}...")
    analyzer = PairAnalyzer(
        a, b, start, end, significance=sig,
        use_purged_walk_forward=purged,
        run_ml=run_ml,
        capital=capital,
    )
    return analyzer.analyze()


def build_heatmap(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    parallel: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Build pairwise cointegration heatmap matrix."""
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    pair_list = list(combinations(tickers, 2))
    results: list[tuple[str, str, float]] = []

    logger.info(f"Building heatmap for {len(tickers)} tickers ({len(pair_list)} pairs)...")

    def _cell(pair: tuple[str, str]) -> tuple[str, str, float]:
        try:
            dm = DataManager(list(pair), start, end)
            prices = dm.fetch()
            ct = CointegrationTester(prices, significance)
            res = ct.run()
            return (pair[0], pair[1], -np.log10(max(res.p_value, 1e-15)))
        except Exception as e:
            logger.warning(f"Failed {pair[0]}/{pair[1]}: {e}")
            return (pair[0], pair[1], 0.0)

    if parallel and len(pair_list) > 2:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_cell, p): p for p in pair_list}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
    else:
        for pair in pair_list:
            results.append(_cell(pair))

    matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    matrix.values[:] = 0.0
    for a, b, val in results:
        matrix.loc[a, b] = val
        matrix.loc[b, a] = val

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair analysis
  python -m stats_arb.cli --pair NVDA,META

  # With ML models
  python -m stats_arb.cli --pair KO,PEP --ml --start 2010-01-01

  # Multiple pairs with ranking
  python -m stats_arb.cli --pairs-file pairs.json --rank --top-n 10

  # Cointegration heatmap
  python -m stats_arb.cli --heatmap NVDA,AMD,INTC,AAPL,MSFT,GOOGL

  # Purged walk-forward
  python -m stats_arb.cli --pair XOM,CVX --purged

  # Output JSON
  python -m stats_arb.cli --pair JPM,GS --json
        """,
    )

    parser.add_argument("--pair", type=str, help="Comma-separated pair (e.g., NVDA,META)")
    parser.add_argument("--pairs-file", type=str, help="JSON file with list of [a, b] pairs")
    parser.add_argument("--auto-discover", type=str, nargs="?", const="sp500",
                        help="Auto-discover profitable pairs from a universe: sp500, nasdaq100, dow30, "
                             "or comma-separated tickers (default: sp500)")
    parser.add_argument("--universe-file", type=str,
                        help="File with one ticker per line for auto-discover")
    parser.add_argument("--top-candidates", type=int, default=AUTO_DISCOVER_TOP_CANDIDATES,
                        help=f"Number of top cointegrated pairs to fully analyze (default: {AUTO_DISCOVER_TOP_CANDIDATES})")
    parser.add_argument("--min-sharpe", type=float, default=AUTO_DISCOVER_MIN_SHARPE,
                        help=f"Minimum Sharpe ratio filter (default: {AUTO_DISCOVER_MIN_SHARPE})")
    parser.add_argument("--min-return-pct", type=float, default=AUTO_DISCOVER_MIN_RETURN_PCT,
                        help=f"Minimum total return %% filter (default: {AUTO_DISCOVER_MIN_RETURN_PCT})")
    parser.add_argument("--max-drawdown-pct", type=float, default=AUTO_DISCOVER_MAX_DRAWDOWN_PCT,
                        help=f"Maximum drawdown %% filter (default: {AUTO_DISCOVER_MAX_DRAWDOWN_PCT})")
    parser.add_argument("--require-mean-reverting", action="store_true",
                        help="Only keep pairs in mean-reverting regime")
    parser.add_argument("--require-no-breaks", action="store_true",
                        help="Exclude pairs with structural breaks")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Save Markdown report to file")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date (default=today)")
    parser.add_argument("--significance", type=float, default=DEFAULT_SIGNIFICANCE,
                        help="Cointegration significance threshold")
    parser.add_argument("--rank", action="store_true", help="Rank multiple pairs")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top pairs")
    parser.add_argument("--heatmap", type=str, help="Comma-separated tickers for heatmap")
    parser.add_argument("--no-plots", action="store_true", help="Skip plots")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel")
    parser.add_argument("--ml", action="store_true", help="Run ML models")
    parser.add_argument("--no-purged", action="store_true",
                        help="Use simple walk-forward (not purged)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Initial capital for backtest")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.getLogger("stats_arb").setLevel(getattr(logging, args.log_level.upper()))

    if not args.pair and not args.pairs_file and not args.heatmap and not args.auto_discover:
        parser.print_help()
        sys.exit(1)

    if args.heatmap:
        tickers = [t.strip().upper() for t in args.heatmap.split(",")]
        matrix = build_heatmap(
            tickers, args.start, args.end, args.significance,
            parallel=not args.no_parallel,
        )
        path = Visualizer.plot_cointegration_heatmap(matrix)
        logger.info(f"Heatmap saved: {path}")
        print("\nCointegration Score Matrix (-log10 p-value):")
        print(matrix.round(2).to_string())
        return

    if args.pairs_file and args.rank:
        with open(args.pairs_file) as f:
            pair_data = json.load(f)
        pairs = [(p[0], p[1]) for p in pair_data]
        results = analyze_multiple_cli(
            pairs, args.start, args.end, args.significance,
            top_n=args.top_n,
            parallel=not args.no_parallel,
            use_purged_wfv=not args.no_purged,
            run_ml=args.ml,
            save_plots=not args.no_plots,
            capital=args.capital,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        return

    if args.auto_discover:
        auto_discover(
            universe=args.auto_discover,
            universe_file=args.universe_file,
            start=args.start,
            end=args.end,
            significance=args.significance,
            top_candidates=args.top_candidates,
            min_sharpe=args.min_sharpe,
            min_return_pct=args.min_return_pct,
            max_drawdown_pct=args.max_drawdown_pct,
            require_mean_reverting=args.require_mean_reverting,
            require_no_breaks=args.require_no_breaks,
            parallel=not args.no_parallel,
            capital=args.capital,
            output_json=args.json,
            output_file=args.output_md.replace(".md", ".json") if args.output_md and args.json else None,
            output_md=args.output_md,
        )
        return

    if args.pair:
        a, b = [t.strip().upper() for t in args.pair.split(",")]
        result = analyze_pair_cli(
            a, b, args.start, args.end, args.significance,
            use_purged_wfv=not args.no_purged,
            run_ml=args.ml,
            save_plots=not args.no_plots,
            output_json=args.json,
            capital=args.capital,
        )
        return


if __name__ == "__main__":
    main()
