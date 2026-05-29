"""
Auto-Discovery — automatically finds profitable pairs from a universe of stocks.

Pipeline:
    1. Screen: quick EG cointegration on all O(n�) pairs in parallel
    2. Filter: keep pairs with p < significance, sort by -log10(p)
    3. Analyze: full pipeline on top N candidates in parallel
    4. Score: rank by composite score (PairRanker)
    5. Filter: apply profitability filters (Sharpe, return, drawdown, regime, breaks)
    6. Output: ranked table + optional JSON + optional Markdown report
"""

from __future__ import annotations

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
)
from .cointegration import CointegrationTester
from .data import DataManager
from .pipeline import PairAnalyzer
from .ranking import PairRanker

logger = logging.getLogger("stats_arb.discover")


def resolve_universe(
    universe: str | list[str] | None = None,
    universe_file: str | None = None,
) -> list[str]:
    """Resolve a universe of tickers from presets, list, or file."""
    if universe_file:
        with open(universe_file) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        logger.info(f"Loaded {len(tickers)} tickers from {universe_file}")
        return tickers

    if isinstance(universe, list):
        return [t.upper() for t in universe]

    if isinstance(universe, str):
        parts = [t.strip().upper() for t in universe.split(",")]
        if len(parts) == 1 and parts[0].lower() in _UNIVERSE_PRESETS:
            return _fetch_preset_tickers(parts[0].lower())
        if len(parts) >= 1 and parts[0]:
            return parts

    logger.info("No universe specified — defaulting to S&P 500")
    return _fetch_preset_tickers("sp500")


_UNIVERSE_PRESETS: dict[str, str] = {
    "sp500": "S&P 500",
    "nasdaq100": "NASDAQ-100",
    "dow30": "Dow Jones Industrial Average",
}


def _fetch_preset_tickers(preset: str) -> list[str]:
    if preset == "sp500":
        return _get_sp500_tickers()
    if preset == "nasdaq100":
        return _get_nasdaq100_tickers()
    if preset == "dow30":
        return _get_dow30_tickers()
    raise ValueError(f"Unknown preset: {preset}")


def _fetch_wikipedia_table(url: str) -> pd.DataFrame:
    """Fetch a Wikipedia HTML table using proper headers to avoid 403."""
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(html)


def _get_sp500_tickers() -> list[str]:
    try:
        table = _fetch_wikipedia_table(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        tickers = sorted(table["Symbol"].tolist())
        logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 tickers: {e}")
        raise RuntimeError(
            "Could not fetch S&P 500 tickers. "
            "Use --universe with comma-separated tickers or --universe-file."
        ) from e


def _get_nasdaq100_tickers() -> list[str]:
    try:
        table = _fetch_wikipedia_table(
            "https://en.wikipedia.org/wiki/Nasdaq-100"
        )[4]
        tickers = sorted(table["Ticker"].tolist())
        logger.info(f"Fetched {len(tickers)} NASDAQ-100 tickers from Wikipedia")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch NASDAQ-100 tickers: {e}")
        raise RuntimeError(
            "Could not fetch NASDAQ-100 tickers. "
            "Use --universe with comma-separated tickers or --universe-file."
        ) from e


def _get_dow30_tickers() -> list[str]:
    return [
        "AAPL", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD",
        "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
    ]


def screen_pairs(
    tickers: list[str],
    start: str,
    end: str,
    significance: float = DEFAULT_SIGNIFICANCE,
    parallel: bool = True,
) -> list[tuple[str, str, float]]:
    """Phase 1: quick EG cointegration on all pairs.

    Returns sorted list of (a, b, -log10(p_value)) for cointegrated pairs.
    """
    pair_list = list(combinations(tickers, 2))
    logger.info(
        f"Screening {len(tickers)} tickers = {len(pair_list):,} pairs ..."
    )

    results: list[tuple[str, str, float]] = []

    def _test(pair: tuple[str, str]) -> tuple[str, str, float]:
        a, b = pair
        try:
            dm = DataManager([a, b], start, end)
            prices = dm.fetch()
            ct = CointegrationTester(prices, significance)
            res = ct.run()
            if res.is_cointegrated:
                return (a, b, -np.log10(max(res.p_value, 1e-15)))
            return (a, b, 0.0)
        except Exception as e:
            logger.debug(f"Screening failed {a}/{b}: {e}")
            return (a, b, 0.0)

    if parallel and len(pair_list) > 2:
        import warnings
        max_workers = min(8, len(pair_list))
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_test, p): p for p in pair_list}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception:
                        pass
        except (MemoryError, OSError) as e:
            warnings.warn(f"Parallel screening failed ({e}), falling back to sequential")
            for p in pair_list:
                results.append(_test(p))
    else:
        for p in pair_list:
            results.append(_test(p))

    scored = [(a, b, s) for a, b, s in results if s > 0]
    scored.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Found {len(scored)} cointegrated pairs (p < {significance})")
    return scored


def _analyze_single(args: tuple) -> Any:
    a, b, start, end, sig, capital = args
    try:
        analyzer = PairAnalyzer(
            a, b, start, end, significance=sig, capital=capital,
        )
        return analyzer.analyze()
    except Exception as e:
        logger.error(f"Full analysis failed {a}/{b}: {e}")
        return None


def auto_discover(
    universe: str | list[str] | None = None,
    universe_file: str | None = None,
    start: str = "2015-01-01",
    end: str | None = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    top_candidates: int = AUTO_DISCOVER_TOP_CANDIDATES,
    min_sharpe: float = AUTO_DISCOVER_MIN_SHARPE,
    min_return_pct: float = AUTO_DISCOVER_MIN_RETURN_PCT,
    max_drawdown_pct: float = AUTO_DISCOVER_MAX_DRAWDOWN_PCT,
    require_mean_reverting: bool = False,
    require_no_breaks: bool = False,
    parallel: bool = True,
    capital: float = 100_000.0,
    output_json: bool = False,
    output_file: str | None = None,
    output_md: str | None = None,
) -> list[dict[str, Any]]:
    """Run the full auto-discovery pipeline.

    Parameters
    ----------
    universe : str or list of str, optional
        Preset name (sp500/nasdaq100/dow30) or comma-separated tickers.
    universe_file : str, optional
        Path to file with one ticker per line.
    start, end : str
        Date range for analysis.
    significance : float
        Cointegration p-value threshold.
    top_candidates : int
        Number of top cointegrated pairs to fully analyze.
    min_sharpe : float
        Minimum Sharpe ratio for profitability filter.
    min_return_pct : float
        Minimum total return %.
    max_drawdown_pct : float
        Maximum allowable drawdown % (e.g. -50.0).
    require_mean_reverting : bool
        Only keep pairs currently in mean-reverting regime.
    require_no_breaks : bool
        Exclude pairs with structural breaks.
    parallel : bool
        Use parallel processing.
    capital : float
        Initial capital for backtests.
    output_json : bool
        Print JSON to stdout.
    output_file : str, optional
        Save JSON to file.
    output_md : str, optional
        Save Markdown report to file.

    Returns
    -------
    list of dict
        Ranked and filtered pairs with metrics.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    tickers = resolve_universe(universe, universe_file)
    logger.info(f"Universe: {len(tickers)} tickers")
    logger.info(f"Period: {start} -> {end}")

    screened = screen_pairs(tickers, start, end, significance, parallel)
    if not screened:
        logger.warning("No cointegrated pairs found")
        return []

    candidates = screened[:top_candidates]
    logger.info(
        f"Top {len(candidates)} candidates for full analysis "
        f"(from {len(screened)} cointegrated)"
    )

    pair_args = [
        (a, b, start, end, significance, capital)
        for a, b, _ in candidates
    ]

    logger.info(f"Running full pipeline on {len(pair_args)} pairs...")
    all_results: list[Any] = []

    if parallel and len(pair_args) > 2:
        max_workers = min(8, len(pair_args))
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_analyze_single, args): args
                    for args in pair_args
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Analysis failed: {e}")
        except (MemoryError, OSError) as e:
            logger.warning(f"Parallel analysis failed ({e}), falling back to sequential")
            for args in pair_args:
                result = _analyze_single(args)
                if result is not None:
                    all_results.append(result)
    else:
        for args in pair_args:
            result = _analyze_single(args)
            if result is not None:
                all_results.append(result)

    if not all_results:
        logger.warning("No full analyses succeeded")
        return []

    ranker = PairRanker()
    ranked = ranker.rank(all_results, top_n=len(all_results))
    filtered = _apply_filters(
        ranked, all_results, min_sharpe, min_return_pct,
        max_drawdown_pct, require_mean_reverting, require_no_breaks,
    )

    output = _format_results(filtered, all_results)
    _print_results(output)

    if output_json:
        print(json.dumps(output, indent=2))

    if output_file:
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"JSON saved: {output_file}")

    if output_md:
        _save_markdown(output, output_md, start, end)

    return output


def _apply_filters(
    ranked: list,
    all_results: list,
    min_sharpe: float,
    min_return_pct: float,
    max_drawdown_pct: float,
    require_mean_reverting: bool,
    require_no_breaks: bool,
) -> list[tuple]:
    lookup: dict[tuple[str, str], Any] = {}
    for r in all_results:
        lookup[(r.ticker_a.upper(), r.ticker_b.upper())] = r
        lookup[(r.ticker_b.upper(), r.ticker_a.upper())] = r

    passed = []
    for rp in ranked:
        key = (rp.ticker_a.upper(), rp.ticker_b.upper())
        orig = lookup.get(key)
        if orig is None:
            continue

        bt = (
            getattr(orig, "wf_backtest", None)
            or getattr(orig, "in_sample_backtest", None)
            or getattr(orig, "backtest", None)
        )
        if bt is None:
            continue

        sharpe = getattr(bt, "sharpe_ratio", 0.0)
        ret = getattr(bt, "total_return_pct", 0.0)
        dd = getattr(bt, "max_drawdown_pct", 0.0)
        regime = getattr(orig, "regime", None)
        current_regime = getattr(regime, "current_regime", None)
        has_breaks = getattr(regime, "structural_break", False) if regime else False

        if sharpe < min_sharpe:
            continue
        if ret < min_return_pct:
            continue
        if dd < max_drawdown_pct:
            continue
        if require_mean_reverting:
            rs = str(current_regime).lower()
            if "mean_revert" not in rs:
                continue
        if require_no_breaks and has_breaks:
            continue

        passed.append((rp, orig, bt, sharpe, ret, dd, current_regime, has_breaks))

    return passed


def _format_results(
    filtered: list[tuple],
    all_results: list,
) -> list[dict[str, Any]]:
    output = []
    for rank_idx, (rp, orig, bt, sharpe, ret, dd, regime, has_breaks) in enumerate(filtered, 1):
        wfbt = getattr(orig, "wf_backtest", None)
        wf_sharpe = getattr(wfbt, "sharpe_ratio", None) if wfbt else None

        coint = getattr(orig, "cointegration", None)
        spread = getattr(orig, "spread", None)
        wf = getattr(orig, "walk_forward", None) or {}
        num_trades = getattr(bt, "num_trades", 0)
        win_rate = getattr(bt, "win_rate_pct", 0.0)

        entry = {
            "rank": rank_idx,
            "pair": f"{rp.ticker_a}/{rp.ticker_b}",
            "ticker_a": rp.ticker_a,
            "ticker_b": rp.ticker_b,
            "composite_score": round(rp.composite_score, 4),
            "coint_p_value": round(getattr(coint, "p_value", 1.0), 6) if coint else 1.0,
            "adj_p_value": round(rp.adjusted_p_value, 6),
            "hedge_ratio": round(getattr(coint, "hedge_ratio", 0.0), 4) if coint else 0.0,
            "half_life_days": round(getattr(spread, "half_life", 0.0), 1) if spread else 0.0,
            "hurst": round(getattr(spread, "hurst_exponent", 0.5), 3) if spread else 0.5,
            "wf_coint_pct": round(getattr(wf, "cointegration_percentage", 0.0), 1) if wf else 0.0,
            "wf_oos_p_value": round(getattr(wf, "avg_oos_p_value", 1.0), 6) if wf else 1.0,
            "return_pct": round(ret, 2),
            "sharpe_ratio": round(sharpe, 3),
            "wf_sharpe_ratio": round(wf_sharpe, 3) if wf_sharpe is not None else None,
            "max_drawdown_pct": round(dd, 2),
            "win_rate_pct": round(win_rate, 1),
            "num_trades": num_trades,
            "regime": str(regime) if regime else "unknown",
            "structural_breaks": has_breaks,
            "confidence": rp.confidence,
            "risk": rp.risk,
        }
        output.append(entry)
    return output


def _print_results(output: list[dict[str, Any]]) -> None:
    if not output:
        print("\nNo pairs passed the profitability filters.")
        print("Try lowering --min-sharpe, --min-return-pct, or --max-drawdown-pct.")
        return

    header = (
        f"{'Rank':<5} {'Pair':<20} {'Score':<7} {'Sharpe':<8} "
        f"{'Ret%':<8} {'DD%':<8} {'WF-Shrp':<8} {'Regime':<18} {'Breaks':<7} {'Conf':<6}"
    )
    sep = "-" * len(header)

    print(f"\n=== Top {len(output)} Profitable Pairs ===")
    print(header)
    print(sep)
    for r in output:
        wf_str = f"{r['wf_sharpe_ratio']:.2f}" if r['wf_sharpe_ratio'] is not None else "N/A"
        print(
            f"{r['rank']:<5} {r['pair']:<20} {r['composite_score']:<7.4f} "
            f"{r['sharpe_ratio']:<8.2f} {r['return_pct']:<8.1f} {r['max_drawdown_pct']:<8.1f} "
            f"{wf_str:<8} {r['regime']:<18} {str(r['structural_breaks']):<7} {r['confidence']:<6}"
        )
    print()


def _save_markdown(
    output: list[dict[str, Any]],
    path: str,
    start: str,
    end: str,
) -> None:
    """Write a Markdown report of discovered pairs."""
    lines = [
        "# Pairs Discovery Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Period:** {start} -> {end}",
        f"**Profitable pairs found:** {len(output)}",
        "",
        "---",
        "",
        "## Ranked Pairs",
        "",
        "| Rank | Pair | Score | Sharpe | Ret% | DD% | WF Sharpe | Half-Life | Hurst | Regime | Breaks | Confidence |",
        "|------|------|-------|--------|------|-----|-----------|-----------|-------|--------|--------|------------|",
    ]

    for r in output:
        wf_str = f"{r['wf_sharpe_ratio']:.2f}" if r['wf_sharpe_ratio'] is not None else "N/A"
        lines.append(
            f"| {r['rank']} | {r['pair']} | {r['composite_score']:.4f} "
            f"| {r['sharpe_ratio']:.2f} | {r['return_pct']:.1f}% | {r['max_drawdown_pct']:.1f}% "
            f"| {wf_str} | {r['half_life_days']:.1f}d | {r['hurst']:.3f} "
            f"| {r['regime']} | {r['structural_breaks']} | {r['confidence']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Pair Details",
        "",
    ])

    for r in output:
        lines.extend([
            f"### {r['rank']}. {r['pair']}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Composite Score | {r['composite_score']:.4f} |",
            f"| Confidence | {r['confidence']} / Risk: {r['risk']} |",
            f"| Cointegration p-value | {r['coint_p_value']:.6f} (adj: {r['adj_p_value']:.6f}) |",
            f"| Hedge Ratio | {r['hedge_ratio']:.4f} |",
            f"| Half-Life | {r['half_life_days']:.1f} days |",
            f"| Hurst Exponent | {r['hurst']:.3f} |",
            f"| WF Cointegration % | {r['wf_coint_pct']:.1f}% |",
            f"| Total Return | {r['return_pct']:.1f}% |",
            f"| Sharpe Ratio | {r['sharpe_ratio']:.2f} |",
            f"| WF Sharpe Ratio | {wf_str if r['wf_sharpe_ratio'] is not None else 'N/A'} |",
            f"| Max Drawdown | {r['max_drawdown_pct']:.1f}% |",
            f"| Win Rate | {r['win_rate_pct']:.1f}% |",
            f"| Total Trades | {r['num_trades']} |",
            f"| Current Regime | {r['regime']} |",
            f"| Structural Breaks | {r['structural_breaks']} |",
            "",
        ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Markdown report saved: {path}")
