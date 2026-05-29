"""Pairs trading API — cointegration analysis, spread analysis, ranking, heatmap."""

import logging
from datetime import datetime

from fastapi import APIRouter

from backend.api.models import (
    BacktestMetricsModel,
    CointegrationResultModel,
    CorrelationMetrics,
    HeatmapRequest,
    HeatmapResponse,
    JohansenResultModel,
    PairAnalysisRequest,
    PairAnalysisResponse,
    PairData,
    PairRankRequest,
    PairRankResponse,
    RegimeResultModel,
    SpreadResultModel,
    WalkForwardMetrics,
    WFBacktestMetricsModel,
)
from backend.stats_arb.cointegration import CointegrationTester
from backend.stats_arb.data import DataManager
from backend.stats_arb.pipeline import PairAnalyzer
from backend.stats_arb.ranking import PairRanker

logger = logging.getLogger("pairs_api")

router = APIRouter()


def _series_to_list(series, max_points: int = 1000) -> list[dict[str, float]]:
    """Convert a pandas Series to a list of {time, value} dicts, sampling if needed."""
    if series is None or len(series) == 0:
        return []
    step = max(1, len(series) // max_points)
    sampled = series.iloc[::step]
    return [
        {"time": int(idx.timestamp()), "value": round(float(val), 6)}
        for idx, val in sampled.items()
    ]


def _pair_to_response(pair_result) -> PairData:
    corr_list = []
    for w, res in pair_result.correlation.items():
        corr_list.append(CorrelationMetrics(
            window=w,
            current=round(getattr(res, "current", 0), 4),
            mean=round(getattr(res, "mean", 0), 4),
            std=round(getattr(res, "std", 0), 4),
            drift=round(getattr(res, "overall_slope", getattr(res, "drift", 0)), 4),
            regime_changes=getattr(res, "regime_changes", getattr(res, "threshold_crossings", {}).get("sign_flip", 0)),
            threshold_crossings=getattr(res, "threshold_crossings", {}),
            correlation_collapse=getattr(res, "correlation_collapse", False),
            overall_slope=round(getattr(res, "overall_slope", 0), 6),
            max_correlation_drawdown=round(getattr(res, "max_correlation_drawdown", 0), 4),
            stability_score=round(getattr(res, "stability_score", 0), 4),
        ))

    c = pair_result.cointegration
    hr = pair_result.hedge_ratio
    coint_model = CointegrationResultModel(
        method=c.method,
        p_value=round(float(c.p_value), 6),
        test_statistic=round(float(c.test_statistic), 4),
        critical_values={k: round(float(v), 4) for k, v in c.critical_values.items()},
        hedge_ratio=round(float(c.hedge_ratio), 6),
        hedge_ratio_intercept=round(float(getattr(hr, "intercept", 0)), 6),
        hedge_ratio_method=getattr(hr, "method", "ols"),
        is_cointegrated=c.is_cointegrated,
    )

    johansen_model = None
    if pair_result.johansen is not None:
        j = pair_result.johansen
        johansen_model = JohansenResultModel(
            is_cointegrated=j.is_cointegrated,
            trace_statistic=round(float(j.trace_statistic), 4),
            trace_critical_values={k: round(float(v), 4) for k, v in j.trace_critical_values.items()},
            eigenvalue_statistic=round(float(j.eigenvalue_statistic), 4),
            eigenvalue_critical_values={k: round(float(v), 4) for k, v in j.eigenvalue_critical_values.items()},
        )

    s = pair_result.spread
    spread_model = SpreadResultModel(
        mean=round(float(s.mean), 6),
        std=round(float(s.std), 6),
        current_zscore=round(float(s.current_zscore), 4),
        half_life=round(float(s.half_life), 2) if s.half_life != float("inf") else 9999.0,
        hurst_exponent=round(float(s.hurst_exponent), 4),
        adf_statistic=round(float(s.adf_statistic), 4),
        adf_pvalue=round(float(s.adf_pvalue), 6),
        is_stationary=s.is_stationary,
        mean_reversion_speed=round(float(s.mean_reversion_speed), 6),
        expected_time_to_mean=round(float(s.expected_time_to_mean), 2) if s.expected_time_to_mean != float("inf") else 9999.0,
        persistence=round(float(s.persistence), 4),
        spread_autocorr_5=round(float(s.spread_autocorr_5), 4),
        variance_ratio=round(float(s.variance_ratio), 4),
        spread_series=_series_to_list(s.spread),
        zscore_series=_series_to_list(s.zscore_series),
    )

    wf = pair_result.walk_forward
    wf_model = WalkForwardMetrics(
        avg_train_p_value=round(float(getattr(wf, "avg_train_p_value", wf.avg_p_value if hasattr(wf, "avg_p_value") else 1.0)), 6),
        avg_oos_p_value=round(float(getattr(wf, "avg_oos_p_value", 1.0)), 6),
        avg_half_life=round(float(wf.avg_half_life), 2) if getattr(wf, "avg_half_life", float("inf")) != float("inf") else 9999.0,
        cointegration_percentage=round(float(wf.cointegration_percentage), 2),
        avg_spread_sharpe=round(float(wf.avg_spread_sharpe), 4),
        hedge_ratio_stability=round(float(wf.hedge_ratio_stability), 4),
        avg_spread_drawdown=round(float(getattr(wf, "avg_spread_drawdown", 0)), 2),
        oos_stationarity_pct=round(float(getattr(wf, "oos_stationarity_pct", 0)), 2),
        regime_stability_pct=round(float(getattr(wf, "regime_stability_pct", 0)), 2),
        num_folds=len(wf.folds) if wf.folds else 0,
    )

    bt = getattr(pair_result, "in_sample_backtest", None) or getattr(pair_result, "backtest", None)
    bt_model = BacktestMetricsModel(
        total_return_pct=round(float(bt.total_return_pct), 2),
        annualised_return_pct=round(float(getattr(bt, "annualised_return_pct", 0)), 2),
        sharpe_ratio=round(float(bt.sharpe_ratio), 4),
        sortino_ratio=round(float(getattr(bt, "sortino_ratio", 0)), 4),
        calmar_ratio=round(float(getattr(bt, "calmar_ratio", 0)), 4),
        max_drawdown_pct=round(float(bt.max_drawdown_pct), 2),
        win_rate_pct=round(float(bt.win_rate_pct), 2),
        num_trades=bt.num_trades,
        avg_holding_period=round(float(bt.avg_holding_period), 2),
        turnover=round(float(bt.turnover), 4),
        final_equity=round(float(bt.final_equity), 2),
        profit_factor=round(float(getattr(bt, "profit_factor", 0)), 4),
        exposure_pct=round(float(getattr(bt, "exposure_pct", 0)), 2),
        beta_to_market=round(float(getattr(bt, "beta_to_market", 0)), 4),
        equity_curve=_series_to_list(bt.equity_curve),
        drawdown_series=_series_to_list(getattr(bt, "drawdown_series", None)),
    )

    r = pair_result.regime
    wfbt = getattr(pair_result, "wf_backtest", None)
    wfbt_model = WFBacktestMetricsModel()
    if wfbt is not None:
        wfbt_model = WFBacktestMetricsModel(
            total_return_pct=round(float(getattr(wfbt, "total_return_pct", 0)), 2),
            annualised_return_pct=round(float(getattr(wfbt, "annualised_return_pct", 0)), 2),
            sharpe_ratio=round(float(getattr(wfbt, "sharpe_ratio", 0)), 4),
            max_drawdown_pct=round(float(getattr(wfbt, "max_drawdown_pct", 0)), 2),
            win_rate_pct=round(float(getattr(wfbt, "win_rate_pct", 0)), 2),
            num_trades=getattr(wfbt, "num_trades", 0),
            avg_holding_period=round(float(getattr(wfbt, "avg_holding_period", 0)), 2),
            turnover=round(float(getattr(wfbt, "turnover", 0)), 4),
            num_folds=len(getattr(wfbt, "folds", [])),
        )

    regime_model = RegimeResultModel(
        current_regime=r.current_regime.value if hasattr(r.current_regime, "value") else str(r.current_regime),
        trading_allowed=r.trading_allowed,
        signal_suppressed=r.signal_suppressed,
        structural_break=r.structural_break,
        correlation_breakdown=r.correlation_breakdown,
        spread_variance_expansion=r.spread_variance_expansion,
        vix_level=round(float(r.vix_level), 2),
        regime_summary={k: round(float(v), 3) for k, v in r.regime_summary.items()},
        cusum_break_detected=getattr(r, "cusum_break_detected", False),
        cusum_break_indices=getattr(r, "cusum_break_indices", []),
        chow_break_detected=getattr(r, "chow_break_detected", False),
        chow_break_dates=getattr(r, "chow_break_dates", []),
        bai_perron_breaks=getattr(r, "bai_perron_breaks", []),
        num_structural_breaks=getattr(r, "num_structural_breaks", 0),
    )

    return PairData(
        ticker_a=pair_result.ticker_a,
        ticker_b=pair_result.ticker_b,
        correlations=corr_list,
        cointegration=coint_model,
        johansen=johansen_model,
        spread=spread_model,
        walk_forward=wf_model,
        backtest=bt_model,
        wf_backtest=wfbt_model,
        regime=regime_model,
        score=round(float(pair_result.score), 4),
    )


def _empty_pair(ticker_a: str, ticker_b: str) -> PairData:
    return PairData(
        ticker_a=ticker_a, ticker_b=ticker_b,
        correlations=[],
        cointegration=CointegrationResultModel(
            method="engle-granger", p_value=1.0, test_statistic=0.0,
            critical_values={}, hedge_ratio=0.0, is_cointegrated=False,
        ),
        spread=SpreadResultModel(
            mean=0.0, std=0.0, current_zscore=0.0,
            half_life=0.0, hurst_exponent=0.5,
            spread_series=[], zscore_series=[],
        ),
        walk_forward=WalkForwardMetrics(
            avg_half_life=0.0, cointegration_percentage=0.0,
            avg_spread_sharpe=0.0, hedge_ratio_stability=0.0, num_folds=0,
        ),
        backtest=BacktestMetricsModel(
            total_return_pct=0.0, sharpe_ratio=0.0,
            max_drawdown_pct=0.0, win_rate_pct=0.0,
            num_trades=0, avg_holding_period=0.0,
            turnover=0.0, final_equity=0.0, equity_curve=[],
        ),
    )


@router.post("/api/pairs/analyze", response_model=PairAnalysisResponse)
def analyze_pair(req: PairAnalysisRequest):
    end = req.end or datetime.today().strftime("%Y-%m-%d")
    analyzer = PairAnalyzer(
        req.ticker_a.upper(),
        req.ticker_b.upper(),
        req.start,
        end,
        significance=req.significance,
        run_johansen=req.run_johansen,
    )
    try:
        result = analyzer.analyze()
    except Exception as e:
        logger.warning(f"Pair analysis failed: {e}")
        return PairAnalysisResponse(status=f"error: {e}", pair=_empty_pair(req.ticker_a.upper(), req.ticker_b.upper()))

    pair_data = _pair_to_response(result)
    return PairAnalysisResponse(status="ok", pair=pair_data)


@router.post("/api/pairs/rank", response_model=PairRankResponse)
def rank_pairs(req: PairRankRequest):
    end = req.end or datetime.today().strftime("%Y-%m-%d")
    results = []
    for pair in req.pairs:
        try:
            analyzer = PairAnalyzer(
                pair[0].upper(), pair[1].upper(),
                req.start, end,
                significance=req.significance,
            )
            result = analyzer.analyze()
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed {pair[0]}/{pair[1]}: {e}")
            continue

    ranker = PairRanker()
    ranked = ranker.rank(results, top_n=req.top_n)
    result_lookup = {(r.ticker_a.upper(), r.ticker_b.upper()): r for r in results}
    ranked_data = []
    for rp in ranked:
        orig = result_lookup.get((rp.ticker_a.upper(), rp.ticker_b.upper()))
        if orig is not None:
            ranked_data.append(_pair_to_response(orig))
    return PairRankResponse(status="ok", ranked_pairs=ranked_data)


@router.post("/api/pairs/heatmap", response_model=HeatmapResponse)
def cointegration_heatmap(req: HeatmapRequest):
    end = req.end or datetime.today().strftime("%Y-%m-%d")
    tickers = [t.upper() for t in req.tickers]
    try:
        import numpy as np
        from itertools import combinations
        matrix = {}
        pair_list = list(combinations(tickers, 2))
        for a, b in pair_list:
            try:
                dm = DataManager([a, b], req.start, end)
                prices = dm.fetch()
                ct = CointegrationTester(prices, req.significance)
                res = ct.run()
                raw = -np.log10(max(res.p_value, 1e-15))
                val = round(float(raw), 2) + 0.0  # +0.0 avoids -0.0 in JSON
                logger.info(f"Heatmap {a}/{b}: -log10(p)={val}, p={res.p_value:.6g}")
            except Exception as e:
                logger.warning(f"Heatmap {a}/{b}: {e}")
                val = 0.0
            matrix.setdefault(a, {})[b] = val
            matrix.setdefault(b, {})[a] = val
        for t in tickers:
            matrix.setdefault(t, {})[t] = 0.0
    except Exception as e:
        return HeatmapResponse(status=f"error: {e}", tickers=tickers, matrix={})

    return HeatmapResponse(status="ok", tickers=tickers, matrix=matrix)
