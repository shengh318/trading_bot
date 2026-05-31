"""Tests for Time-Series Momentum strategy."""

from __future__ import annotations
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from backend.strategies.base import Portfolio, Signal
from backend.strategies.registry import get_strategy, list_strategies
from backend.strategies.tsmom_strategy import (
    TSMOMBacktestEngine, TSMOMBenchmark, TSMOMConfig, TSMOMFeatures,
    TSMOMForecastModel, TSMOMRegimeDetector, TSMOMSignalGenerator,
    TSMOMStrategy, TSMOMTrendFilter, TSMOMWalkForward,
    TSMOMPortfolioConstruction, TSMOMPositionSizer,
    ParameterResearchConfig, TSMOMParameterResearch, TSMOMVisualizer,
)


def _make_price_df(days=500, start_price=100.0, trend=0.0005):
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    np.random.seed(42)
    returns = np.random.randn(days) * 0.01 + trend
    prices = start_price * np.cumprod(1 + returns)
    high = prices * (1 + np.abs(np.random.randn(days)) * 0.005)
    low = prices * (1 - np.abs(np.random.randn(days)) * 0.005)
    return pd.DataFrame({
        "open": prices * (1 + np.random.randn(days) * 0.002),
        "high": np.maximum(high, prices * 1.001),
        "low": np.minimum(low, prices * 0.999),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, days),
    }, index=dates)


def _make_multi(symbols, days=500):
    return {sym: _make_price_df(days, 100.0 + i * 10, 0.0003 + i * 0.0001)
            for i, sym in enumerate(symbols)}


class TestFeatures:
    def test_compute_features_returns_expected_columns(self):
        fe = TSMOMFeatures(TSMOMConfig())
        features = fe.compute_all(_make_multi(["T"], 500))
        for df in features.values():
            for col in ["momentum_21d", "momentum_63d", "momentum_126d", "momentum_252d",
                        "vol_20d", "vol_60d", "atr", "drawdown",
                        "sharpe_21d", "sharpe_63d", "sortino_21d", "sortino_63d"]:
                assert col in df.columns

    def test_no_lookahead_in_momentum(self):
        df = _make_price_df(300)
        fe = TSMOMFeatures(TSMOMConfig())
        feat_df = fe.compute_all({"T": df})["T"]
        for lb in [21, 63]:
            col = f"momentum_{lb}d"
            for i in range(lb, 250):
                expected = df["close"].iloc[i] / df["close"].iloc[i - lb] - 1
                actual = feat_df[col].iloc[i]
                assert not np.isnan(actual), f"NaN at index {i}, lb={lb}"
                assert abs(actual - expected) < 1e-10

    def test_nan_at_beginning(self):
        fe = TSMOMFeatures(TSMOMConfig())
        feat_df = fe.compute_all({"T": _make_price_df(100)})["T"]
        assert feat_df["momentum_21d"].iloc[:20].isna().all()
        assert feat_df["momentum_252d"].iloc[:251].isna().all()


class TestTrendFilter:
    def test_filter_returns_bool(self):
        df = _make_price_df(300)
        features = TSMOMFeatures(TSMOMConfig()).compute_all({"T": df})
        result = TSMOMTrendFilter(TSMOMConfig()).apply_filters(df, features["T"], ["price_above_ma200"])
        assert result["qualified"].dtype == bool

    def test_empty_filters_all_qualified(self):
        df = _make_price_df(300)
        features = TSMOMFeatures(TSMOMConfig()).compute_all({"T": df})
        result = TSMOMTrendFilter(TSMOMConfig()).apply_filters(df, features["T"], [])
        assert result["qualified"].all()


class TestSignalGenerator:
    def test_binary_signals(self):
        prices = _make_multi(["T1"], 300)
        features = TSMOMFeatures(TSMOMConfig()).compute_all(prices)
        sg = TSMOMSignalGenerator(TSMOMConfig(signal_type="binary"))
        signals = sg.generate_signals(features)
        assert signals["T1"]["signal_binary"].dropna().isin([0.0, 1.0]).all()

    def test_weighted_signals(self):
        prices = _make_multi(["T1"], 300)
        features = TSMOMFeatures(TSMOMConfig()).compute_all(prices)
        sg = TSMOMSignalGenerator(TSMOMConfig(signal_type="weighted"))
        signals = sg.generate_signals(features)
        vals = signals["T1"]["signal_raw"].dropna()
        assert (vals >= -1).all() and (vals <= 1).all()


class TestPositionSizer:
    def test_equal_weight(self):
        prices = _make_multi(["A", "B", "C"], 300)
        fe = TSMOMFeatures(TSMOMConfig()).compute_all(prices)
        sig = TSMOMSignalGenerator(TSMOMConfig(signal_type="binary")).generate_signals(fe)
        sizer = TSMOMPositionSizer(TSMOMConfig(sizing_method="equal"))
        w = sizer.compute_weights(["A", "B", "C"], fe, sig, 250)
        assert len(w) == 3
        assert sum(w.values()) <= 1.0 + 1e-10

    def test_vol_target(self):
        prices = _make_multi(["A"], 300)
        fe = TSMOMFeatures(TSMOMConfig()).compute_all(prices)
        sig = TSMOMSignalGenerator(TSMOMConfig(signal_type="binary")).generate_signals(fe)
        sizer = TSMOMPositionSizer(TSMOMConfig(sizing_method="vol_target", target_volatility=0.15))
        w = sizer.compute_weights(["A"], fe, sig, 250)
        assert "A" in w and w["A"] >= 0

    def test_max_position_cap(self):
        sizer = TSMOMPositionSizer(TSMOMConfig(sizing_method="equal", max_position_pct=0.1))
        w = sizer._apply_caps({"A": 0.5, "B": 0.5}, ["A", "B"])
        assert all(v <= 0.1 for v in w.values())


class TestPortfolioConstruction:
    def test_build_portfolio_returns_valid_df(self):
        prices = _make_multi(["A", "B", "C", "D", "E"], 500)
        fe = TSMOMFeatures(TSMOMConfig(momentum_lookbacks=[21, 63, 126, 252])).compute_all(prices)
        sig = TSMOMSignalGenerator(TSMOMConfig(signal_type="binary")).generate_signals(fe)
        pf = TSMOMPortfolioConstruction(TSMOMConfig(top_n=3, rebalance_freq="ME")).build_portfolio(fe, sig)
        assert not pf.empty and "symbol" in pf.columns and "weight" in pf.columns

    def test_top_n_selection(self):
        prices = _make_multi(["A", "B", "C", "D", "E"], 500)
        fe = TSMOMFeatures(TSMOMConfig(momentum_lookbacks=[21, 63, 126, 252])).compute_all(prices)
        sig = TSMOMSignalGenerator(TSMOMConfig(signal_type="binary")).generate_signals(fe)
        pf = TSMOMPortfolioConstruction(TSMOMConfig(top_n=2, rebalance_freq="ME")).build_portfolio(fe, sig)
        in_pf = pf[pf["in_portfolio"]]
        for d in in_pf["date"].unique():
            assert len(in_pf[in_pf["date"] == d]) <= 2


class TestWalkForward:
    def test_periods_no_lookahead(self):
        periods = TSMOMWalkForward().generate_periods(1000, 500, 100, True)
        for p in periods:
            assert p.test_start >= p.train_end

    def test_expanding_windows_widen(self):
        periods = TSMOMWalkForward().generate_periods(1000, 300, 100, True)
        for i in range(1, len(periods)):
            assert periods[i].train_end >= periods[i - 1].train_end

    def test_rolling_windows_fixed(self):
        periods = TSMOMWalkForward().generate_periods(1000, 300, 100, False)
        for p in periods:
            assert p.train_end - p.train_start == 300


class TestBacktestEngine:
    def test_backtest_produces_result(self):
        prices = _make_multi(["A", "B", "C"], 500)
        config = TSMOMConfig(symbols=["A", "B", "C"], top_n=2, initial_capital=100000.0,
                             momentum_lookbacks=[21, 63, 126], rebalance_freq="ME", transaction_cost_pct=0.001)
        result = TSMOMBacktestEngine(config).run(prices)
        assert result is not None and not result.equity_curve.empty

    def test_metrics_computed(self):
        prices = _make_multi(["A", "B"], 500)
        config = TSMOMConfig(symbols=["A", "B"], top_n=2, initial_capital=100000.0,
                             momentum_lookbacks=[21, 63, 126], rebalance_freq="ME")
        result = TSMOMBacktestEngine(config).run(prices)
        for key in ["total_return_pct", "annual_return_pct", "sharpe_ratio",
                     "max_drawdown_pct", "volatility_pct", "num_trades"]:
            assert key in result.summary_metrics

    def test_drawdown_series(self):
        prices = _make_multi(["A"], 300)
        config = TSMOMConfig(symbols=["A"], top_n=1, initial_capital=100000.0)
        result = TSMOMBacktestEngine(config).run(prices)
        assert not result.drawdown_series.empty and "drawdown" in result.drawdown_series.columns


class TestBenchmark:
    def test_benchmark_comparison(self):
        prices = _make_multi(["A", "B"], 500)
        config = TSMOMConfig(symbols=["A", "B"], top_n=2, initial_capital=100000.0)
        result = TSMOMBacktestEngine(config).run(prices)
        bench = TSMOMBenchmark(config).compare(result, {"SPY": _make_price_df(500)})
        assert isinstance(bench, dict)
        if bench:
            m = list(bench.values())[0]
            for k in ["alpha", "beta", "information_ratio", "tracking_error"]:
                assert k in m


class TestRegimeDetector:
    def test_detect_valid_df(self):
        prices = _make_multi(["SPY"], 600)
        rd = TSMOMRegimeDetector().detect_regimes(prices)
        assert isinstance(rd, pd.DataFrame) and "regime" in rd.columns

    def test_analyze_by_regime(self):
        prices = _make_multi(["SPY"], 600)
        config = TSMOMConfig(symbols=["SPY"], top_n=1, initial_capital=100000.0)
        result = TSMOMBacktestEngine(config).run(prices)
        rd = TSMOMRegimeDetector()
        regimes = rd.detect_regimes(prices)
        analysis = rd.analyze_by_regime(result.equity_curve, regimes)
        assert isinstance(analysis, dict)
        for name, m in analysis.items():
            for k in ["days", "return_pct", "sharpe"]:
                assert k in m


class TestVisualizer:
    def test_imports(self):
        assert TSMOMVisualizer() is not None


class TestParameterResearch:
    def test_grid_runs(self):
        prices = _make_multi(["A", "B"], 300)
        rc = ParameterResearchConfig(lookbacks=[[21], [63]], trend_filter_sets=[[], ["price_above_ma200"]],
                                     rebalance_freqs=["ME"], top_n_values=[1], sizing_methods=["equal"])
        results = TSMOMParameterResearch(TSMOMConfig(symbols=["A", "B"])).run_grid(prices, rc)
        assert len(results) > 0

    def test_to_dataframe(self):
        prices = _make_multi(["A"], 300)
        rc = ParameterResearchConfig(lookbacks=[[21]], trend_filter_sets=[[]],
                                     rebalance_freqs=["ME"], top_n_values=[1], sizing_methods=["equal"])
        results = TSMOMParameterResearch(TSMOMConfig(symbols=["A"])).run_grid(prices, rc)
        df = TSMOMParameterResearch(TSMOMConfig(symbols=["A"])).to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "sharpe_ratio" in df.columns


class TestForecastModel:
    def test_prepare_features(self):
        prices = _make_multi(["A", "B"], 500)
        features = TSMOMFeatures(TSMOMConfig(momentum_lookbacks=[21, 63, 126, 252])).compute_all(prices)
        fm = TSMOMForecastModel()
        X, y = fm.prepare_features(features)
        assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series)
        if not X.empty:
            assert len(X) == len(y)

    def test_ml_augments_signal(self):
        fm = TSMOMForecastModel()
        augmented = fm.augment_signal(np.array([0.5, -0.3, 0.1]), np.array([0.8, 0.2, 0.6]), 0.5)
        assert augmented[0] == 0.5 and augmented[1] == 0.0 and augmented[2] == 0.1


class TestTSMOMStrategy:
    def test_registered(self):
        names = [s["name"] for s in list_strategies()]
        assert "TSMOM Strategy" in names

    def test_instantiated(self):
        s = get_strategy("TSMOM Strategy")
        assert isinstance(s, TSMOMStrategy) and s.momentum_lookback == 126

    def test_custom_params(self):
        s = get_strategy("TSMOM Strategy", {"momentum_lookback": 63, "long_only": False,
                                            "trend_filter_type": "ma50_above_ma200", "stop_loss_pct": 0.05})
        assert s.momentum_lookback == 63 and s.long_only is False
        assert s.trend_filter_type == "ma50_above_ma200" and s.stop_loss_pct == 0.05

    def test_init_computes_indicators(self):
        s = TSMOMStrategy(momentum_lookback=63)
        s.init(_make_price_df(300))
        assert s._mom_series is not None and s._ma_long is not None

    def test_next_returns_valid_signal(self):
        s = TSMOMStrategy(momentum_lookback=63)
        df = _make_price_df(300)
        s.init(df)
        p = Portfolio(10000)
        for i in range(250, 260):
            sig = s.next(i, df, p)
            assert sig in [Signal.BUY, Signal.SELL, Signal.HOLD, Signal.EXIT]

    def test_trend_filter_gate(self):
        s = TSMOMStrategy(momentum_lookback=63, use_trend_filter=True, trend_filter_type="price_above_ma200")
        df = _make_price_df(300)
        s.init(df)
        result = s._check_filter(260, df)
        assert isinstance(result, bool)
