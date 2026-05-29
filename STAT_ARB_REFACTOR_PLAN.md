# Statistical Arbitrage Framework — Refactoring Plan

## Context

This plan addresses research-validity issues in the existing `backend/stats_arb/` package. The framework (~3,700 lines across 14 modules) already covers the full pipeline — correlation, cointegration, spread analysis, regime detection, walk-forward validation, strategy, backtest, ranking, ML, visualization, and CLI. However, several critical issues compromise out-of-sample validity:

1. **Walk-forward leak**: `coint()` re-estimates hedge ratio on test data
2. **Decoupled backtest**: Strategy uses full-sample HR, not walk-forward folds
3. **No structural break detection**: Framework can't detect when pair relationships break
4. **Correlation metrics are misleading**: `regime_changes` counts zero-crossings (always 0), drift uses only two halves
5. **Duplicate hedge ratio code**: Same OLS in 3 files, will diverge
6. **No multiple comparison correction**: 10 pairs × p=0.05 → ~0.5 false positives expected
7. **Z-score lookahead bias**: `strategy.py` uses full-sample mean/std

## What is NOT affected

- `backend/ml/` (train.py, auto_optimize.py, stacking.py, models/)
- `backend/strategies/ml_strategy.py` (38-feature ML strategy)
- `backend/stats_arb/ml_models.py` (SpreadPredictor — stat-arb ML, not the main ML pipeline)
- `backend/stats_arb/pipeline.py:_run_ml()` — unchanged
- `frontend/` (except Pairs.tsx type updates in Phase 8)

Your model training continues uninterrupted.

---

## Phase 0 — Consolidation

### 0.A Delete legacy duplicate
- **File**: `backend/pairs_trading.py` — delete entirely
- **Rationale**: All 1,759 lines are superseded by `stats_arb/` modules. The API already imports from `stats_arb`, so this is dead code that will diverge.

### 0.B Centralize hedge ratio estimation (Fix #6)
- **File**: `backend/stats_arb/hedge_ratio.py`
  - Add standalone function:
    ```python
    def estimate_ols(
        a: np.ndarray,
        b: np.ndarray,
        add_const: bool = True,
        robust: bool = False,
    ) -> float:
        """Single source of truth for OLS hedge ratio estimation."""
    ```
  - `robust=True` uses `sklearn.linear_model.HuberRegressor`
  - When `add_const=False`, regresses a ~ b with no intercept
- **File**: `backend/stats_arb/cointegration.py` — remove inline `_estimate_hedge_ratio` at line 161, import from `hedge_ratio`
- **File**: `backend/stats_arb/walk_forward.py` — remove inline `_estimate_hedge_ratio` at line 237, import from `hedge_ratio`
- **File**: `backend/stats_arb/__init__.py` — export `estimate_ols` from hedge_ratio module

---

## Phase 1 — Walk-Forward OOS Leakage (Fix #1)

### 1.A Replace EG on test data with ADF on OOS spread
- **File**: `backend/stats_arb/walk_forward.py`
  - **Line 211**: Replace:
    ```python
    _, p_val, _ = coint(a_test, b_test, maxlag=1, autolag="AIC")
    ```
    With:
    ```python
    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_p_val = adfuller(spread_test, maxlag=1, autolag="AIC")
    ```
  - **`WalkForwardFold`** dataclass:
    - Rename `p_value: float` → `train_p_value: float` (EG on training data)
    - Add `oos_p_value: float` (ADF on test spread with frozen HR)
  - **`WalkForwardResult`** dataclass:
    - Add `avg_train_p_value: float`
    - Replace `avg_p_value` with `avg_oos_p_value: float`
    - `to_dict()`: expose both as `avg_train_p_value` and `avg_oos_p_value`
  - **`_evaluate_fold`**:
    - Keep `coint(a_train, b_train)` for training EG p-value → stored in `train_p_value`
    - Add `adfuller(spread_test)` → stored in `oos_p_value`
  - **`_aggregate`**: compute both averages

- **File**: `backend/stats_arb/config.py`
  - Add: `WALK_FORWARD_ADF_MAXLAG: int = 1`

### 1.B Update consumers of walk-forward
- **File**: `backend/stats_arb/ranking.py`
  - `_score_pair`: read `avg_oos_p_value` instead of old `avg_p_value`
- **File**: `backend/stats_arb/visualization.py`
  - `_plot_walk_forward`: bar chart shows OOS ADF p-values, not EG p-values
- **File**: `backend/api/models.py`
  - `WalkForwardMetrics`: add `avg_oos_p_value: float = 1.0`, rename `avg_p_value` → `avg_train_p_value`
- **File**: `backend/api/pairs_routes.py`
  - `_pair_to_response`: map new fields

---

## Phase 2 — Walk-Forward Backtest (Fix #2)

### 2.A New walk-forward backtest classes
- **File**: `backend/stats_arb/walk_forward.py` — add:
  ```python
  @dataclass
  class WFBacktestFold:
      """Single fold result from walk-forward backtest."""
      fold: int
      train_start: pd.Timestamp
      train_end: pd.Timestamp
      test_start: pd.Timestamp
      test_end: pd.Timestamp
      hedge_ratio: float
      equity_curve: pd.Series
      trades: list[TradeRecord]
      total_return_pct: float
      sharpe_ratio: float
      max_drawdown_pct: float
      num_trades: int

  @dataclass
  class WFBacktestResult:
      """Aggregated walk-forward backtest results."""
      folds: list[WFBacktestFold]
      combined_equity: pd.Series          # concatenated OOS equity
      combined_trades: list[TradeRecord]   # all trades across folds
      total_return_pct: float
      annualised_return_pct: float
      sharpe_ratio: float
      max_drawdown_pct: float
      win_rate_pct: float
      num_trades: int
      avg_holding_period: float
      turnover: float

      def to_dict(self) -> dict[str, Any]: ...
  ```
- Add class:
  ```python
  class WalkForwardBacktest:
      """Runs actual spread trading strategy on each walk-forward fold.
      
      For each fold:
          1. Estimate hedge ratio on training data ONLY
          2. Run TradingStrategy on test data using fold's hedge ratio
          3. Record fold-level equity curve and trades
      
      No lookahead — each fold knows only its own training data.
      """
      
      def __init__(
          self,
          prices: pd.DataFrame,
          train_days: int = 252,
          test_days: int = 63,
          embargo_days: int = 21,
          z_entry: float = 2.0,
          z_exit: float = 0.0,
          stop_loss: float = 3.0,
          transaction_cost: float = 0.001,
          slippage: float = 0.0005,
          borrow_fee: float = 0.0003,
          max_holding_days: int = 60,
          initial_capital: float = 100_000.0,
      ) -> None: ...
      
      def run(self) -> WFBacktestResult: ...
  ```

### 2.B Update pipeline
- **File**: `backend/stats_arb/pipeline.py`
  - `_run_backtest`: replace with `_run_walk_forward_backtest` calling `WalkForwardBacktest`
  - Keep old `_run_in_sample_backtest` for reference → stored as `PairAnalysisResult.in_sample_backtest`
  - New WF backtest → stored as `PairAnalysisResult.wf_backtest`
  - `PairAnalysisResult`:
    - Rename existing `backtest` to `in_sample_backtest` (clearly labelled "IS Reference")
    - Add `wf_backtest: Optional[WFBacktestResult]`
  - `PairAnalysisResult.to_dict()`: expose both with clear labels

### 2.C Update consumers
- **File**: `backend/stats_arb/visualization.py`
  - Add `_plot_wf_equity_curve`: concatenated OOS equity with fold boundaries
- **File**: `backend/api/models.py`
  - Add `WFBacktestMetricsModel`:
    ```python
    class WFBacktestMetricsModel(BaseModel):
        total_return_pct: float = 0.0
        annualised_return_pct: float = 0.0
        sharpe_ratio: float = 0.0
        max_drawdown_pct: float = 0.0
        win_rate_pct: float = 0.0
        num_trades: int = 0
        avg_holding_period: float = 0.0
        turnover: float = 0.0
        num_folds: int = 0
    ```
  - `PairData`: add `wf_backtest: WFBacktestMetricsModel = WFBacktestMetricsModel()`
- **File**: `backend/api/pairs_routes.py`
  - `_pair_to_response`: map WF backtest

---

## Phase 3 — Structural Break Detection (Fix #3)

### 3.A CUSUM test
- **File**: `backend/stats_arb/regime.py` — add:
  ```python
  def _detect_cusum_break(
      self, confidence: float = 0.95,
  ) -> tuple[bool, list[int], np.ndarray]:
      """CUSUM test via recursive residuals.
      
      - Fits OLS of spread ~ constant
      - Computes recursive residuals
      - Cumulative sum with confidence bounds
      
      Returns:
          has_break: True if cumulative sum exceeds bounds
          break_indices: indices where CUSUM crosses boundary
          cusum_series: full CUSUM statistic series
      """
  ```
  Standard CUSUM: recursive residuals are standardized one-step-ahead forecast errors. Under H0 (no break), CUSUM ~ Brownian bridge. Reject when |CUSUM| exceeds ±0.948 × sqrt(T) at 95%.

### 3.B Chow test
- **File**: `backend/stats_arb/regime.py` — add:
  ```python
  def _detect_chow_break(
      self, candidate_break: int, significance: float = 0.05,
  ) -> bool:
      """Chow breakpoint F-test.
      
      Tests: restricted (single regression) vs unrestricted (split at point).
      F = ((RSS_pooled - RSS_split) / k) / (RSS_split / (n - 2k))
      where k = number of parameters, n = total observations.
      """
  ```

### 3.C Bai-Perron (simplified sequential)
- **File**: `backend/stats_arb/regime.py` — add:
  ```python
  def _detect_bai_perron_breaks(
      self, max_breaks: int = 3, min_segment: int = 60,
  ) -> list[int]:
      """Sequential breakpoint detection.
      
      1. Find single break via argmin RSS over all valid positions
      2. Split at break, recurse on each segment
      3. Stop when: BIC doesn't improve, max_breaks reached, or segment too small
      
      This is a simplified version of Bai-Perron (1998) — not the full
      dynamic programming solution, but captures the essential idea.
      """
  ```

### 3.D Update RegimeResult
- **File**: `backend/stats_arb/regime.py`
  - Add fields:
    ```python
    cusum_break_detected: bool
    cusum_break_indices: list[int]
    chow_break_detected: bool
    chow_break_dates: list[str]
    bai_perron_breaks: list[int]
    num_structural_breaks: int
    ```
  - Update `to_dict()` to include new fields
  - Call all three detection methods in `detect()` (after variance expansion check)
  - Aggregate: `structural_break = cusum_break_detected or chow_break_detected or len(bai_perron_breaks) > 0`

### 3.E Connect to strategy
- **File**: `backend/stats_arb/strategy.py`
  - `TradingStrategy.__init__`: add param `structural_break_detected: bool = False`
  - `_check_exit` (line 255-268): when structural break is flagged and position is open, return True (force exit)
  - `_close_position`: pass exit_reason through
  - `_get_exit_reason`: add `"structural_break"` case
  - Exit order: check structural break BEFORE other exit conditions

### 3.F Connect to ranking
- **File**: `backend/stats_arb/ranking.py`
  - `_score_pair`: add dimension `structural_break_frequency` with weight 0.08
  - Scoring: `max(0, 1.0 - num_breaks / max_expected_breaks)` where `max_expected_breaks = 3`
  - Redistribute weights: reduce `half_life_score` from 0.10 → 0.07, `correlation_stability` from 0.05 → 0.05 to accommodate new dimension

### 3.G Spread variance explosion
- **File**: `backend/stats_arb/spread.py`
  - `SpreadResult`: add field `variance_explosion_events: int`
  - In `SpreadAnalyzer.run()`:
    ```python
    rolling_vol = spread.rolling(63).std()
    expanding_mean_vol = rolling_vol.expanding().mean()
    variance_explosion_events = int(
        (rolling_vol / expanding_mean_vol > 3.0).sum()
    ) if len(spread) > 63 else 0
    ```
  - Update `to_dict()` to include this field

### 3.H Visualization
- **File**: `backend/stats_arb/visualization.py`
  - New method `_plot_structural_breaks(result, pfx)`: 3-panel figure
    - Top: spread with highlighted break regions and break point markers
    - Middle: CUSUM statistic with 95% confidence bounds
    - Bottom: segment means (step function)
  - Update `_plot_spread`: add vertical dashed lines at break points
  - Update `_plot_zscore`: add shaded regions during break windows
  - Update `_plot_regime`: overlay structural break markers

### 3.I Config
- **File**: `backend/stats_arb/config.py` — add:
  ```python
  # ── Structural Break Detection ─────────────────
  CUSUM_CONFIDENCE: float = 0.95
  CHOW_SIGNIFICANCE: float = 0.05
  BAI_PERRON_MAX_BREAKS: int = 3
  BAI_PERRON_MIN_SEGMENT: int = 60
  ```

---

## Phase 4 — Correlation Improvements (Fixes #4 & #5)

### 4.A Threshold crossing detection
- **File**: `backend/stats_arb/correlation.py`
  - Replace `regime_changes: int` field with:
    ```python
    threshold_crossings: dict[str, int]
    # e.g., {"below_0.5": 3, "below_0.3": 1, "sign_flip": 0}
    ```
  - Replace computation (lines 123-125):
    ```python
    # OLD (zero-crossings, always 0 for equity pairs):
    regime_changes = int(((corr_series.shift(1) * corr_series) < 0).sum())
    
    # NEW (meaningful threshold crossings):
    threshold_crossings = {
        "below_0.5": int(((corr_series.shift(1) >= 0.5) & (corr_series < 0.5)).sum()),
        "below_0.3": int(((corr_series.shift(1) >= 0.3) & (corr_series < 0.3)).sum()),
        "sign_flip": int(((corr_series.shift(1) * corr_series) < 0).sum()),
    }
    ```
  - Add `correlation_collapse: bool`:
    ```python
    correlation_collapse = bool(
        corr_series.iloc[-1] < 0.3 and corr_series.mean() > 0.6
    )
    ```

### 4.B Rolling correlation slope (replaces drift)
- **File**: `backend/stats_arb/correlation.py`
  - Replace `drift: float` field (first-half vs second-half means) with:
    ```python
    overall_slope: float  # polyfit slope over full series
    rolling_slope: pd.Series  # 63d window slope series
    ```
  - Replace computation (lines 119-121):
    ```python
    # OLD:
    mid = len(corr_series) // 2
    drift = abs(corr_series.iloc[:mid].mean() - corr_series.iloc[mid:].mean())
    
    # NEW:
    x = np.arange(len(corr_series))
    overall_slope = float(np.polyfit(x, corr_series.values, 1)[0])
    # rolling slope:
    rolling_slope = corr_series.rolling(63).apply(
        lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) == 63 else np.nan,
        raw=False,
    ).dropna()
    ```
  - Keep `drift` as a deprecated computed property for backward compatibility (or remove if clean break is acceptable)

### 4.C Correlation drawdown
- **File**: `backend/stats_arb/correlation.py` — add:
  ```python
  max_correlation_drawdown: float
  
  # Computation:
  corr_drawdown = 1.0 - (corr_series / corr_series.cummax())
  max_correlation_drawdown = float(corr_drawdown.max())
  ```

### 4.D Stability scoring
- **File**: `backend/stats_arb/correlation.py` — add:
  ```python
  stability_score: float
  
  # Computation:
  vol_of_corr = corr_series.rolling(63).std().mean()
  stability_score = 1.0 / (1.0 + vol_of_corr + abs(overall_slope) + max_correlation_drawdown)
  ```

### 4.E Update RollingCorrelationResult
- **File**: `backend/stats_arb/correlation.py`
  - New fields: `threshold_crossings`, `correlation_collapse`, `overall_slope`, `rolling_slope_series`, `max_correlation_drawdown`, `stability_score`
  - Keep `regime_changes` as a computed alias returning `threshold_crossings.get("sign_flip", 0)` for soft backward compat (or remove if breaking change is acceptable)
  - Replace `drift` with `overall_slope` (breaking change — consumers must update)
  - Update `to_dict()` to expose all new fields

### 4.F Update consumers
- **File**: `backend/api/models.py`
  - `CorrelationMetrics`: add `threshold_crossings`, `correlation_collapse`, `overall_slope`, `max_correlation_drawdown`, `stability_score`
  - Keep `drift` with default `0.0` for backward compat? Or remove. Recommend removal for clean API.
  - Keep `regime_changes` as deprecated field marking it optional, or remove.
- **File**: `backend/api/pairs_routes.py` — update `_pair_to_response` mapping
- **File**: `backend/stats_arb/ranking.py`
  - `_score_pair`: read `stability_score` from correlation result instead of computing inverse of std
  - `correlation_stability` component: `components["correlation_stability"] = min(1.0, max(0.0, corr_stability_score))`

---

## Phase 5 — Multiple Comparison Correction (Fix #7)

### 5.A Correction functions
- **File**: `backend/stats_arb/ranking.py` — add:
  ```python
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
      # enforce monotonicity
      adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
      # clamp
      adjusted = np.minimum(adjusted, 1.0)
      # unsort back to original order
      result = np.zeros(n)
      result[sorted_idx] = adjusted
      return result.tolist()
  ```

### 5.B Update ranking
- **File**: `backend/stats_arb/ranking.py`
  - `PairRanker.__init__`: add param `multiple_comparison: str = "bh"` (values: "bonferroni", "bh", "none")
  - `PairRanker.rank()`:
    ```python
    def rank(self, pair_results, top_n=10, multiple_comparison="bh"):
        if not pair_results:
            return []
        p_values = [self._get_coint_p(p) for p in pair_results]
        
        if multiple_comparison == "bonferroni":
            adjusted_p = bonferroni_correct(p_values, len(p_values))
        elif multiple_comparison == "bh":
            adjusted_p = benjamini_hochberg_correct(p_values)
        else:
            adjusted_p = p_values
        
        scored = []
        for pair, adj_p in zip(pair_results, adjusted_p):
            rp = self._score_pair(pair, adjusted_p_value=adj_p)
            scored.append(rp)
        
        scored.sort(key=lambda x: x.composite_score, reverse=True)
        return scored[:top_n]
    ```
  - `_score_pair`: add `adjusted_p_value` param, use it for `coint_p` scoring component
  - `RankedPair`: add `raw_p_value: float` and `adjusted_p_value: float` fields
  - `RankedPair.to_dict()`: expose both

### 5.C CLI and config
- **File**: `backend/stats_arb/cli.py`
  - `analyze_multiple_cli`: pass adjusted p-values to ranker
  - Log: "Raw p: {raw:.6f}, FDR-adjusted: {adj:.6f}"
- **File**: `backend/stats_arb/config.py`
  - Add: `MULTIPLE_COMPARISON_METHOD: str = "bh"`

---

## Phase 6 — Research Validity (Fix #8)

### 6.A Z-score lookahead fix
- **File**: `backend/stats_arb/strategy.py` (lines 187-190)
  - **Current** (lookahead bias — uses full sample):
    ```python
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std
    ```
  - **Fixed** (expanding window — only uses past data):
    ```python
    expanding_mean = spread.expanding().mean()
    expanding_std = spread.expanding().std()
    zscore = (spread - expanding_mean) / expanding_std
    ```
  - Also affects the walk-forward backtest in Phase 2 (which calls `TradingStrategy.execute()`)
  - Add unit test verifying expanding z-score at time t matches manually computed z-score using data[:t]

### 6.B Benchmark comparison
- **File**: `backend/stats_arb/backtest.py`
  - `BacktestEngine.__init__`: add `benchmark_returns: pd.Series | None = None`
  - `BacktestResult`: add fields:
    ```python
    alpha: float       # annualised excess return over benchmark
    beta: float        # strategy beta to benchmark
    information_ratio: float  # alpha / tracking_error
    tracking_error: float     # std of excess returns
    ```
  - `run()`: when benchmark is provided, compute:
    ```python
    excess = strategy_daily_returns - benchmark_daily_returns
    alpha = excess.mean() * 252
    tracking_error = excess.std() * sqrt(252)
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0
    beta = cov(strategy, benchmark) / var(benchmark)
    ```
  - Update `to_dict()` to include benchmark metrics

### 6.C Pipeline purity assertions
- **File**: `backend/stats_arb/pipeline.py`
  - Add logging/disclaimer before backtest:
    ```python
    logger.info(
        "Running full-sample in-sample backtest (REFERENCE ONLY — "
        "contains lookahead bias. For OOS results, use walk-forward backtest.)"
    )
    ```
  - Add assertion on data minimum:
    ```python
    if len(prices) < 504:  # 2 years of daily data
        logger.warning(
            f"Only {len(prices)} days of data — "
            f"walk-forward results may be unreliable"
        )
    ```

### 6.D Realistic costs
- **File**: `backend/stats_arb/config.py`
  - Update defaults if needed:
    ```python
    DEFAULT_MARKET_IMPACT: float = 0.001  # 10bps for illiquid names
    ```

---

## Phase 7 — Tests

### 7.A New test file
- **File**: `backend/tests/test_stats_arb.py`

| # | Test Name | What it Verifies | Category |
|---|---|---|---|
| 1 | `test_walk_forward_no_eg_leakage` | ADF on OOS spread, not EG on test data | Fix #1 |
| 2 | `test_walk_forward_hr_frozen` | HR estimated on train, frozen during test | Fix #1 |
| 3 | `test_walk_forward_embargo_respected` | Purged gap is non-overlapping | Fix #1 |
| 4 | `test_strategy_zscore_no_lookahead` | Z-score uses expanding window | Fix #8 |
| 5 | `test_hedge_ratio_centralized` | All modules import same function | Fix #6 |
| 6 | `test_cusum_detects_break` | CUSUM flags synthetic mean shift | Fix #3 |
| 7 | `test_chow_detects_break` | Chow F-test on known breakpoint | Fix #3 |
| 8 | `test_bai_perron_detects_multi_break` | Sequential detection on multi-break series | Fix #3 |
| 9 | `test_bonferroni_correction` | Correct adjusted p-values | Fix #7 |
| 10 | `test_benjamini_hochberg_correction` | BH FDR monotonicity and correctness | Fix #7 |
| 11 | `test_correlation_threshold_crossings` | Counts below-0.5/0.3 events correctly | Fix #4 |
| 12 | `test_correlation_slope_accuracy` | Slope matches known linear trend | Fix #5 |
| 13 | `test_correlation_drawdown` | Max DD computed correctly | Fix #5 |
| 14 | `test_ranking_penalizes_breaks` | Frequent-break pairs score lower | Fix #3 |
| 15 | `test_ranking_uses_adjusted_p` | FDR p-values affect rank order | Fix #7 |
| 16 | `test_walk_forward_backtest_oos` | WF backtest doesn't leak future data | Fix #2 |

### 7.B Test infrastructure
- Use `numpy.random.seed(42)` for deterministic synthetic data
- Follow existing pattern from `backend/tests/test_correlation.py`:
  - Patch `yf.download` to return synthetic DataFrames
  - Create helper functions for known data scenarios:
    - `make_cointegrated_pair(length=500, seed=42)` — two series with known cointegration
    - `make_series_with_break(length=500, break_point=250)` — spread with structural break
    - `make_correlated_pair(length=500, target_corr=0.7)` — series with known correlation
- Use `pytest.approx` for floating point tolerance
- Each test is self-contained (no shared mutable state)

---

## Phase 8 — API & Frontend Updates

### 8.A Pydantic model updates
- **File**: `backend/api/models.py`
  - `CorrelationMetrics`:
    ```python
    class CorrelationMetrics(BaseModel):
        window: int
        current: float
        mean: float
        std: float
        drift: float = 0.0                     # kept for backward compat, maps to overall_slope
        regime_changes: int = 0                # kept for backward compat, deprecated
        threshold_crossings: dict[str, int] = {}  # NEW
        correlation_collapse: bool = False        # NEW
        overall_slope: float = 0.0               # NEW
        max_correlation_drawdown: float = 0.0    # NEW
        stability_score: float = 0.0             # NEW
    ```
  - `WalkForwardMetrics`:
    ```python
    class WalkForwardMetrics(BaseModel):
        avg_train_p_value: float = 1.0     # RENAMED from avg_p_value
        avg_oos_p_value: float = 1.0       # NEW
        avg_half_life: float
        cointegration_percentage: float
        avg_spread_sharpe: float
        hedge_ratio_stability: float
        avg_spread_drawdown: float = 0.0
        oos_stationarity_pct: float = 0.0
        regime_stability_pct: float = 0.0
        num_folds: int
    ```
  - `RegimeResultModel`:
    ```python
    class RegimeResultModel(BaseModel):
        current_regime: str = "unknown"
        trading_allowed: bool = True
        signal_suppressed: bool = False
        structural_break: bool = False
        correlation_breakdown: bool = False
        spread_variance_expansion: bool = False
        vix_level: float = 0.0
        regime_summary: dict[str, float] = {}
        cusum_break_detected: bool = False       # NEW
        cusum_break_indices: list[int] = []      # NEW
        chow_break_detected: bool = False        # NEW
        chow_break_dates: list[str] = []         # NEW
        bai_perron_breaks: list[int] = []        # NEW
        num_structural_breaks: int = 0           # NEW
    ```
  - New `WFBacktestMetricsModel`:
    ```python
    class WFBacktestMetricsModel(BaseModel):
        total_return_pct: float = 0.0
        annualised_return_pct: float = 0.0
        sharpe_ratio: float = 0.0
        max_drawdown_pct: float = 0.0
        win_rate_pct: float = 0.0
        num_trades: int = 0
        avg_holding_period: float = 0.0
        turnover: float = 0.0
        num_folds: int = 0
    ```
  - `PairData`:
    ```python
    class PairData(BaseModel):
        ...
        wf_backtest: WFBacktestMetricsModel = WFBacktestMetricsModel()  # NEW
    ```

### 8.B Route updates
- **File**: `backend/api/pairs_routes.py`
  - `_pair_to_response`: map new CorrelationMetrics fields, RegimeResultModel fields, WalkForwardMetrics fields
  - Add WFBacktestMetricsModel mapping from `pair_result.wf_backtest`
  - Handle `Optional` fields gracefully (missing data → empty model defaults)

### 8.C Frontend updates (optional for this session)
- **File**: `frontend/src/pages/Pairs.tsx` — add display for:
  - Structural break count and event markers
  - Correlation stability score, slope, drawdown
  - Walk-forward OOS p-value vs train p-value
  - WF backtest metrics alongside in-sample reference
- **File**: `frontend/src/api/client.ts` — update TypeScript types
  - Update `PairData`, `WalkForwardMetrics`, `RegimeResultModel`, `CorrelationMetrics` interfaces
  - Add `WFBacktestMetrics` interface

---

## Dependencies Between Phases

```
Phase 0 (consolidation)
  ├─→ Phase 1 (WF leakage fix) — imports from centralized hedge_ratio
  ├─→ Phase 2 (WF backtest) — uses Phase 0 imports
  └─→ Phase 3 (breaks) — independent
  │
Phase 1 ──→ Phase 2 (uses fixed WalkForward folds)
  │
Phase 4 (correlation) ── independent, can parallel with Phase 3
  │
Phase 5 (multi comp) ──→ Phase 6 (ranking uses adjusted p-values)
  │
Phase 3 (breaks) ──→ Phase 6 (ranking penalizes breaks)
  │
Phase 6 (validity)
  │
Phase 7 (tests) ── needs everything else done first
  │
Phase 8 (API/frontend) ── needs everything else done first
```

**Recommended implementation order**: 0 → 1 → 2 → (3 ∥ 4 ∥ 5) → 6 → 7 → 8

---

## File Change Summary

| File | Change |
|---|---|
| `backend/pairs_trading.py` | DELETE |
| `backend/stats_arb/hedge_ratio.py` | Add `estimate_ols()` with robust/no-intercept modes |
| `backend/stats_arb/cointegration.py` | Remove inline `_estimate_hedge_ratio`, import from `hedge_ratio` |
| `backend/stats_arb/walk_forward.py` | Fix EG→ADF (Fix #1), add `WalkForwardBacktest` (Fix #2), centralize HR import (Fix #6) |
| `backend/stats_arb/correlation.py` | Threshold crossing, slope, drawdown, stability (Fixes #4 & #5) |
| `backend/stats_arb/regime.py` | CUSUM, Chow, Bai-Perron (Fix #3) |
| `backend/stats_arb/spread.py` | Variance explosion events (Fix #3) |
| `backend/stats_arb/strategy.py` | Expanding window z-score (Fix #8), structural break exit (Fix #3) |
| `backend/stats_arb/backtest.py` | Benchmark comparison (Fix #8) |
| `backend/stats_arb/ranking.py` | FDR correction (Fix #7), break penalty (Fix #3), stability scoring (Fix #4) |
| `backend/stats_arb/pipeline.py` | WF backtest (Fix #2), purity assertions (Fix #8) |
| `backend/stats_arb/visualization.py` | Break plots, HR drift plot, OOS p-value chart (Fixes #3, #6) |
| `backend/stats_arb/config.py` | New config parameters for breaks, correction, costs |
| `backend/stats_arb/__init__.py` | New exports |
| `backend/api/models.py` | Updated/split models for new metrics |
| `backend/api/pairs_routes.py` | Updated response mapping |
| `backend/tests/test_stats_arb.py` | NEW — 16 tests covering all fixes |
| `frontend/src/pages/Pairs.tsx` | UI updates for new metrics |
| `frontend/src/api/client.ts` | Type updates |
