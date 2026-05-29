# Pearson Correlation Trading Bot

## What Is Pearson Correlation (and Why It Can Time Trades)

Pearson correlation measures the **linear relationship** between two variables on a scale from -1 to +1.

In trading, you use it to answer: *"When stock A moves, does stock B tend to move with it, against it, or independently?"* — and more importantly, **when that relationship breaks down**, a reversion or divergence trade is possible.

### The Formula

Given two series X and Y of length n:

```
       Σ (xᵢ - x̄)(yᵢ - ȳ)
r = ─────────────────────────────────
    √[Σ(xᵢ - x̄)²] × √[Σ(yᵢ - ȳ)²]
```

Where:
- `xᵢ`, `yᵢ` = individual observations (e.g. daily returns of stock A and B)
- `x̄`, `ȳ`  = means of each series
- `Σ`        = sum over all n observations
- `r`        = Pearson correlation coefficient ∈ [-1, 1]

**Interpretation:**

| r value       | Meaning                                  |
|---------------|------------------------------------------|
| +0.8 to +1.0  | Strong positive — move together          |
| +0.4 to +0.8  | Moderate positive                        |
| -0.2 to +0.4  | Weak / no relationship                   |
| -0.4 to -0.8  | Moderate negative — move opposite        |
| -0.8 to -1.0  | Strong negative — almost perfect inverse |

### Rolling Pearson Correlation

Rather than computing one static correlation over 20 years, you compute it over a **rolling window** (e.g. 20 or 60 trading days) so it updates every bar:

```
r_t = pearson(returns_A[t-N : t], returns_B[t-N : t])
```

This gives you a time series of correlation values — and **shifts in that time series are your signals**.

---

## Core Trading Strategies Using Pearson Correlation

### Strategy 1: Correlation Divergence (Pairs Trading)

**Concept:** Two historically correlated assets (e.g. NVDA and AMD) temporarily diverge in price. The spread should revert.

**Logic:**
1. Compute rolling 60-day Pearson correlation between NVDA returns and AMD returns
2. Compute the **price spread**: `spread = price_A - (β × price_B)`  where β is the hedge ratio (slope from linear regression of A on B)
3. Compute the **z-score** of the spread: `z = (spread - mean_spread) / std_spread`
4. When `z > +2.0`: A is overpriced relative to B → Short A, Long B
5. When `z < -2.0`: A is underpriced relative to B → Long A, Short B
6. When `z crosses 0`: Exit the trade

**Only enter trades when correlation > 0.7** (otherwise the pair isn't behaving as a pair).

### Strategy 2: Correlation Breakdown Signal

**Concept:** When a previously stable correlation suddenly drops, it signals a structural change — one asset is decoupling. The decoupling asset may be about to make a directional move.

**Logic:**
1. Compute short-window (20-day) and long-window (60-day) rolling correlations between your target stock and a benchmark (e.g. SPY)
2. Compute: `corr_delta = corr_20 - corr_60`
3. If `corr_delta < -0.3` (short-term correlation dropped sharply vs long-term): the stock is diverging from the market → potential mean reversion or breakout signal
4. Combine with momentum to determine direction: if the stock is also outperforming SPY over 5 days, it may be breaking out upward

### Strategy 3: Correlation Regime Filter

**Concept:** Use the correlation of your stock to SPY as a market regime filter. High correlation = trending market, low correlation = choppy/idiosyncratic.

**Logic:**
1. Compute 30-day rolling correlation between stock returns and SPY returns
2. If `r > 0.6`: "macro-driven" regime → favor trend-following signals
3. If `r < 0.3`: "stock-specific" regime → favor mean-reversion signals
4. Apply this as a filter on top of any other model (including your existing ML model)

---

## Phase 0 — Data Foundation

**What you need:**
- Daily OHLCV price data for your symbol universe + at least one benchmark (SPY or QQQ)
- At minimum 1 year of history; 3–5 years is better for stable correlation estimates
- Aligned timestamps (same trading days, forward-filled gaps)

**Implementation:**

```python
import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS   = ["NVDA", "AMD", "META", "SPY"]
BENCHMARK = "SPY"
YEARS     = 5

def download_data(symbols, years):
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    raw   = yf.download(symbols, start=start, end=end,
                        auto_adjust=True)["Close"]
    raw.dropna(how="all", inplace=True)
    raw.ffill(inplace=True)          # fill missing trading days
    return raw

prices  = download_data(SYMBOLS, YEARS)
returns = prices.pct_change().dropna()   # daily returns
```

**Key rule:** Always work in **returns** (percent change), not raw prices. Raw prices are non-stationary and will give you spuriously high correlations.

---

## Phase 1 — Computing Rolling Pearson Correlation

### Manual Implementation (so you understand it)

```python
def rolling_pearson(series_a: pd.Series,
                    series_b: pd.Series,
                    window: int) -> pd.Series:
    """
    Compute rolling Pearson r between two return series.
    Formula:
        r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / (σ_x × σ_y × n)
    """
    corr_values = []
    for i in range(len(series_a)):
        if i < window:
            corr_values.append(np.nan)
            continue
        x = series_a.iloc[i - window : i].values
        y = series_b.iloc[i - window : i].values
        x_mean = x.mean()
        y_mean = y.mean()
        numerator   = np.sum((x - x_mean) * (y - y_mean))
        denom_x     = np.sqrt(np.sum((x - x_mean) ** 2))
        denom_y     = np.sqrt(np.sum((y - y_mean) ** 2))
        if denom_x == 0 or denom_y == 0:
            corr_values.append(np.nan)
        else:
            corr_values.append(numerator / (denom_x * denom_y))
    return pd.Series(corr_values, index=series_a.index)
```

### Fast Implementation (use this in production)

```python
# pandas has rolling().corr() built in — same math, vectorized
window_short = 20
window_long  = 60

corr_short = returns["NVDA"].rolling(window_short).corr(returns["AMD"])
corr_long  = returns["NVDA"].rolling(window_long).corr(returns["AMD"])

corr_delta = corr_short - corr_long   # correlation divergence signal
```

**Sanity check your output:**
```python
assert corr_short.between(-1, 1).all() or corr_short.isna().any()
print(corr_short.describe())
# mean should be ~0.5–0.8 for correlated pairs like NVDA/AMD
```

---

## Phase 2 — Pairs Trading (Spread + Z-Score)

### Step 2a: Compute the Hedge Ratio (β)

β tells you how many shares of B to hold per share of A so the portfolio is dollar-neutral.

```python
from sklearn.linear_model import LinearRegression

def compute_hedge_ratio(price_a: pd.Series,
                        price_b: pd.Series,
                        window: int) -> pd.Series:
    """Rolling OLS beta of A regressed on B."""
    betas = []
    for i in range(len(price_a)):
        if i < window:
            betas.append(np.nan)
            continue
        y = price_a.iloc[i - window : i].values.reshape(-1, 1)
        x = price_b.iloc[i - window : i].values.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True).fit(x, y)
        betas.append(float(model.coef_[0]))
    return pd.Series(betas, index=price_a.index)

beta = compute_hedge_ratio(prices["NVDA"], prices["AMD"], window=60)
```

### Step 2b: Compute the Spread and Z-Score

```python
spread = prices["NVDA"] - (beta * prices["AMD"])

# Rolling z-score of the spread
spread_mean = spread.rolling(60).mean()
spread_std  = spread.rolling(60).std()
z_score     = (spread - spread_mean) / spread_std
```

**The math of the z-score:**
```
         spread_t - μ_spread
z_t  =  ─────────────────────
               σ_spread
```

Where μ and σ are computed over the same rolling window as the correlation.

### Step 2c: Generate Entry/Exit Signals

```python
ENTRY_THRESHOLD = 2.0    # z-score magnitude to enter
EXIT_THRESHOLD  = 0.5    # z-score magnitude to exit (near zero)
MIN_CORR        = 0.70   # only trade when pair is actually correlated

def generate_pairs_signals(z: pd.Series,
                           corr: pd.Series) -> pd.DataFrame:
    signals = pd.DataFrame(index=z.index)
    signals["z"]         = z
    signals["corr"]      = corr
    signals["position"]  = 0   # +1 = long spread, -1 = short spread

    pos = 0
    for i in range(len(signals)):
        c = signals["corr"].iloc[i]
        z_val = signals["z"].iloc[i]

        if pd.isna(z_val) or pd.isna(c):
            continue

        if pos == 0 and c >= MIN_CORR:
            if z_val >  ENTRY_THRESHOLD:
                pos = -1   # spread too high: short A, long B
            elif z_val < -ENTRY_THRESHOLD:
                pos =  1   # spread too low: long A, short B

        elif pos != 0:
            if abs(z_val) < EXIT_THRESHOLD:
                pos = 0    # spread reverted, exit

        signals["position"].iloc[i] = pos

    return signals

signals = generate_pairs_signals(z_score, corr_long)
```

---

## Phase 3 — Correlation Regime Filter (Overlaying on Any Strategy)

This phase makes Pearson correlation a **meta-filter** that can wrap your existing ML model or any other strategy.

```python
def compute_regime(stock_returns: pd.Series,
                   benchmark_returns: pd.Series,
                   window: int = 30) -> pd.Series:
    """
    Returns regime label per bar:
      'trend'     → r > 0.6  (macro-driven, use trend signals)
      'reversion' → r < 0.3  (stock-specific, use mean-reversion)
      'neutral'   → otherwise
    """
    corr = stock_returns.rolling(window).corr(benchmark_returns)
    regime = pd.Series("neutral", index=corr.index)
    regime[corr >= 0.6] = "trend"
    regime[corr <= 0.3] = "reversion"
    return regime

regime = compute_regime(returns["NVDA"], returns["SPY"])

# Apply as filter on any signal series
def filter_by_regime(raw_signal: pd.Series,
                     regime: pd.Series,
                     strategy_type: str) -> pd.Series:
    """
    strategy_type: "trend" or "reversion"
    Suppress signals when the market regime doesn't match.
    """
    filtered = raw_signal.copy()
    mismatch = regime != strategy_type
    filtered[mismatch] = 0   # no trade in wrong regime
    return filtered
```

**Example:** If your ML model is a trend-follower, wrap it:
```python
ml_raw_signal    = pd.Series(...)    # your existing ML signal (+1/0/-1)
ml_filtered      = filter_by_regime(ml_raw_signal, regime, "trend")
# ml_filtered will be 0 on choppy/idiosyncratic days,
# even when the ML model wants to trade
```

---

## Phase 4 — Correlation Breakdown Signal (Decoupling)

```python
def correlation_breakdown_signal(stock_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  short_window: int = 20,
                                  long_window: int  = 60,
                                  threshold: float  = -0.30) -> pd.Series:
    """
    Returns +1 when short-term correlation drops sharply vs long-term
    (stock is decoupling from market — potential directional move).
    Returns 0 otherwise.
    """
    corr_short = stock_returns.rolling(short_window).corr(benchmark_returns)
    corr_long  = stock_returns.rolling(long_window).corr(benchmark_returns)
    delta      = corr_short - corr_long

    # Direction: if stock outperforming benchmark over 5 days → bullish break
    rel_perf_5d = (
        (1 + stock_returns).rolling(5).apply(np.prod) -
        (1 + benchmark_returns).rolling(5).apply(np.prod)
    )

    breakdown = pd.Series(0, index=delta.index)
    # Decoupling + stock outperforming → long signal
    breakdown[(delta < threshold) & (rel_perf_5d > 0)] =  1
    # Decoupling + stock underperforming → short signal
    breakdown[(delta < threshold) & (rel_perf_5d < 0)] = -1

    return breakdown

breakdown_signal = correlation_breakdown_signal(
    returns["NVDA"], returns["SPY"]
)
```

---

## Phase 5 — Backtesting the Correlation Strategy

```python
def backtest_pairs(signals: pd.DataFrame,
                   price_a: pd.Series,
                   price_b: pd.Series,
                   initial_cash: float = 10_000.0,
                   cost_per_trade: float = 0.001) -> dict:
    """
    Simple pairs backtest.
    position = +1: long A, short B (β shares)
    position = -1: short A, long B (β shares)
    cost_per_trade = 0.1% slippage per leg (each way)
    """
    cash     = initial_cash
    equity   = []
    position = 0
    entry_price_a = entry_price_b = 0.0
    beta_at_entry = 1.0

    for i in range(len(signals)):
        new_pos = signals["position"].iloc[i]
        pa = price_a.iloc[i]
        pb = price_b.iloc[i]
        b  = signals.get("beta", pd.Series(1.0, index=signals.index)).iloc[i]

        # Exit old position
        if position != 0 and new_pos != position:
            pnl_a = position * (pa - entry_price_a)
            pnl_b = -position * beta_at_entry * (pb - entry_price_b)
            trade_cost = (pa + beta_at_entry * pb) * cost_per_trade * 2
            cash += pnl_a + pnl_b - trade_cost
            position = 0

        # Enter new position
        if position == 0 and new_pos != 0:
            position      = new_pos
            entry_price_a = pa
            entry_price_b = pb
            beta_at_entry = b if not pd.isna(b) else 1.0

        # Mark to market
        if position != 0:
            mtm_a = position * (pa - entry_price_a)
            mtm_b = -position * beta_at_entry * (pb - entry_price_b)
            equity.append(cash + mtm_a + mtm_b)
        else:
            equity.append(cash)

    equity_curve = pd.Series(equity, index=signals.index)
    ret          = equity_curve.pct_change().dropna()
    sharpe       = (ret.mean() / ret.std() * np.sqrt(252)
                    if ret.std() > 0 else 0.0)
    max_dd       = ((equity_curve / equity_curve.cummax()) - 1).min()
    total_return = (equity_curve.iloc[-1] / initial_cash - 1) * 100

    return {
        "equity_curve" : equity_curve,
        "total_return" : round(total_return, 2),
        "sharpe"       : round(sharpe, 2),
        "max_drawdown" : round(max_dd * 100, 2),
        "final_equity" : round(equity_curve.iloc[-1], 2),
    }

results = backtest_pairs(signals, prices["NVDA"], prices["AMD"])
print(f"Return: {results['total_return']}%  |  "
      f"Sharpe: {results['sharpe']}  |  "
      f"Max DD: {results['max_drawdown']}%")
```

---

## Phase 6 — Risk Management Rules

These are non-negotiable guardrails regardless of which correlation strategy you run.

### Rule 1: Minimum Correlation Threshold
Only enter a pairs trade when the rolling correlation is above your threshold. Below 0.7, the pair is not behaving as a pair and spread divergence is not mean-reverting.

```python
MIN_CORRELATION = 0.70
if corr_long.iloc[-1] < MIN_CORRELATION:
    signal = 0  # no trade
```

### Rule 2: Z-Score Hard Stop
If the z-score keeps widening past 3.0 after you've entered at 2.0, the spread is **not** reverting — you're fighting a structural break. Exit.

```python
STOP_ZSCORE = 3.5
if abs(z_score.iloc[-1]) > STOP_ZSCORE and position != 0:
    exit_position()   # forced stop
```

### Rule 3: Max Holding Period
Even if z-score hasn't reverted, exit after N days. Correlation can take months to revert or not at all.

```python
MAX_HOLD_DAYS = 15
if bars_in_trade >= MAX_HOLD_DAYS:
    exit_position()
```

### Rule 4: Correlation Stability Check
Before entering, check that the rolling correlation itself is not highly variable (high std means the pair is unstable):

```python
CORR_STABILITY_WINDOW = 60
corr_std = corr_long.rolling(CORR_STABILITY_WINDOW).std()
if corr_std.iloc[-1] > 0.15:
    signal = 0  # pair is too unstable, skip
```

### Rule 5: Position Sizing Cap
Never put more than 20% of capital in a single pairs trade:

```python
MAX_POSITION_PCT = 0.20
position_size = min(available_cash * MAX_POSITION_PCT,
                    target_position_dollars)
```

---

## Phase 7 — Integration Into Your Existing ML Bot

Since you already have the ML training pipeline described in your plan, you can slot Pearson correlation in as a **feature layer** and a **gate layer**:

### As Features (feed into your 38-indicator set)

```python
# Add to features.py alongside your existing 38 indicators
def add_correlation_features(df: pd.DataFrame,
                              benchmark_df: pd.DataFrame,
                              windows: list = [20, 60]) -> pd.DataFrame:
    stock_ret = df["close"].pct_change()
    bench_ret = benchmark_df["close"].pct_change()

    for w in windows:
        df[f"corr_spy_{w}"]  = stock_ret.rolling(w).corr(bench_ret)

    # Correlation delta (breakdown signal magnitude)
    df["corr_delta_20_60"] = (
        df["corr_spy_20"] - df["corr_spy_60"]
    )

    # Correlation trend (is correlation rising or falling?)
    df["corr_trend"] = df["corr_spy_20"].diff(5)

    return df
```

### As a Gate (filter ML signals by correlation regime)

```python
# In ml_strategy.py, before executing any ML signal:
corr_spy = compute_rolling_corr(stock_returns, spy_returns, window=30)

if corr_spy < 0.3:
    # idiosyncratic regime — only take mean-reversion signals
    if ml_signal == TREND_SIGNAL:
        ml_signal = HOLD

elif corr_spy > 0.7:
    # macro regime — only take trend-following signals
    if ml_signal == REVERSION_SIGNAL:
        ml_signal = HOLD
```

---

## Common Mistakes to Avoid

| Mistake | Why it's a problem | Fix |
|---|---|---|
| Computing correlation on raw prices | Prices are non-stationary, gives spuriously high r | Always use **returns** |
| Using too short a window (< 15 days) | Noisy, will give false signals daily | Minimum 20 days, prefer 60 |
| Entering when correlation is low | Spread isn't mean-reverting below r=0.7 | Gate entry on MIN_CORR |
| No stop on z-score widening | Structural breaks can wipe a pairs trade | Hard stop at z=3.5 |
| Not accounting for transaction costs | Pairs trades are 2-leg, costs double | At least 0.1% per leg |
| Assuming the pair stays correlated forever | Correlations shift (e.g. AMD pivoting to AI) | Re-estimate hedge ratio rolling, not static |
| Data leakage in backtesting | Using future correlation to generate past signals | Always use `.shift(1)` on signals before evaluating |

---

## Full Pipeline Summary

```
Phase 0: Download prices → compute daily returns (never raw prices)
    ↓
Phase 1: Compute rolling Pearson r (short + long window)
    ↓
Phase 2: Compute spread + z-score (for pairs trading)
    ↓
Phase 3: Label each bar as trend/reversion/neutral regime
    ↓
Phase 4: Generate breakdown signal (decoupling detection)
    ↓
Phase 5: Backtest with transaction costs + compare vs buy-and-hold
    ↓
Phase 6: Apply all 5 risk rules before any live trade
    ↓
Phase 7: Integrate as features + regime gate in your ML bot
```