# Statistical Arbitrage — Pairs Trading Framework

17-module research framework for pairs trading: cointegration testing, spread modeling, regime detection, walk-forward validation, backtesting, and ranking.

## Pipeline Order

Data → Correlation → Cointegration (EG + Johansen) → Hedge Ratio (OLS/rolling/Kalman) → Spread (half-life, Hurst, ADF, OU) → Regime (Hurst/VIX/CUSUM/Chow/Bai-Perron) → Walk-Forward (purged) → Strategy (z-score mean reversion) → Backtest → Optional ML

## CLI Usage

### Single pair analysis
```bash
.venv/bin/python -m backend.stats_arb.cli --pair KO,PEP --start 2010-01-01
.venv/bin/python -m backend.stats_arb.cli --pair NVDA,META --ml
.venv/bin/python -m backend.stats_arb.cli --pair XOM,CVX --purged
.venv/bin/python -m backend.stats_arb.cli --pair JPM,GS --json
```

### Heatmap
```bash
.venv/bin/python -m backend.stats_arb.cli --heatmap NVDA,AMD,INTC,AAPL,MSFT,GOOGL
```

### Auto-discovery
```bash
.venv/bin/python -m backend.stats_arb.cli --auto-discover
.venv/bin/python -m backend.stats_arb.cli --auto-discover nasdaq100
.venv/bin/python -m backend.stats_arb.cli --auto-discover NVDA,AMD,KO,PEP,JPM,GS,MSFT \
    --min-sharpe 1.5 --require-mean-reverting --output-md pairs_report.md
```

### Rank from file
```bash
.venv/bin/python -m backend.stats_arb.cli --pairs-file pairs.json --rank --top-n 10
```

## `find_pairs` — 3-Phase Discovery Pipeline

Phase 1: Bulk download → pairwise correlation filter (default min 0.5)
Phase 2: EG cointegration test on correlated pairs
Phase 3: Full pipeline on top N candidates → ranked Markdown report

```bash
.venv/bin/python -m backend.find_pairs
.venv/bin/python -m backend.find_pairs nasdaq100
.venv/bin/python -m backend.find_pairs NVDA,AMD,KO,PEP
.venv/bin/python -m backend.find_pairs sp500 --min-sharpe 1.5 --min-corr 0.7 -o best.md
```

| Argument | Default | Description |
|---|---|---|
| `universe` | `sp500` | `sp500`, `nasdaq100`, `dow30`, or tickers |
| `-o` | `pairs_report_<timestamp>.md` | Output file |
| `--min-corr` | `0.5` | Min Pearson correlation |
| `--min-sharpe` | `0.0` | Min Sharpe ratio |
| `--min-return` | `-1000.0` | Min return % |
| `--max-drawdown` | `-100.0` | Max drawdown % |
| `--top-candidates` | `50` | Candidates for full analysis |
| `--require-mean-reverting` | — | Mean-reverting regime only |
| `--require-no-breaks` | — | Exclude structural breaks |
| `--start` | `2015-01-01` | Start date |
| `--capital` | `100000` | Initial capital |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/pairs/analyze` | Full pair analysis |
| POST | `/api/pairs/rank` | Rank multiple pairs |
| POST | `/api/pairs/heatmap` | Cointegration heatmap |

## Module Contents

| File | Purpose |
|------|---------|
| `config.py` | Global defaults (significance, windows, fees) |
| `data.py` | yfinance download + caching |
| `correlation.py` | Rolling correlation, stability scoring |
| `cointegration.py` | Engle-Granger + Johansen tests |
| `hedge_ratio.py` | OLS, rolling OLS, Kalman filter |
| `spread.py` | Half-life, Hurst, ADF, OU process |
| `regime.py` | Hurst, VIX, CUSUM, Chow, Bai-Perron |
| `walk_forward.py` | Purged walk-forward validation |
| `strategy.py` | Z-score mean reversion strategy |
| `backtest.py` | P&L, equity curve, Sharpe, drawdown |
| `ranking.py` | Multiple comparison correction |
| `ml_models.py` | RF/GBT/XGB for spread prediction |
| `pipeline.py` | Full pipeline orchestrator |
| `cli.py` | CLI entry point |
| `discover.py` | Auto-discovery engine |
| `visualization.py` | matplotlib/seaborn plots |
