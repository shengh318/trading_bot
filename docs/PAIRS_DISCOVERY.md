# Pairs Discovery — Automated Pipeline

Two entry points for finding cointegrated stock pairs:

- **`backend.find_pairs`** — 3-phase pipeline for batch discovery
- **`backend.stats_arb.cli --auto-discover`** — Full research pipeline on top candidates

## `find_pairs` — 3-Phase Pipeline

Phase 1: Download all tickers, compute pairwise Pearson correlation,
         keep pairs above `--min-corr` (default 0.5).
Phase 2: Test filtered pairs for Engle-Granger cointegration (p < 0.05).
Phase 3: Run full pairs analysis (spread, regime, walk-forward,
         backtest, ranking) on top N candidates, output ranked report.

### Usage

```bash
# S&P 500 (default)
.venv/bin/python -m backend.find_pairs

# NASDAQ-100 / Dow 30
.venv/bin/python -m backend.find_pairs nasdaq100
.venv/bin/python -m backend.find_pairs dow30

# Custom universe
.venv/bin/python -m backend.find_pairs NVDA,AMD,KO,PEP,JPM,GS,MSFT

# Strict filters
.venv/bin/python -m backend.find_pairs sp500 --min-sharpe 1.5 --min-return 10 \
    --require-mean-reverting --output best_pairs.md

# Lower correlation threshold for more candidates
.venv/bin/python -m backend.find_pairs nasdaq100 --min-corr 0.3 --top-candidates 50
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `universe` | `sp500` | `sp500`, `nasdaq100`, `dow30`, or comma-separated tickers |
| `-o` / `--output` | `pairs_report_<timestamp>.md` | Output path |
| `--min-corr` | `0.5` | Min Pearson correlation filter |
| `--min-sharpe` | `0.0` | Min Sharpe ratio filter |
| `--min-return` | `-1000.0` | Min total return % |
| `--max-drawdown` | `-100.0` | Max drawdown % |
| `--top-candidates` | `50` | Cointegrated pairs to fully analyze |
| `--require-mean-reverting` | — | Only mean-reverting regime pairs |
| `--require-no-breaks` | — | Exclude structural break pairs |
| `--start` | `2015-01-01` | Start date |
| `--capital` | `100000` | Initial capital |
| `--json` | — | Also print JSON to stdout |

## `auto-discover` — Full Research Pipeline

Runs the complete stats-arb pipeline on each candidate pair:
correlation → cointegration → hedge ratio → spread → regime →
walk-forward → strategy → backtest → ranking.

```bash
# From CLI
.venv/bin/python -m backend.stats_arb.cli --auto-discover

# NASDAQ-100 with Markdown report
.venv/bin/python -m backend.stats_arb.cli --auto-discover nasdaq100 \
    --output-md nasdaq_pairs.md --min-sharpe 1.0

# Custom universe, no plots
.venv/bin/python -m backend.stats_arb.cli \
    --auto-discover NVDA,AMD,INTC,AAPL,MSFT,GOOGL \
    --no-plots --no-parallel --output-md discover.md
```

## Pair Analysis (Single Pair)

```bash
.venv/bin/python -m backend.stats_arb.cli --pair KO,PEP --start 2010-01-01
.venv/bin/python -m backend.stats_arb.cli --pair NVDA,META --ml   # ML-enhanced
.venv/bin/python -m backend.stats_arb.cli --pair XOM,CVX --purged # Walk-forward
.venv/bin/python -m backend.stats_arb.cli --pair JPM,GS --json    # JSON output
```

## Cointegration Heatmap

```bash
.venv/bin/python -m backend.stats_arb.cli --heatmap NVDA,AMD,INTC,AAPL,MSFT,GOOGL
```

## Pair Ranking from File

```bash
# pairs.json: [{"a": "KO", "b": "PEP"}, ...]
.venv/bin/python -m backend.stats_arb.cli --pairs-file pairs.json --rank --top-n 10
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/pairs/analyze` | Run full analysis on a pair |
| POST | `/api/pairs/rank` | Rank multiple pairs |
| POST | `/api/pairs/heatmap` | Cointegration heatmap |
