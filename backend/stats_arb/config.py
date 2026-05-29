"""
Global configuration defaults for the statistical arbitrage framework.

All tunable parameters are centralized here to avoid magic numbers
scattered across modules.
"""

from __future__ import annotations

# ── Correlation ───────────────────────────────────
CORRELATION_WINDOWS: list[int] = [20, 60, 120, 252]

# ── Cointegration ─────────────────────────────────
DEFAULT_SIGNIFICANCE: float = 0.05

# ── Trading Strategy ──────────────────────────────
DEFAULT_Z_ENTRY: float = 2.0
DEFAULT_Z_EXIT: float = 0.0
DEFAULT_STOP_LOSS: float = 3.0
DEFAULT_TRANSACTION_COST: float = 0.001       # 10 bps per side
DEFAULT_SLIPPAGE: float = 0.0005              # 5 bps slippage
DEFAULT_BORROW_FEE: float = 0.0003            # 3 bps daily borrow (annualised ~7-8%)
DEFAULT_MAX_HOLDING_DAYS: int = 60

# ── Train / Validation / Test Split ───────────────
DEFAULT_TRAIN_PCT: float = 0.60
DEFAULT_VAL_PCT: float = 0.20
DEFAULT_TEST_PCT: float = 0.20

# ── Walk-Forward Validation ───────────────────────
DEFAULT_WALK_FORWARD_TRAIN: int = 252
DEFAULT_WALK_FORWARD_TEST: int = 63
DEFAULT_EMBARGO_DAYS: int = 21
DEFAULT_MIN_TRAIN_SAMPLES: int = 126
WALK_FORWARD_ADF_MAXLAG: int = 1

# ── Regime Detection ──────────────────────────────
REGIME_VOL_WINDOW: int = 63
VIX_TICKER: str = "^VIX"
CUSUM_CONFIDENCE: float = 0.95
CHOW_SIGNIFICANCE: float = 0.05
BAI_PERRON_MAX_BREAKS: int = 3
BAI_PERRON_MIN_SEGMENT: int = 60

# ── ML ────────────────────────────────────────────
ML_TEST_SIZE: float = 0.20
ML_N_ITER: int = 100

# ── Ranking / Multiple Comparison ─────────────────
MULTIPLE_COMPARISON_METHOD: str = "bh"

# ── Market Impact ─────────────────────────────────
DEFAULT_MARKET_IMPACT: float = 0.001
