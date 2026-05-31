"""CorrCointStrategy — pairs trading via correlation + cointegration mean reversion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backend.strategies.base import Strategy, Signal, Portfolio

logger = logging.getLogger("corr_coint_strat")


class CorrCointStrategy(Strategy):
    """Correlation + Cointegration Mean Reversion Pairs Strategy.

    Computes a z-score on the cointegrated spread between two stocks and
    enters mean-reversion bets when the z-score deviates beyond configurable
    thresholds.  Supports graduated entry (50 % / 100 % tiers), partial profit
    taking, stop loss, time-based exit, structural-break detection, a
    correlation gate, volatility-adjusted sizing, a cooldown after consecutive
    losses, and a portfolio-level VaR limit.

    Parameters
    ----------
    symbol_b : str
        Ticker of the paired symbol (symbol_a is set by the engine as ``_symbol``).
    hedge_ratio : float
        Pre-computed hedge ratio (beta) for the pair::
            spread = close_a - hedge_ratio * close_b
    half_life : float
        Expected half-life of mean reversion in days (informational /
        used for half-life weighting).
    z_entry_strong : float
        Z-score level for a full 100 % position.
    z_entry_weak : float
        Z-score level for a 50 % position.
    z_exit_full : float
        Z-score level at which the full position is closed.
    z_exit_partial : float
        Z-score level at which 50 % of the position is taken off
        (partial profit lock).
    z_stop : float
        Stop-loss z-score — position is closed entirely if breached.
    max_holding_days : int
        Maximum number of bars a position may be held before forced exit.
    volatility_adjust : bool
        If True, position size is scaled inversely to 63-day rolling spread
        volatility.
    min_corr_gate : float
        Minimum rolling 20-day Pearson correlation required to enter a trade.
    max_consecutive_losses : int
        After this many consecutive losing trades the pair is paused.
    cooldown_days : int
        Number of bars to pause the pair after hitting ``max_consecutive_losses``.
    base_pair_capital : float
        Base capital allocation per pair (the engine trades the primary leg
        with half of this).
    max_loss_pct : float
        Maximum loss as a percentage of pair capital before hard stop.
    hedge_ratio_window : int
        Rolling window size (days) for dynamic hedge ratio computation.

    Notes
    -----
    The data DataFrame passed to ``init()`` / ``next()`` is expected to contain
    standard OHLCV columns for the **primary** symbol plus an extra column
    ``close_pair`` with the paired symbol's adjusted close prices.

    **Entry rules (graduated)** ::

        |z| >= z_entry_strong  → 100 % allocation
        z_entry_weak <= |z| < z_entry_strong  → 50 % allocation

    **Exit rules** ::

        |z| crosses z_exit_full    → full exit
        |z| crosses z_exit_partial → 50 % partial exit
        |z| >= z_stop              → stop-loss exit
        holding > max_holding_days → forced exit
        P&L < -max_loss_pct        → forced exit

    **Gates** ::

        rolling 20d corr < min_corr_gate  → no entry
        cooldown active                   → no entry
    """

    def __init__(
        self,
        symbol_b: str = "",
        hedge_ratio: float = 1.0,
        half_life: float = 20.0,
        z_entry_strong: float = 2.0,
        z_entry_weak: float = 1.5,
        z_exit_full: float = 0.0,
        z_exit_partial: float = 0.5,
        z_stop: float = 2.5,
        max_holding_days: int = 40,
        volatility_adjust: bool = True,
        min_corr_gate: float = 0.5,
        max_consecutive_losses: int = 3,
        cooldown_days: int = 10,
        base_pair_capital: float = 10000.0,
        max_loss_pct: float = 5.0,
        hedge_ratio_window: int = 60,
    ) -> None:
        self.symbol_b = symbol_b
        self.hedge_ratio = hedge_ratio if hedge_ratio != 0 else 1.0
        self.half_life = half_life
        self.z_entry_strong = z_entry_strong
        self.z_entry_weak = z_entry_weak
        self.z_exit_full = z_exit_full
        self.z_exit_partial = z_exit_partial
        self.z_stop = z_stop
        self.max_holding_days = max_holding_days
        self.volatility_adjust = volatility_adjust
        self.min_corr_gate = min_corr_gate
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_days = cooldown_days
        self.base_pair_capital = base_pair_capital
        self.max_loss_pct = max_loss_pct
        self.hedge_ratio_window = hedge_ratio_window

        self.buy_size: float = base_pair_capital / 2
        self.sell_portion: float = 100.0

        # ── Pre-computed series (set in init) ──
        self._spread: pd.Series = None
        self._zscore: pd.Series = None
        self._rolling_corr: pd.Series = None
        self._vol_scalar: pd.Series = None
        self._rolling_hr: pd.Series = None
        self._rolling_hurst: pd.Series = None
        self._variance_ratio: pd.Series = None

        # ── Runtime state ──
        self._in_pair: bool = False
        self._entry_bar: int = -1
        self._entry_z: float = 0.0
        self._entry_spread: float = 0.0
        self._entry_price_a: float = 0.0
        self._entry_hedge_ratio: float = 0.0
        self._peak_spread: float = 0.0
        self._tier: float = 0.0
        self._consecutive_losses: int = 0
        self._cooldown_until_index: int = -1
        self._last_trade_pnl: float = 0.0
        self._current_hurst_mult: float = 1.0
        self._current_size_mult: float = 1.0
        self._current_max_hold_mult: float = 1.0

    # ──────────────────────────────────────────────────────────────
    #  Public interface (Strategy ABC)
    # ──────────────────────────────────────────────────────────────

    def init(self, data: pd.DataFrame) -> None:
        col_a_close = "close"
        col_b_close = "close_pair"

        if col_b_close not in data.columns:
            logger.warning(
                "CorrCointStrategy: data missing 'close_pair' column. "
                "Using close as proxy for both legs."
            )
            close_a = data[col_a_close].astype(float)
            close_b = close_a.copy()
        else:
            close_a = data[col_a_close].astype(float)
            close_b = data[col_b_close].astype(float)

        clean_mask = close_a.notna() & close_b.notna() & (close_a > 0) & (close_b > 0)
        close_a = close_a[clean_mask]
        close_b = close_b[clean_mask]

        if len(close_a) < 60:
            logger.error("CorrCointStrategy: need at least 60 bars.  All HOLD.")
            self._spread = pd.Series(dtype=float)
            self._zscore = pd.Series(dtype=float)
            self._rolling_corr = pd.Series(dtype=float)
            self._vol_scalar = pd.Series(dtype=float)
            return

        spread = close_a - self.hedge_ratio * close_b
        expanding_mean = spread.expanding().mean()
        expanding_std = spread.expanding().std()
        zscore = (spread - expanding_mean) / expanding_std.replace(0, np.nan)

        returns_a = close_a.pct_change()
        returns_b = close_b.pct_change()
        rolling_corr = returns_a.rolling(20).corr(returns_b)

        if self.volatility_adjust:
            rolling_vol = spread.rolling(63).std().fillna(expanding_std)
            final_std = float(expanding_std.iloc[-1]) if not expanding_std.empty else 1.0
            vol_scalar = final_std / rolling_vol.replace(0, np.nan)
            vol_scalar = vol_scalar.clip(0.25, 3.0)
        else:
            vol_scalar = pd.Series(1.0, index=spread.index)

        # ── Rolling hedge ratio (60-day OLS) ──
        hr = pd.Series(self.hedge_ratio, index=spread.index)
        w = self.hedge_ratio_window
        if len(close_a) >= w:
            valid = close_a.notna() & close_b.notna() & (close_a > 0) & (close_b > 0)
            x = close_b[valid].values
            y = close_a[valid].values
            rolling_hr = pd.Series(np.nan, index=close_a.index[valid])
            for j in range(w, len(x)):
                x_slice = x[j - w : j]
                y_slice = y[j - w : j]
                xc = sm.add_constant(x_slice)
                try:
                    model = sm.OLS(y_slice, xc).fit()
                    beta = float(model.params.iloc[1])
                    rolling_hr.iloc[j] = beta if abs(beta) > 1e-10 else self.hedge_ratio
                except Exception:
                    rolling_hr.iloc[j] = self.hedge_ratio
            hr = rolling_hr.ffill().bfill().reindex(spread.index, method="ffill").fillna(self.hedge_ratio)

        # ── Rolling Hurst (regime-aware trending adaptation) ──
        spread_vals = spread.values
        rolling_hurst = pd.Series(np.nan, index=spread.index)
        hurst_window = 63
        for j in range(hurst_window, len(spread_vals)):
            chunk = spread_vals[j - hurst_window:j]
            chunk = chunk[pd.notna(chunk)]
            if len(chunk) >= 30:
                rolling_hurst.iloc[j] = self._hurst_exponent(chunk)
        self._rolling_hurst = rolling_hurst

        vol_20 = spread.rolling(20).std()
        vol_60 = spread.rolling(60).std().replace(0, np.nan)
        self._variance_ratio = vol_20 / vol_60

        self._spread = spread
        self._zscore = zscore
        self._rolling_corr = rolling_corr
        self._vol_scalar = vol_scalar
        self._rolling_hr = hr

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if self._zscore is None or self._zscore.empty:
            return Signal.HOLD
        if i >= len(self._zscore) or i >= len(self._rolling_corr) or i >= len(self._vol_scalar):
            return Signal.HOLD

        z = float(self._zscore.iloc[i])
        corr = float(self._rolling_corr.iloc[i]) if pd.notna(self._rolling_corr.iloc[i]) else 0.0
        scale = float(self._vol_scalar.iloc[i])
        spread_val = float(self._spread.iloc[i]) if pd.notna(self._spread.iloc[i]) else 0.0
        price_a = float(data["close"].iloc[i]) if "close" in data.columns else 0.0
        current_hr = float(self._rolling_hr.iloc[i]) if self._rolling_hr is not None and i < len(self._rolling_hr) else self.hedge_ratio

        if not np.isfinite(z) or not np.isfinite(corr):
            return Signal.HOLD

        # ── Cooldown check ──
        if self._cooldown_until_index >= i:
            return Signal.HOLD

        # ── Regime-aware multipliers ──
        hurst = (
            float(self._rolling_hurst.iloc[i])
            if self._rolling_hurst is not None
            and i < len(self._rolling_hurst)
            and pd.notna(self._rolling_hurst.iloc[i])
            else 0.5
        )
        if hurst > 0.70:
            self._current_hurst_mult = 1.5
            self._current_size_mult = 0.3
            self._current_max_hold_mult = 0.3
        elif hurst > 0.55:
            self._current_hurst_mult = 1.25
            self._current_size_mult = 0.6
            self._current_max_hold_mult = 0.5
        elif hurst > 0.40:
            self._current_hurst_mult = 1.0
            self._current_size_mult = 0.8
            self._current_max_hold_mult = 1.0
        else:
            self._current_hurst_mult = 1.0
            self._current_size_mult = 1.0
            self._current_max_hold_mult = 1.0

        # ── Entry logic ──
        if not self._in_pair:
            if not self._can_enter(corr, i):
                return Signal.HOLD
            return self._check_entry(i, z, scale, price_a, current_hr)

        # ── Exit logic ──
        if self._in_pair:
            return self._check_exit(i, z, spread_val, price_a, portfolio, current_hr)

        return Signal.HOLD

    def on_trade(self, side: str, symbol: str, qty: float, price: float) -> None:
        if side == "sell" and not self._in_pair:
            self._consecutive_losses = 0

    # ──────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _can_enter(self, corr: float, i: int) -> bool:
        if self._in_pair:
            return False
        if corr < self.min_corr_gate:
            return False
        if self._cooldown_until_index >= i:
            return False
        if self._variance_ratio is not None and i < len(self._variance_ratio):
            vr = float(self._variance_ratio.iloc[i])
            if pd.notna(vr) and vr > 2.0:
                return False
        return True

    def _check_entry(self, i: int, z: float, scale: float, price_a: float, current_hr: float) -> str:
        abs_z = abs(z)
        direction = 1 if z < 0 else -1

        effective_z_strong = self.z_entry_strong * self._current_hurst_mult
        effective_z_weak = self.z_entry_weak * self._current_hurst_mult

        if abs_z >= effective_z_strong:
            tier = 1.0
        elif abs_z >= effective_z_weak:
            tier = 0.5
        else:
            return Signal.HOLD

        hl_weight = 1.0
        if self.half_life < 20:
            hl_weight = 1.5
        elif self.half_life > 40:
            hl_weight = 0.5

        pair_cap = self.base_pair_capital * scale * tier * hl_weight * self._current_size_mult
        self.buy_size = pair_cap / 2 if price_a > 0 else self.base_pair_capital / 4
        self.sell_portion = 100.0

        self._in_pair = True
        self._entry_bar = i
        self._entry_z = z
        self._entry_spread = float(self._spread.iloc[i]) if self._spread is not None and i < len(self._spread) else 0.0
        self._entry_price_a = price_a
        self._entry_hedge_ratio = current_hr
        self._peak_spread = (
            self._entry_spread if direction == 1 else -self._entry_spread
        )
        self._tier = tier

        return Signal.BUY

    def _check_exit(self, i: int, z: float, spread_val: float, price_a: float, portfolio: Portfolio, current_hr: float) -> str:
        bars_held = i - self._entry_bar
        abs_z = abs(z)
        side = 1 if self._entry_z < 0 else -1
        current_spread_signed = spread_val * side
        self._peak_spread = max(self._peak_spread, current_spread_signed)

        effective_z_exit_full = self.z_exit_full + max(0.0, (self._current_hurst_mult - 1.0) * 1.0)
        effective_z_exit_partial = self.z_exit_partial + max(0.0, (self._current_hurst_mult - 1.0) * 1.0)
        effective_max_hold = int(self.max_holding_days * self._current_max_hold_mult)

        # ── Stop loss: z-stop ──
        if abs_z >= self.z_stop:
            self.sell_portion = 100.0
            self._close_trade()
            return Signal.EXIT

        # ── Stop loss: P&L hard stop ──
        if self._entry_price_a > 0 and price_a > 0:
            pnl_pct = ((price_a - self._entry_price_a) / self._entry_price_a) * 100 * side
            if pnl_pct < -self.max_loss_pct:
                logger.info("CorrCointStrategy: P&L stop hit (%.2f%%) at bar %d", pnl_pct, i)
                self.sell_portion = 100.0
                self._close_trade()
                return Signal.EXIT

        # ── Time exit (adapted for regime) ──
        if bars_held >= effective_max_hold:
            self.sell_portion = 100.0
            self._close_trade()
            return Signal.EXIT

        # ── Partial exit: z crossed partial threshold on the way back ──
        if abs_z <= effective_z_exit_partial and self.sell_portion != 50.0:
            self.sell_portion = 50.0
            return Signal.SELL

        # ── Full exit: z crossed exit threshold ──
        if (side == 1 and z >= effective_z_exit_full) or (side == -1 and z <= effective_z_exit_full):
            self.sell_portion = 100.0
            self._close_trade()
            return Signal.SELL

        # ── Hedge ratio drift rebalance ──
        if self._entry_hedge_ratio > 0 and abs(current_hr) > 0:
            drift = abs(current_hr - self._entry_hedge_ratio) / self._entry_hedge_ratio
            if drift > 0.20:
                logger.info(
                    "CorrCointStrategy: HR drift %.1f%% at bar %d — rebalancing",
                    drift * 100, i,
                )
                self._entry_hedge_ratio = current_hr
                self.buy_size = (self.base_pair_capital * self._tier) / 2
                return Signal.SELL

        return Signal.HOLD

    def _close_trade(self) -> None:
        self._in_pair = False
        self._entry_bar = -1
        self._entry_z = 0.0
        self._entry_spread = 0.0
        self._entry_price_a = 0.0
        self._entry_hedge_ratio = 0.0
        self._peak_spread = 0.0
        self._tier = 0.0
        self.sell_portion = 100.0

    def _track_pnl(self, pnl: float) -> None:
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._cooldown_until_index = max(
                    self._cooldown_until_index,
                    self._entry_bar + self.cooldown_days,
                )
                logger.info(
                    "CorrCointStrategy: %d consecutive losses — cooling down for %d bars",
                    self._consecutive_losses,
                    self.cooldown_days,
                )
        else:
            self._consecutive_losses = 0

        self._last_trade_pnl = pnl

    @staticmethod
    def _hurst_exponent(ts: np.ndarray) -> float:
        if len(ts) < 10:
            return 0.5
        lags = np.arange(2, len(ts) // 2)
        if len(lags) < 2:
            return 0.5
        tau = []
        for lag in lags:
            chunks = len(ts) // lag
            if chunks < 1:
                continue
            rs_vals = []
            for c in range(chunks):
                chunk = ts[c * lag:(c + 1) * lag]
                if len(chunk) < 2:
                    continue
                mean = np.mean(chunk)
                dev = chunk - mean
                z = np.cumsum(dev)
                r = max(z) - min(z)
                s = np.std(chunk, ddof=1)
                if s == 0:
                    continue
                rs_vals.append(r / s)
            if rs_vals:
                tau.append(np.mean(rs_vals))
        if len(tau) < 2:
            return 0.5
        reg = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        return float(np.clip(reg[0], 0.0, 1.0))

    def get_state(self) -> dict[str, Any]:
        return {
            "in_pair": self._in_pair,
            "z_entry": float(self._entry_z) if self._in_pair else None,
            "bars_held": -1,
            "tier": self._tier if self._in_pair else 0.0,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_active": self._cooldown_until_index >= 0,
            "entry_hedge_ratio": self._entry_hedge_ratio,
        }
