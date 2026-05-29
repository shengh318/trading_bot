from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

from backend.ml.features import compute_features, get_feature_columns, merge_context_features
from backend.ml.model import load_model
from backend.strategies.base import Strategy, Signal, Portfolio

_UNSET = object()


class MLStrategy(Strategy):
    def __init__(
        self,
        model_name: str = "multi_symbol_model",
        confidence_threshold: float | object = _UNSET,
        base_buy_size: float = 1000.0,
        use_kelly: bool | object = _UNSET,
        max_hold_bars: int | object = _UNSET,
        trailing_stop_pct: float | object = _UNSET,
        context_symbols: list[str] | None = None,
        online_learning: bool = False,
        online_learning_every_n: int = 5,
    ):
        self._explicitly_set: set[str] = set()

        self.model_name = model_name
        if confidence_threshold is not _UNSET:
            self.confidence_threshold = float(confidence_threshold)
            self._explicitly_set.add("confidence_threshold")
        else:
            self.confidence_threshold = 0.55
        if use_kelly is not _UNSET:
            self.use_kelly = bool(use_kelly)
            self._explicitly_set.add("use_kelly")
        else:
            self.use_kelly = False
        if max_hold_bars is not _UNSET:
            self.max_hold_bars = int(max_hold_bars)
            self._explicitly_set.add("max_hold_bars")
        else:
            self.max_hold_bars = 30
        if trailing_stop_pct is not _UNSET:
            self.trailing_stop_pct = float(trailing_stop_pct)
            self._explicitly_set.add("trailing_stop_pct")
        else:
            self.trailing_stop_pct = 0.05

        self.base_buy_size = base_buy_size
        self.buy_size = base_buy_size
        self.sell_portion = 100.0
        self.context_symbols = context_symbols or []
        self.online_learning = online_learning
        self.online_learning_every_n = online_learning_every_n
        self._forecast_horizon = 1
        self._online_update_count = 0
        self._online_burn_in = 20
        self._online_rolling_window = 50
        self._online_accuracy: deque = deque(maxlen=self._online_rolling_window)
        self._online_accuracy_threshold = 0.45
        self._kelly_wins = 0.0
        self._kelly_losses = 0.0
        self._kelly_win_pct = 0.0
        self._kelly_loss_pct = 0.0
        self._kelly_count = 0
        self._entry_bar = -1
        self._peak_price = 0.0
        self._has_position = False

    def _kelly_fraction(self, prob_up: float) -> float:
        if self._kelly_count < 3:
            return 0.5 * (2 * prob_up - 1)
        avg_win = self._kelly_win_pct / max(self._kelly_wins, 1)
        avg_loss = self._kelly_loss_pct / max(self._kelly_losses, 1)
        b = avg_win / max(avg_loss, 0.001)
        if b <= 0:
            return 0.0
        k = max(b * prob_up - (1 - prob_up), 0.0) / b
        return float(np.clip(k, 0.0, 0.25))

    def _position_size(self, prob_up: float, portfolio: Portfolio) -> float:
        if self.use_kelly:
            fraction = self._kelly_fraction(prob_up)
            return portfolio.cash * fraction
        if prob_up >= 0.85:
            return self.base_buy_size * 2.0
        if prob_up >= 0.75:
            return self.base_buy_size * 1.5
        if prob_up >= 0.65:
            return self.base_buy_size * 1.0
        return self.base_buy_size * 0.5

    def init(self, data: pd.DataFrame) -> None:
        df = compute_features(data.copy())

        try:
            self.model, metadata = load_model(self.model_name)
            self.feature_columns = metadata.feature_columns
            self._model_loaded = True
        except FileNotFoundError:
            print(
                f"WARNING [MLStrategy]: Model '{self.model_name}' not found. "
                f"Train one first: python -m backend.ml.train"
            )
            self.feature_columns = []
            self._model_loaded = False
            df = df.reset_index(drop=True)
            self._features_df = df
            return

        self._apply_metadata_params(metadata)

        # ── Cross-symbol features (C3) ──
        ctx_symbols = metadata.context_symbols if metadata.context_symbols else self.context_symbols
        if ctx_symbols and len(data) > 0:
            start_date = data.index.min()
            end_date = data.index.max()
            context_data: dict[str, pd.DataFrame] = {}
            for ctx_sym in ctx_symbols:
                try:
                    ticker = yf.Ticker(ctx_sym)
                    ctx_df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                    if not ctx_df.empty:
                        col_map = {
                            "Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume",
                        }
                        ctx_df = ctx_df.rename(columns=col_map)
                        ctx_df = ctx_df[list(col_map.values())]
                        context_data[ctx_sym] = ctx_df
                except Exception:
                    pass
            if context_data:
                df = merge_context_features(df, context_data)

        df = df.reset_index(drop=True)
        self._features_df = df

    def _apply_metadata_params(self, metadata) -> None:
        """Override strategy defaults with saved training params from metadata,
        but only for params not explicitly set by the user."""
        train_params = metadata.params
        if not isinstance(train_params, dict):
            return

        if "confidence_threshold" not in self._explicitly_set:
            tp = train_params.get("confidence_threshold")
            if tp is not None:
                self.confidence_threshold = float(tp)
        if "use_kelly" not in self._explicitly_set:
            kelly = train_params.get("kelly")
            if kelly is not None:
                self.use_kelly = bool(kelly)
        if "max_hold_bars" not in self._explicitly_set:
            mhb = train_params.get("max_hold_bars")
            if mhb is not None:
                self.max_hold_bars = int(mhb)
        if "trailing_stop_pct" not in self._explicitly_set:
            tsp = train_params.get("trailing_stop_pct")
            if tsp is not None:
                self.trailing_stop_pct = float(tsp)

        fh = train_params.get("forecast_horizon")
        if fh is not None:
            self._forecast_horizon = int(fh)

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if not self._model_loaded or self._features_df is None:
            return Signal.HOLD

        if i >= len(self._features_df) or len(data) > len(self._features_df):
            self._features_df = compute_features(data.copy()).reset_index(drop=True)

        if i >= len(self._features_df):
            return Signal.HOLD

        # ── Online learning: rate-limited, drift-aware partial_fit ──
        horizon = self._forecast_horizon
        if self.online_learning and i >= horizon and hasattr(self.model, "partial_fit"):
            lookback_idx = i - horizon
            prev_row = self._features_df.iloc[lookback_idx]
            try:
                prev_feat = prev_row[self.feature_columns].values.reshape(1, -1)
                if np.any(np.isnan(prev_feat)) or np.any(np.isinf(prev_feat)):
                    pass  # skip — feature integrity check failed
                elif self._online_update_count < self._online_burn_in:
                    # ── Burn-in: always update until threshold reached ──
                    self._do_partial_fit(prev_feat, data, i, lookback_idx)
                elif i % self.online_learning_every_n == 0:
                    # ── Rate-limit check passed ──
                    # Check rolling accuracy for drift
                    if (len(self._online_accuracy) == self._online_rolling_window
                            and sum(self._online_accuracy) / self._online_rolling_window
                            < self._online_accuracy_threshold):
                        pass  # drift detected — pause updates
                    else:
                        self._do_partial_fit(prev_feat, data, i, lookback_idx)
            except Exception:
                pass

        return self._evaluate_signal(i, data, portfolio)

    def _do_partial_fit(self, prev_feat, data, i, lookback_idx):
        """Update the model on one bar and track rolling accuracy."""
        actual_target = int(
            data.iloc[i]["close"] > data.iloc[lookback_idx]["close"]
        )
        classes = getattr(self.model, "classes_", np.array([0, 1]))

        # Check if model's pre-update prediction agreed with reality (drift signal)
        prev_proba = self.model.predict_proba(prev_feat)[0]
        prev_pred = 1 if prev_proba[1] >= 0.5 else 0
        self._online_accuracy.append(1 if prev_pred == actual_target else 0)

        self.model.partial_fit(prev_feat, [actual_target], classes=classes)
        self._online_update_count += 1

    def _evaluate_signal(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        """Run model inference and generate a trading signal."""
        row = self._features_df.iloc[i]
        try:
            features = row[self.feature_columns].values.reshape(1, -1)
        except (KeyError, ValueError):
            return Signal.HOLD

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return Signal.HOLD

        proba = self.model.predict_proba(features)[0]
        if len(proba) < 2:
            return Signal.HOLD
        prob_up = proba[1]

        symbol = getattr(self, "_symbol", "ASSET")
        has_position = self._has_position or portfolio.positions.get(symbol, 0) > 0
        price = float(data.iloc[i]["close"])

        # --- Exit checks (override model signal) ---
        if has_position:
            self._peak_price = max(self._peak_price, price)

            # Time-based exit
            if self.max_hold_bars > 0 and self._entry_bar >= 0 and i - self._entry_bar >= self.max_hold_bars:
                return self._exit_position(symbol, portfolio, price)

            # Trailing stop
            if self.trailing_stop_pct > 0 and self._peak_price > 0:
                dd = (price - self._peak_price) / self._peak_price
                if dd <= -self.trailing_stop_pct:
                    return self._exit_position(symbol, portfolio, price)

        # --- Model signals ---
        if prob_up >= self.confidence_threshold:
            self.buy_size = self._position_size(prob_up, portfolio)
            if not has_position:
                self._entry_bar = i
                self._peak_price = price
                self._has_position = True
            return Signal.BUY

        if prob_up <= (1 - self.confidence_threshold) and has_position:
            return self._exit_position(symbol, portfolio, price)

        return Signal.HOLD

    def _exit_position(self, symbol: str, portfolio: Portfolio, price: float) -> str:
        """Handle position exit with Kelly state tracking."""
        qty = portfolio.positions.get(symbol, 0)
        if qty <= 0 and not self._has_position:
            return Signal.HOLD
        if self.use_kelly:
            avg_entry = portfolio.avg_entry.get(symbol, 0)
            if avg_entry > 0:
                pnl_pct = (price - avg_entry) / avg_entry
                self._kelly_count += 1
                if pnl_pct > 0:
                    self._kelly_wins += 1
                    self._kelly_win_pct += pnl_pct
                else:
                    self._kelly_losses += 1
                    self._kelly_loss_pct += abs(pnl_pct)
        self._entry_bar = -1
        self._peak_price = 0.0
        self._has_position = False
        return Signal.SELL
