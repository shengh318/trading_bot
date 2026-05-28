import numpy as np
import pandas as pd

from backend.ml.features import compute_features, get_feature_columns
from backend.ml.model import load_model
from backend.strategies.base import Strategy, Signal, Portfolio


class MLStrategy(Strategy):
    def __init__(
        self,
        model_name: str = "multi_symbol_model",
        confidence_threshold: float = 0.55,
        base_buy_size: float = 1000.0,
        use_kelly: bool = False,
        max_hold_bars: int = 30,
        trailing_stop_pct: float = 0.05,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.base_buy_size = base_buy_size
        self.buy_size = base_buy_size
        self.sell_portion = 100.0
        self.use_kelly = use_kelly
        self.max_hold_bars = max_hold_bars
        self.trailing_stop_pct = trailing_stop_pct
        self.model = None
        self.feature_columns: list[str] = []
        self._features_df: pd.DataFrame | None = None
        self._model_loaded = False
        self._kelly_wins = 0.0
        self._kelly_losses = 0.0
        self._kelly_win_pct = 0.0
        self._kelly_loss_pct = 0.0
        self._kelly_count = 0
        self._entry_bar = -1
        self._peak_price = 0.0

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
        df = df.reset_index(drop=True)
        self._features_df = df

        try:
            self.model, metadata = load_model(self.model_name)
            self.feature_columns = metadata.feature_columns
            self._model_loaded = True
        except FileNotFoundError:
            print(
                f"WARNING [MLStrategy]: Model '{self.model_name}' not found. "
                f"Train one first: python -m backend.ml.train"
            )
            self._model_loaded = False
            return

        self._apply_metadata_params(metadata)

    def _apply_metadata_params(self, metadata) -> None:
        """Override strategy defaults with saved training params from metadata."""
        train_params = metadata.params
        if not isinstance(train_params, dict):
            return

        tp = train_params.get("confidence_threshold")
        if tp is not None:
            self.confidence_threshold = float(tp)
        kelly = train_params.get("kelly")
        if kelly is not None:
            self.use_kelly = bool(kelly)
        mhb = train_params.get("max_hold_bars")
        if mhb is not None:
            self.max_hold_bars = int(mhb)
        tsp = train_params.get("trailing_stop_pct")
        if tsp is not None:
            self.trailing_stop_pct = float(tsp)

    def next(self, i: int, data: pd.DataFrame, portfolio: Portfolio) -> str:
        if not self._model_loaded or self._features_df is None:
            return Signal.HOLD

        if i >= len(self._features_df):
            return Signal.HOLD

        row = self._features_df.iloc[i]
        try:
            features = row[self.feature_columns].values.reshape(1, -1)
        except (KeyError, ValueError):
            return Signal.HOLD

        if np.any(np.isnan(features)):
            return Signal.HOLD

        proba = self.model.predict_proba(features)[0]
        if len(proba) < 2:
            return Signal.HOLD
        prob_up = proba[1]

        symbol = getattr(self, "_symbol", "ASSET")
        has_position = portfolio.positions.get(symbol, 0) > 0
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
            self._entry_bar = i
            self._peak_price = price
            return Signal.BUY

        if prob_up <= (1 - self.confidence_threshold) and has_position:
            return self._exit_position(symbol, portfolio, price)

        return Signal.HOLD

    def _exit_position(self, symbol: str, portfolio: Portfolio, price: float) -> str:
        """Handle position exit with Kelly state tracking."""
        qty = portfolio.positions.get(symbol, 0)
        if qty <= 0:
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
        return Signal.SELL
