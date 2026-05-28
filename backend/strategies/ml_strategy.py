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
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.base_buy_size = base_buy_size
        self.buy_size = base_buy_size
        self.sell_portion = 100.0
        self.model = None
        self.feature_columns: list[str] = []
        self._features_df: pd.DataFrame | None = None
        self._model_loaded = False

    def _position_size(self, prob_up: float) -> float:
        if prob_up >= 0.85:
            return self.base_buy_size * 2.0
        if prob_up >= 0.75:
            return self.base_buy_size * 1.5
        if prob_up >= 0.65:
            return self.base_buy_size * 1.0
        return self.base_buy_size * 0.5

    def init(self, data: pd.DataFrame) -> None:
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

        df = compute_features(data.copy())
        df = df.reset_index(drop=True)
        self._features_df = df

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

        if prob_up >= self.confidence_threshold:
            self.buy_size = self._position_size(prob_up)
            return Signal.BUY

        if prob_up <= (1 - self.confidence_threshold) and has_position:
            return Signal.SELL

        return Signal.HOLD
