"""Ensemble stacking — trains multiple base models and blends them via a meta-model.

Uses expanding-window out-of-fold predictions to prevent the meta-model
from overfitting on base model predictions.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression


class TimeSeriesOOF:
    """Expanding-window time series split for out-of-fold predictions.

    Each fold trains on an increasingly large prefix of the data and
    validates on the next contiguous chunk.  An *embargo* gap is left
    between train and validation to prevent leakage from autocorrelation.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X) -> list[tuple[list[int], list[int]]]:
        n = len(X)
        folds: list[tuple[list[int], list[int]]] = []
        for i in range(1, self.n_splits + 1):
            split_idx = int(n * i / (self.n_splits + 1))
            embargo_end = min(split_idx + self.embargo, n)
            train_idx = list(range(split_idx))
            if i < self.n_splits:
                next_idx = int(n * (i + 1) / (self.n_splits + 1))
                val_idx = list(range(max(split_idx, embargo_end), max(embargo_end, next_idx)))
            else:
                val_idx = list(range(max(split_idx, embargo_end), n))
            folds.append((train_idx, val_idx))
        return folds


class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """Stack multiple classifiers with a meta-model using OOF predictions.

    Usage::

        base = {"rf": RandomForestClassifier(...), "xgb": XGBClassifier(...)}
        stacker = StackedEnsemble(base_models=base)
        stacker.fit(X_train, y_train)
        proba = stacker.predict_proba(X_test)

    The meta-model (default ``LogisticRegression(C=1.0)``) is trained on
    out-of-fold predictions from the base models, which prevents overfitting.
    """

    def __init__(
        self,
        base_models: dict[str, Any] | None = None,
        meta_model: Any | None = None,
        n_splits: int = 5,
    ):
        self.base_models = base_models or {}
        self.meta_model = meta_model or LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.n_splits = n_splits
        self._is_fitted = False
        self.classes_: np.ndarray | None = None

    def fit(self, X, y):
        # ── Train base models on full data ──
        for name, model in self.base_models.items():
            model.fit(X, y)

        # ── Generate OOF meta-features ──
        n = len(X)
        meta_features = np.zeros((n, len(self.base_models)))
        oof_splitter = TimeSeriesOOF(self.n_splits)

        for col_idx, (name, model) in enumerate(self.base_models.items()):
            oof_preds = np.zeros(n)
            for train_idx, val_idx in oof_splitter.split(X):
                if isinstance(X, pd.DataFrame):
                    X_tr = X.iloc[train_idx]
                    y_tr = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
                    X_va = X.iloc[val_idx]
                else:
                    X_tr = X[train_idx]
                    y_tr = y[train_idx]
                    X_va = X[val_idx]

                fold_model = clone(model)
                fold_model.fit(X_tr, y_tr)
                oof_preds[val_idx] = fold_model.predict_proba(X_va)[:, 1]

            # Retrain on full data after OOF
            model.fit(X, y)
            meta_features[:, col_idx] = oof_preds

        # ── Train meta-model on OOF predictions (clone to avoid mutating original) ──
        self.meta_model = clone(self.meta_model)
        self.meta_model.fit(meta_features, y)
        self._is_fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("StackedEnsemble has not been fitted yet.")
        meta = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.base_models.values()
        ])
        return self.meta_model.predict_proba(meta)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        """Return meta-model coefficients as importance of each base model."""
        if self._is_fitted and hasattr(self.meta_model, "coef_"):
            return np.abs(self.meta_model.coef_[0])
        return np.array([])


class MetaLabeledModel(BaseEstimator, ClassifierMixin):
    """Two-stage model: primary predicts direction, meta-labeler filters false signals.

    The meta-labeler (secondary model) predicts *whether the primary model's
    prediction will be correct*.  When it lacks confidence, the wrapper
    returns ``[0.5, 0.5]`` (HOLD), suppressing low-conviction trades.

    Usage::

        primary = RandomForestClassifier(...)
        primary.fit(X_train, y_train)
        wrapper = MetaLabeledModel(primary)
        wrapper.fit_meta(X_train, y_train)
        proba = wrapper.predict_proba(X_test)  # filtered probabilities
    """

    def __init__(
        self,
        primary_model: Any,
        meta_model: Any | None = None,
        meta_threshold: float = 0.5,
    ):
        self.primary_model = primary_model
        self.meta_model = meta_model or LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self.meta_threshold = meta_threshold
        self._meta_fitted = False
        self.classes_: np.ndarray | None = None

    def fit_meta(self, X, y):
        """Train the meta-labeler on top of an already-fitted primary model.

        Creates meta-labels: ``1`` if the primary's prediction matches the
        true label, ``0`` otherwise.  Trains the meta-model on original
        features + primary confidence to predict this correctness target.
        """
        preds = self.primary_model.predict(X)
        primary_conf = self.primary_model.predict_proba(X)[:, 1]
        meta_y = (preds == y).astype(int)
        meta_X = self._build_meta_features(X, primary_conf)
        self.meta_model.fit(meta_X, meta_y)
        self._meta_fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not self._meta_fitted:
            return self.primary_model.predict_proba(X)
        primary_proba = self.primary_model.predict_proba(X)
        primary_conf = primary_proba[:, 1]
        meta_X = self._build_meta_features(X, primary_conf)
        meta_proba = self.meta_model.predict_proba(meta_X)
        meta_conf = meta_proba[:, 1]
        result = np.zeros_like(primary_proba)
        for i in range(len(primary_proba)):
            if meta_conf[i] >= self.meta_threshold:
                result[i] = primary_proba[i]
            else:
                result[i] = [0.5, 0.5]
        return result

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def _build_meta_features(self, X, primary_conf):
        if isinstance(X, pd.DataFrame):
            meta = X.copy()
            meta["primary_conf"] = primary_conf
        else:
            meta = np.column_stack([X, primary_conf])
        return meta

    @property
    def feature_importances_(self):
        """Return the primary model's feature importances."""
        if hasattr(self.primary_model, "feature_importances_"):
            return self.primary_model.feature_importances_
        return np.array([])


class MultiHorizonEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble of models trained at different forecast horizons.

    Each model predicts direction over its own horizon (1-day, 5-day, etc.).
    During inference, predictions are averaged (equal weight) to produce
    a consensus signal that captures information at multiple time scales.

    Usage::

        ensemble = MultiHorizonEnsemble(horizons=[1, 5, 21], base_model_type="rf")
        ensemble.fit(X, y, feature_cols=feature_cols)
        proba = ensemble.predict_proba(X_test)
    """

    def __init__(
        self,
        horizons: list[int] | None = None,
        base_model_type: str = "rf",
        base_params: dict | None = None,
    ):
        self.horizons = horizons or [1, 5, 21]
        self.base_model_type = base_model_type
        self.base_params = base_params or {}
        self.models: dict[int, Any] = {}
        self._is_fitted = False
        self.classes_: np.ndarray | None = None
        self.feature_columns_: list[str] = []

    def fit(self, X, y=None, **kwargs):
        feature_cols = kwargs.get("feature_cols", [])
        if not feature_cols and isinstance(X, pd.DataFrame):
            feature_cols = [c for c in X.columns if c != "target"]
        self.feature_columns_ = feature_cols

        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "close" not in df.columns:
            raise ValueError("MultiHorizonEnsemble.fit() requires a DataFrame with a 'close' column")

        for h in self.horizons:
            df_h = df.copy()
            df_h["target"] = (df_h["close"].shift(-h) > df_h["close"]).astype(int)
            clean = df_h.dropna(subset=feature_cols + ["target"])
            if len(clean) < 50:
                print(f"  [WARN] Horizon {h}: only {len(clean)} samples, skipping")
                continue

            from backend.ml.train import train_model
            model = train_model(clean, feature_cols, self.base_model_type, **self.base_params)
            self.models[h] = model

        if not self.models:
            raise RuntimeError("No models could be trained for any horizon")
        self._is_fitted = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("MultiHorizonEnsemble has not been fitted yet.")
        probs = []
        for h, model in self.models.items():
            p = model.predict_proba(X)
            if len(p.shape) == 1:
                p = np.column_stack([1 - p, p])
            probs.append(p)
        return np.mean(probs, axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        if self.models and hasattr(next(iter(self.models.values())), "feature_importances_"):
            return np.mean(
                [m.feature_importances_ for m in self.models.values() if hasattr(m, "feature_importances_")],
                axis=0,
            )
        return np.array([])


class RegimeAwareModel(BaseEstimator, ClassifierMixin):
    """Adjusts predictions based on detected market regime.

    Uses the Hurst exponent and choppiness index (already computed as
    features) to modulate the primary model's confidence:

    * **Hurst > 0.55** (trending): amplify confidence (trend-following mode).
    * **Hurst < 0.45** (mean-reverting): reverse the signal for reversion.
    * **Choppiness > 60**: suppress signal (hold during choppy markets).

    This wrapper is transparent to the strategy — call ``predict_proba(X)``
    as usual.  The feature array **must** contain ``hurst`` and
    ``choppiness`` columns.
    """

    def __init__(self, primary_model: Any):
        self.primary_model = primary_model
        self.classes_: np.ndarray | None = None

    def fit(self, X, y=None, **kwargs):
        if hasattr(self.primary_model, "fit"):
            self.primary_model.fit(X, y, **kwargs)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        primary_proba = self.primary_model.predict_proba(X)
        result = primary_proba.copy()

        if isinstance(X, pd.DataFrame):
            hurst = X.get("hurst", pd.Series([0.5] * len(X)))
            choppiness = X.get("choppiness", pd.Series([0.0] * len(X)))
        elif isinstance(X, np.ndarray):
            hurst = np.full(len(X), 0.5)
            choppiness = np.full(len(X), 0.0)
        else:
            return primary_proba

        for i in range(len(result)):
            h = hurst.iloc[i] if hasattr(hurst, "iloc") else hurst[i]
            c = choppiness[i] if hasattr(choppiness, "__getitem__") else choppiness

            # Choppy market — hold
            if c > 60:
                result[i] = [0.5, 0.5]
                continue

            # Mean-reverting — reverse signal
            if h < 0.45:
                prob_up = primary_proba[i][1]
                result[i] = [prob_up, 1 - prob_up]
                continue

            # Trending — amplify confidence toward extremes
            if h > 0.55:
                center = 0.5
                prob_up = primary_proba[i][1]
                spread = prob_up - center
                amplified = center + spread * min(h * 2, 1.5)
                result[i] = [1 - amplified, amplified]

        return result

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        if hasattr(self.primary_model, "feature_importances_"):
            return self.primary_model.feature_importances_
        return np.array([])
