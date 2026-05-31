"""
Phase 9 — Machine Learning Extensions.

ML models that augment the statistical framework by predicting:
    - Probability of spread reversion
    - Expected reversion magnitude
    - Expected time-to-mean
    - Probability of spread breakout

Important: ML augments the statistical framework — it does NOT replace it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from backend.cpp_ext import compute_time_to_mean, rolling_half_life
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("stats_arb.ml")


class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    LOGISTIC = "logistic"


class TaskType(str, Enum):
    REVERSION_PROB = "reversion_prob"
    REVERSION_MAGNITUDE = "reversion_magnitude"
    TIME_TO_MEAN = "time_to_mean"
    BREAKOUT_PROB = "breakout_prob"


@dataclass
class MLResult:
    """Machine learning prediction result.

    Attributes
    ----------
    task : TaskType
        The prediction task.
    model_type : ModelType
        The model type used.
    accuracy : float
        Classification accuracy (if classification task).
    auc_roc : float
        AUC-ROC score (if applicable).
    r2 : float
        R-squared (if regression task).
    mae : float
        Mean absolute error.
    feature_importance : dict[str, float]
        Feature importance scores.
    predictions : np.ndarray
        Model predictions.
    actuals : np.ndarray
        Actual values.
    """

    task: TaskType
    model_type: ModelType
    accuracy: float
    auc_roc: float
    r2: float
    mae: float
    feature_importance: dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "model_type": self.model_type.value,
            "accuracy": round(self.accuracy, 4),
            "auc_roc": round(self.auc_roc, 4),
            "r2": round(self.r2, 4),
            "mae": round(self.mae, 4),
            "feature_importance": {
                k: round(v, 4) for k, v in sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1], reverse=True
                )[:10]
            },
        }


class SpreadPredictor:
    """ML-based spread prediction models.

    Builds features from the spread and related time series, then
    trains models to predict spread behaviour.

    Parameters
    ----------
    model_type : ModelType
        Which ML model to use.
    task : TaskType
        Prediction target.
    test_size : float
        Fraction of data for testing.
    n_iter : int
        Number of iterations for tuning.
    random_state : int
        Random seed.
    """

    FEATURE_NAMES: list[str] = [
        "zscore",
        "zscore_lag1",
        "zscore_lag2",
        "zscore_lag5",
        "spread_velocity",
        "spread_acceleration",
        "volatility_20d",
        "volatility_60d",
        "hurst_63d",
        "correlation_60d",
        "regime_trending",
        "regime_volatile",
        "half_life_63d",
        "spread_autocorr_1",
        "spread_autocorr_5",
        "rolling_mean_20d",
        "rolling_std_20d",
        "distance_from_mean",
        "zscore_high_10d",
        "zscore_low_10d",
    ]

    def __init__(
        self,
        model_type: ModelType = ModelType.RANDOM_FOREST,
        task: TaskType = TaskType.REVERSION_PROB,
        test_size: float = 0.20,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.task = task
        self.test_size = test_size
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: Any = None
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []

    def build_features(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        hurst: Optional[pd.Series] = None,
        correlation: Optional[pd.Series] = None,
        regime: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Build feature matrix from spread time series.

        Parameters
        ----------
        spread : pd.Series
            The spread series.
        zscore : pd.Series
            Z-score series.
        hurst : pd.Series, optional
            Rolling Hurst series.
        correlation : pd.Series, optional
            Rolling correlation series.
        regime : pd.Series, optional
            Regime series.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by date.
        """
        df = pd.DataFrame(index=spread.index)
        df["zscore"] = zscore
        df["zscore_lag1"] = zscore.shift(1)
        df["zscore_lag2"] = zscore.shift(2)
        df["zscore_lag5"] = zscore.shift(5)
        df["spread_velocity"] = spread.diff()
        df["spread_acceleration"] = spread.diff().diff()
        df["volatility_20d"] = spread.rolling(20).std()
        df["volatility_60d"] = spread.rolling(60).std()

        if hurst is not None:
            df["hurst_63d"] = hurst
        else:
            df["hurst_63d"] = 0.5

        if correlation is not None:
            df["correlation_60d"] = correlation
        else:
            df["correlation_60d"] = 0.0

        if regime is not None:
            df["regime_trending"] = (regime == "trending").astype(float)
            df["regime_volatile"] = (regime == "volatile").astype(float)
        else:
            df["regime_trending"] = 0.0
            df["regime_volatile"] = 0.0

        hl = SpreadPredictor._estimate_half_life_series(spread)
        df["half_life_63d"] = hl
        df["spread_autocorr_1"] = spread.autocorr(lag=1)
        df["spread_autocorr_5"] = spread.autocorr(lag=5)
        df["rolling_mean_20d"] = spread.rolling(20).mean()
        df["rolling_std_20d"] = spread.rolling(20).std()
        df["distance_from_mean"] = spread - spread.expanding().mean()
        df["zscore_high_10d"] = zscore.rolling(10).max()
        df["zscore_low_10d"] = zscore.rolling(10).min()

        return df.dropna()

    def build_labels(
        self, spread: pd.Series, zscore: pd.Series, forecast_horizon: int = 5
    ) -> pd.Series:
        """Build labels for the specified task.

        For reversion_prob:
            1 if |zscore| decreases in the next forecast_horizon days, else 0.
        For reversion_magnitude:
            Change in zscore over next forecast_horizon days.
        For time_to_mean:
            Days until zscore crosses 0.
        For breakout_prob:
            1 if |zscore| > 3 in the next forecast_horizon days, else 0.
        """
        if self.task == TaskType.REVERSION_PROB:
            current_abs = zscore.abs()
            future_abs = zscore.shift(-forecast_horizon).abs()
            labels = (future_abs < current_abs).astype(int)

        elif self.task == TaskType.REVERSION_MAGNITUDE:
            labels = -zscore.diff(forecast_horizon).shift(-forecast_horizon)

        elif self.task == TaskType.TIME_TO_MEAN:
            labels = self._compute_time_to_mean(zscore, max_horizon=60)

        elif self.task == TaskType.BREAKOUT_PROB:
            future_max = zscore.shift(-forecast_horizon).abs()
            labels = (future_max > 3.0).astype(int)

        else:
            raise ValueError(f"Unknown task: {self.task}")

        return labels.dropna()

    def train(
        self,
        spread: pd.Series,
        zscore: pd.Series,
        hurst: Optional[pd.Series] = None,
        correlation: Optional[pd.Series] = None,
        regime: Optional[pd.Series] = None,
        forecast_horizon: int = 5,
    ) -> MLResult:
        """Train model and evaluate on test set.

        Returns MLResult with performance metrics and feature importance.
        """
        features = self.build_features(spread, zscore, hurst, correlation, regime)
        labels = self.build_labels(spread, zscore, forecast_horizon)

        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        if len(features) < 50:
            logger.warning(f"Too few samples ({len(features)}) for ML training")
            return self._empty_result()

        X = features.values
        y = labels.values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False, random_state=self.random_state
        )

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        is_classification = self.task in (
            TaskType.REVERSION_PROB, TaskType.BREAKOUT_PROB,
        )
        self._model = self._build_model(is_classification)
        self._model.fit(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)
        self._feature_names = list(features.columns)

        return self._evaluate(y_test, y_pred, is_classification)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions on new feature data."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self._scaler.transform(features.values)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X)

    def _build_model(self, is_classification: bool) -> Any:
        if self.model_type == ModelType.RANDOM_FOREST:
            if is_classification:
                return RandomForestClassifier(
                    n_estimators=200, max_depth=10,
                    random_state=self.random_state, n_jobs=-1,
                )
            return RandomForestRegressor(
                n_estimators=200, max_depth=10,
                random_state=self.random_state, n_jobs=-1,
            )

        if self.model_type == ModelType.LOGISTIC:
            return LogisticRegression(
                max_iter=self.n_iter, random_state=self.random_state,
            )

        if self.model_type in (ModelType.XGBOOST, ModelType.LIGHTGBM):
            return self._build_boosted_model(is_classification)

        return RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1,
        )

    def _build_boosted_model(self, is_classification: bool) -> Any:
        try:
            if self.model_type == ModelType.XGBOOST:
                import xgboost as xgb
                if is_classification:
                    return xgb.XGBClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.05,
                        random_state=self.random_state, eval_metric="logloss",
                    )
                return xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    random_state=self.random_state,
                )
            elif self.model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                if is_classification:
                    return lgb.LGBMClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.05,
                        random_state=self.random_state, verbose=-1,
                    )
                return lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    random_state=self.random_state, verbose=-1,
                )
        except ImportError:
            logger.warning(f"{self.model_type.value} not installed, falling back to RandomForest")
            return RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1,
            )

    def _evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, is_classification: bool
    ) -> MLResult:
        importances = self._get_feature_importance()

        if is_classification:
            acc = float(accuracy_score(y_true, y_pred))
            auc = 0.0
            try:
                auc = float(roc_auc_score(y_true, y_pred))
            except Exception:
                pass
            r2 = 0.0
            mae = float(mean_absolute_error(y_true, y_pred))
        else:
            acc = 0.0
            auc = 0.0
            r2 = float(r2_score(y_true, y_pred))
            mae = float(mean_absolute_error(y_true, y_pred))

        return MLResult(
            task=self.task,
            model_type=self.model_type,
            accuracy=acc,
            auc_roc=auc,
            r2=r2,
            mae=mae,
            feature_importance=importances,
            predictions=y_pred,
            actuals=y_true,
        )

    def _get_feature_importance(self) -> dict[str, float]:
        if self._model is None or not self._feature_names:
            return {}
        if hasattr(self._model, "feature_importances_"):
            imp = self._model.feature_importances_
            return dict(zip(self._feature_names, imp.tolist()))
        if hasattr(self._model, "coef_"):
            imp = np.abs(self._model.coef_[0]) if self._model.coef_.ndim > 1 else np.abs(self._model.coef_)
            return dict(zip(self._feature_names, imp.tolist()))
        return {}

    @staticmethod
    def _compute_time_to_mean(zscore: pd.Series, max_horizon: int = 60) -> pd.Series:
        """Compute days until z-score crosses 0 — C++ accelerated."""
        result = compute_time_to_mean(zscore.values.astype(float), max_horizon)
        return pd.Series(result, index=zscore.index)

    @staticmethod
    def _estimate_half_life_series(spread: pd.Series) -> pd.Series:
        """Estimate rolling half-life — C++ accelerated."""
        vals = rolling_half_life(spread.values.astype(float), 63)
        return pd.Series(vals, index=spread.index)

    def _empty_result(self) -> MLResult:
        return MLResult(
            task=self.task,
            model_type=self.model_type,
            accuracy=0.0,
            auc_roc=0.0,
            r2=0.0,
            mae=0.0,
            feature_importance={},
            predictions=np.array([]),
            actuals=np.array([]),
        )
