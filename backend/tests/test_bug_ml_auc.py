"""Bug: SpreadPredictor._evaluate always returns auc_roc=0.0 for classification.

Line 377-380 of ml_models.py:
    proba = self._model.predict_proba(
        self._scaler.transform(
            self._model.feature_importances_ is not None  # <-- BUG: passes boolean!
        )
    )

The expression `self._model.feature_importances_ is not None` evaluates to `True`,
which is passed to `_scaler.transform()`. This raises ValueError (boolean can't be
transformed), which is caught by the bare `except Exception: pass`, so AUC stays 0.0
and the error is silently swallowed.

The correct code should be:
    proba = self._model.predict_proba(self._scaler.transform(X_test))
"""

import numpy as np
import pytest

from backend.stats_arb.ml_models import SpreadPredictor, TaskType, ModelType


class TestMlAucComputation:
    def test_auc_is_not_zero_for_perfect_classification(self):
        """AUC should be 1.0 for a perfect classifier, not 0.0."""
        predictor = SpreadPredictor(
            model_type=ModelType.RANDOM_FOREST,
            task=TaskType.REVERSION_PROB,
            random_state=42,
        )

        # Manually set up internal state to test _evaluate directly
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        predictor._scaler = StandardScaler()
        predictor._model = RandomForestClassifier(
            n_estimators=10, random_state=42
        )

        # Create perfect classification data: 10 samples, feature_1 = label
        X = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        X_scaled = predictor._scaler.fit_transform(X)
        predictor._model.fit(X_scaled, y)

        # Now test _evaluate with perfect predictions
        y_pred = predictor._model.predict(X_scaled)
        result = predictor._evaluate(y, y_pred, is_classification=True)

        # BUG: auc_roc is always 0.0 because _evaluate passes a boolean
        # to _scaler.transform instead of the actual test features.
        # Expected: 1.0 for perfect prediction.
        assert result.auc_roc == pytest.approx(1.0, abs=0.01), (
            f"auc_roc should be 1.0 for perfect prediction, got {result.auc_roc}. "
            "Bug: _evaluate line 377-380 passes boolean to scaler.transform() instead of X_test."
        )

    def test_auc_is_above_random_for_noisy_data(self):
        """AUC should be > 0.5 for data with some signal, not 0.0."""
        predictor = SpreadPredictor(
            model_type=ModelType.RANDOM_FOREST,
            task=TaskType.REVERSION_PROB,
            random_state=42,
        )

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        predictor._scaler = StandardScaler()
        predictor._model = RandomForestClassifier(
            n_estimators=20, random_state=42
        )

        # Create data with some signal (not perfect)
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        # Label depends on sum of features with some noise
        prob = 1.0 / (1.0 + np.exp(-X.sum(axis=1)))
        y = (prob > 0.5).astype(int)

        X_scaled = predictor._scaler.fit_transform(X)
        predictor._model.fit(X_scaled, y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        predictor._model.fit(X_train, y_train)
        y_pred = predictor._model.predict(X_test)
        result = predictor._evaluate(y_test, y_pred, is_classification=True)

        # BUG: auc_roc is always 0.0 due to boolean passed to scaler.transform()
        # Expected: should be > 0.5 given the signal in the data
        assert result.auc_roc >= 0.0, "auc_roc should be non-negative"
        assert result.auc_roc > 0.5, (
            f"auc_roc should be > 0.5 for data with signal, got {result.auc_roc}. "
            "Bug: _evaluate never actually computes AUC due to line 377-380."
        )

    def test_auc_roc_stored_in_ml_result(self):
        """AUC should be propagated through the full MLResult correctly."""
        predictor = SpreadPredictor(
            model_type=ModelType.RANDOM_FOREST,
            task=TaskType.BREAKOUT_PROB,
        )

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        predictor._scaler = StandardScaler()
        predictor._model = RandomForestClassifier(
            n_estimators=5, random_state=42
        )

        X = np.array([[0.1], [0.9], [0.2], [0.8], [0.15], [0.85]])
        y = np.array([0, 1, 0, 1, 0, 1])
        X_s = predictor._scaler.fit_transform(X)
        predictor._model.fit(X_s, y)
        predictor._feature_names = ["x1"]

        y_pred = predictor._model.predict(X_s)
        result = predictor._evaluate(y, y_pred, is_classification=True)

        result_dict = result.to_dict()

        # BUG: auc_roc is 0.0 instead of being computed correctly
        assert "auc_roc" in result_dict, "auc_roc should be in MLResult.to_dict()"
        assert result_dict["auc_roc"] >= 0.0, "auc_roc must be non-negative"
