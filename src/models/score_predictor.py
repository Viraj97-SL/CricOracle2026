"""XGBoost Score Predictor — First innings final score prediction.

Module B1: Tabular model for predicting the final score given pre-match features.
Trained ONLY on first innings data (key insight from your Notebook 51).

Usage:
    from src.models.score_predictor import ScorePredictor

    predictor = ScorePredictor()
    predictor.train(X_train, y_train)
    predicted_scores = predictor.predict(X_test)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import settings
from src.utils.logger import logger


class ScorePredictor:
    """XGBoost regressor for first innings score prediction."""

    def __init__(self):
        self.model = None
        self.feature_names: list[str] = []
        self.is_trained = False
        self.metrics: dict = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> dict:
        """Train the score prediction model."""
        logger.info(f"Training ScorePredictor on {len(X_train)} first-innings samples...")
        self.feature_names = list(X_train.columns)

        self.model = XGBRegressor(
            max_depth=settings.xgboost.max_depth,
            learning_rate=settings.xgboost.learning_rate,
            n_estimators=settings.xgboost.n_estimators,
            subsample=settings.xgboost.subsample,
            colsample_bytree=settings.xgboost.colsample_bytree,
            random_state=settings.xgboost.random_state,
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.is_trained = True

        # Evaluate
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        self.metrics = self._evaluate(eval_X, eval_y)

        logger.info(
            f"Training complete — MAE: {self.metrics['mae']:.1f} runs, "
            f"R²: {self.metrics['r2']:.4f}"
        )
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict first innings score."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Calculate regression metrics."""
        preds = self.predict(X)
        return {
            "mae": round(mean_absolute_error(y, preds), 2),
            "rmse": round(np.sqrt(mean_squared_error(y, preds)), 2),
            "r2": round(r2_score(y, preds), 4),
            "mean_actual": round(y.mean(), 1),
            "mean_predicted": round(preds.mean(), 1),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or settings.paths.models / "score_predictor_xgb.pkl"
        joblib.dump({"model": self.model, "feature_names": self.feature_names,
                      "metrics": self.metrics}, path)
        logger.info(f"Score model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        path = path or settings.paths.models / "score_predictor_xgb.pkl"
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.is_trained = True
