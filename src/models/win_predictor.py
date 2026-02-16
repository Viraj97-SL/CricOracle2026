"""Match Outcome Predictor — XGBoost + LightGBM + CatBoost Stacking Ensemble.

Module A of the CricOracle prediction system.
Predicts P(team1 wins) given match features.

Usage:
    from src.models.win_predictor import WinPredictor

    predictor = WinPredictor()
    predictor.train(X_train, y_train, X_val, y_val)
    probs = predictor.predict_proba(X_test)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import settings
from src.utils.logger import logger


@dataclass
class PredictionResult:
    """Structured prediction output."""

    team1: str
    team2: str
    team1_win_prob: float
    team2_win_prob: float
    confidence: str
    model_version: str

    @property
    def predicted_winner(self) -> str:
        return self.team1 if self.team1_win_prob > 0.5 else self.team2


class WinPredictor:
    """Stacking ensemble for match outcome prediction.

    Layer 1: XGBoost, LightGBM, CatBoost (base learners)
    Layer 2: Logistic Regression (meta-learner on base predictions)

    Start with XGBoost only, then add other base learners incrementally.
    """

    def __init__(self):
        self.base_model = None
        self.meta_model = None
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
        """Train the win prediction model.

        Args:
            X_train: Training features.
            y_train: Training labels (1 = team1 wins, 0 = team2 wins).
            X_val: Validation features (for early stopping).
            y_val: Validation labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Training WinPredictor on {len(X_train)} samples...")
        self.feature_names = list(X_train.columns)

        # XGBoost with early stopping
        params = {
            "max_depth": settings.xgboost.max_depth,
            "learning_rate": settings.xgboost.learning_rate,
            "n_estimators": settings.xgboost.n_estimators,
            "subsample": settings.xgboost.subsample,
            "colsample_bytree": settings.xgboost.colsample_bytree,
            "min_child_weight": settings.xgboost.min_child_weight,
            "reg_alpha": settings.xgboost.reg_alpha,
            "reg_lambda": settings.xgboost.reg_lambda,
            "eval_metric": settings.xgboost.eval_metric,
            "random_state": settings.xgboost.random_state,
        }

        self.base_model = XGBClassifier(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        self.base_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.is_trained = True

        # Evaluate
        eval_X = X_val if X_val is not None else X_train
        eval_y = y_val if y_val is not None else y_train
        self.metrics = self._evaluate(eval_X, eval_y)
        logger.info(
            f"Training complete — "
            f"Log Loss: {self.metrics['log_loss']:.4f}, "
            f"AUC-ROC: {self.metrics['auc_roc']:.4f}, "
            f"Accuracy: {self.metrics['accuracy']:.4f}"
        )

        return self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict win probability for team1.

        Args:
            X: Feature DataFrame (same columns as training data).

        Returns:
            Array of probabilities [P(team1_wins)].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        probs = self.base_model.predict_proba(X)[:, 1]
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary outcome (1 = team1 wins)."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Calculate evaluation metrics."""
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        return {
            "log_loss": round(log_loss(y, probs), 4),
            "auc_roc": round(roc_auc_score(y, probs), 4),
            "accuracy": round(accuracy_score(y, preds), 4),
            "n_samples": len(y),
            "win_rate_actual": round(y.mean(), 4),
            "win_rate_predicted": round(probs.mean(), 4),
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        importance = self.base_model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).head(top_n)

        return feat_imp

    def cross_validate_temporal(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict:
        """Time-series cross-validation (NEVER use random split for cricket data).

        Uses expanding window: train on matches 1..N, validate on N+1..M.
        """
        logger.info(f"Running temporal CV with {n_splits} splits...")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = {"log_loss": [], "auc_roc": [], "accuracy": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Temporary model for this fold
            model = XGBClassifier(
                max_depth=settings.xgboost.max_depth,
                learning_rate=settings.xgboost.learning_rate,
                n_estimators=settings.xgboost.n_estimators,
                random_state=settings.xgboost.random_state,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)

            scores["log_loss"].append(log_loss(y_val, probs))
            scores["auc_roc"].append(roc_auc_score(y_val, probs))
            scores["accuracy"].append(accuracy_score(y_val, preds))

            logger.info(
                f"  Fold {fold + 1}: LogLoss={scores['log_loss'][-1]:.4f}, "
                f"AUC={scores['auc_roc'][-1]:.4f}"
            )

        result = {
            metric: {
                "mean": round(np.mean(vals), 4),
                "std": round(np.std(vals), 4),
            }
            for metric, vals in scores.items()
        }

        logger.info(
            f"CV Results — LogLoss: {result['log_loss']['mean']:.4f} "
            f"(±{result['log_loss']['std']:.4f}), "
            f"AUC: {result['auc_roc']['mean']:.4f} "
            f"(±{result['auc_roc']['std']:.4f})"
        )

        return result

    def save(self, path: Optional[Path] = None) -> Path:
        """Save trained model to disk."""
        path = path or settings.paths.models / "win_predictor_xgb.pkl"
        joblib.dump({
            "model": self.base_model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }, path)
        logger.info(f"Model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        """Load trained model from disk."""
        path = path or settings.paths.models / "win_predictor_xgb.pkl"
        data = joblib.load(path)
        self.base_model = data["model"]
        self.feature_names = data["feature_names"]
        self.metrics = data["metrics"]
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
