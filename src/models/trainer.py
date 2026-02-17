"""Training Orchestrator — model training, evaluation, and artifact saving.

Handles the full training lifecycle:
    Load features → Time-split → Train → Evaluate → Cross-validate → Save

Usage:
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    trainer.train_win_model()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import settings
from src.models.win_predictor import WinPredictor
from src.models.score_predictor import ScorePredictor
from src.utils.logger import logger


class ModelTrainer:
    """Orchestrates training for all prediction models."""

    def __init__(self, features_path: Optional[Path] = None):
        self.features_path = features_path or settings.paths.processed_features

    def train_all(self) -> dict:
        """Train all models sequentially."""
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)

        results = {}
        results["win_predictor"] = self.train_win_model()
        results["score_predictor"] = self.train_score_model()

        logger.info("All models trained successfully")
        return results

    def train_win_model(self) -> dict:
        """Train the match outcome predictor (XGBoost).

        Target: team1_won (binary: 1 if team1 won, 0 otherwise)
        Validation: Time-based split (never random for temporal cricket data)
        """
        logger.info("--- Training Win Predictor ---")

        # Load features from pipeline output
        df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(df)} match records with {len(df.columns)} columns")

        # Verify target variable exists
        if "team1_won" not in df.columns:
            raise ValueError(
                "'team1_won' target not found in features. "
                "Run the feature pipeline first: python scripts/train.py --pipeline"
            )

        # Sort by date for temporal split
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        # Select numeric feature columns (exclude IDs, dates, targets, strings)
        exclude_cols = {
            "match_id", "date", "team1", "team2", "venue", "winner",
            "team1_won", "batting_first_won",
            "team_batting_first", "team_bowling_first",
            "toss_winner", "toss_decision",
            "first_innings_score", "second_innings_score",
            "first_innings_wickets", "second_innings_wickets",
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in ("float64", "int64", "int32", "int8", "float32")
        ]

        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

        X = df[feature_cols]
        y = df["team1_won"]

        # Time-based split (70/15/15)
        n = len(df)
        train_end = int(n * settings.training.train_split)
        val_end = int(n * (settings.training.train_split + settings.training.val_split))

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        logger.info(
            f"Time-based split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}"
        )

        # Train
        predictor = WinPredictor()
        metrics = predictor.train(X_train, y_train, X_val, y_val)
        logger.info(f"Training metrics: {metrics}")

        # Cross-validate (temporal)
        cv_results = predictor.cross_validate_temporal(X, y, n_splits=5)
        logger.info(f"CV results: {cv_results}")

        # Save model artifact
        predictor.save()

        # Feature importance
        feat_imp = predictor.get_feature_importance(top_n=15)
        logger.info(f"Top 15 features:\n{feat_imp.to_string(index=False)}")

        # Final test set evaluation
        test_preds = predictor.predict_proba(X_test)
        from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
        test_accuracy = accuracy_score(y_test, (test_preds >= 0.5).astype(int))
        test_auc = roc_auc_score(y_test, test_preds)
        test_logloss = log_loss(y_test, test_preds)

        logger.info("=" * 40)
        logger.info(f"TEST SET RESULTS:")
        logger.info(f"  Accuracy:  {test_accuracy:.4f}")
        logger.info(f"  AUC-ROC:   {test_auc:.4f}")
        logger.info(f"  Log Loss:  {test_logloss:.4f}")
        logger.info("=" * 40)

        return {
            "metrics": metrics,
            "cv_results": cv_results,
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
            "test_logloss": test_logloss,
        }

    def train_score_model(self) -> dict:
        """Train the first innings score predictor (XGBoost regression)."""
        logger.info("--- Training Score Predictor ---")

        df = pd.read_parquet(self.features_path)

        if "first_innings_score" not in df.columns:
            logger.error("'first_innings_score' not found — skipping score model")
            return {"error": "Missing target variable"}

        # Sort by date
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        exclude_cols = {
            "match_id", "date", "team1", "team2", "venue", "winner",
            "team1_won", "batting_first_won",
            "team_batting_first", "team_bowling_first",
            "toss_winner", "toss_decision",
            "first_innings_score", "second_innings_score",
            "first_innings_wickets", "second_innings_wickets",
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and df[c].dtype in ("float64", "int64", "int32", "int8", "float32")
        ]

        X = df[feature_cols]
        y = df["first_innings_score"]

        # Time split (80/20 for regression)
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Split: Train={len(X_train)} | Val={len(X_val)}")

        predictor = ScorePredictor()
        metrics = predictor.train(X_train, y_train, X_val, y_val)
        predictor.save()

        logger.info(f"Score predictor metrics: {metrics}")
        return {"metrics": metrics}
