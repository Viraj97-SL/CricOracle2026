"""Training Orchestrator — coordinates model training, tuning, and evaluation.

Handles the full training lifecycle: data splitting, hyperparameter tuning
with Optuna, model training, evaluation, and artifact saving.

Usage:
    from src.models.trainer import ModelTrainer

    trainer = ModelTrainer()
    trainer.train_all()          # Train all models
    trainer.train_win_model()    # Train just the win predictor
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
        # TODO: Add LSTM training (Phase 2)

        logger.info("All models trained successfully")
        return results

    def train_win_model(self) -> dict:
        """Train the match outcome predictor."""
        logger.info("--- Training Win Predictor ---")

        # Load features
        df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(df)} match records")

        # Prepare target variable
        # TODO: Compute 'team1_won' target from match results
        # For now, placeholder logic:
        if "team1_won" not in df.columns:
            logger.warning("'team1_won' target not found — creating from score comparison")
            if "first_innings_score" in df.columns:
                # Simple proxy: higher first innings score = team1 won (placeholder)
                df["team1_won"] = (df.get("team1_score", 0) > df.get("team2_score", 0)).astype(int)
            else:
                raise ValueError("Cannot determine match winner from features. "
                                 "Add 'team1_won' column to match features.")

        # Select features (exclude IDs, dates, targets, strings)
        exclude_cols = {"match_id", "date", "team1", "team2", "venue",
                        "team1_won", "winner", "toss_winner"}
        feature_cols = [c for c in df.columns
                        if c not in exclude_cols and df[c].dtype in ("float64", "int64", "int8")]

        X = df[feature_cols]
        y = df["team1_won"]

        # Time-based split (CRITICAL: never random for cricket)
        df_sorted = df.sort_values("date")
        split_idx = int(len(df_sorted) * settings.training.train_split)
        val_idx = int(len(df_sorted) * (settings.training.train_split + settings.training.val_split))

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:val_idx]
        y_val = y.iloc[split_idx:val_idx]
        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]

        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Train
        predictor = WinPredictor()
        metrics = predictor.train(X_train, y_train, X_val, y_val)

        # Cross-validate
        cv_results = predictor.cross_validate_temporal(X, y, n_splits=5)

        # Save
        predictor.save()

        # Feature importance
        feat_imp = predictor.get_feature_importance(top_n=15)
        logger.info(f"Top features:\n{feat_imp.to_string(index=False)}")

        return {"metrics": metrics, "cv_results": cv_results}

    def train_score_model(self) -> dict:
        """Train the first innings score predictor."""
        logger.info("--- Training Score Predictor ---")

        df = pd.read_parquet(self.features_path)

        # Filter to first innings only
        if "inning_no" in df.columns:
            df = df[df["inning_no"] == 1]

        if "first_innings_score" not in df.columns:
            logger.error("'first_innings_score' target not found in features")
            return {"error": "Missing target variable"}

        exclude_cols = {"match_id", "date", "team1", "team2", "venue",
                        "first_innings_score", "team1_won", "winner"}
        feature_cols = [c for c in df.columns
                        if c not in exclude_cols and df[c].dtype in ("float64", "int64", "int8")]

        X = df[feature_cols]
        y = df["first_innings_score"]

        # Time split
        split_idx = int(len(df) * settings.training.train_split)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        predictor = ScorePredictor()
        metrics = predictor.train(X_train, y_train, X_val, y_val)
        predictor.save()

        return {"metrics": metrics}
