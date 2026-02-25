"""Training Orchestrator — model training, evaluation, and artifact saving.

Handles the full training lifecycle:
    Load features → Time-split → Train → Evaluate → Cross-validate → Save

Key improvement over v1:
    Score predictor now uses a dedicated feature set that prioritises
    player batting strength features (the primary signal for T20 score prediction),
    rather than re-using the win predictor's pre-match context features.

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


# =============================================================================
# FEATURE SET DEFINITIONS
# These are maintained here so they're easy to audit, version, and tune.
# Think of them like BOMs (Bill of Materials) for each model.
# =============================================================================

# Columns that are NEVER features (IDs, raw targets, string labels, leakage)
_ALWAYS_EXCLUDE = {
    "match_id", "date", "team1", "team2", "venue", "winner",
    "team1_won", "batting_first_won",
    "team_batting_first", "team_bowling_first",
    "toss_winner", "toss_decision",
    "first_innings_score", "second_innings_score",
    "first_innings_wickets", "second_innings_wickets",
}

# Win predictor: pre-match features only — everything we know BEFORE the toss
WIN_PREDICTOR_FEATURE_GROUPS = {
    "date": ["month", "day_of_week", "year"],
    "toss": ["toss_winner_is_team1", "elected_to_bat"],
    "venue": [
        "venue_avg_1st_inn_score", "venue_median_1st_inn_score",
        "venue_std_1st_inn_score", "matches_at_venue",
        "venue_avg_powerplay_rpo", "venue_avg_middle_rpo", "venue_avg_death_rpo",
        "venue_spin_wicket_pct", "venue_pace_wicket_pct",
        "venue_chase_win_pct", "venue_chase_matches", "is_subcontinent",
    ],
    "h2h": ["h2h_matches_played", "h2h_team1_win_rate"],
    "form": [
        "team1_form_L10", "team2_form_L10", "form_diff",
        "team1_matches_played", "team2_matches_played", "experience_diff",
    ],
    # Player strength features (NEW — from MatchPlayerFeatureEngine)
    "player_batting": [
        "team1_batting_power", "team2_batting_power", "batting_power_diff",
        "team1_top3_sr_L10", "team2_top3_sr_L10", "top_order_form_diff",
        "team1_top3_runs_L10", "team2_top3_runs_L10",
        "team1_avg_boundary_pct", "team2_avg_boundary_pct",
        # NOTE: batting_depth is computed from who ACTUALLY batted in the match
        # (in-match data), not from pre-match squad info. It leaks outcome signal
        # (team2_batting_depth has 0.49 correlation with team1_won because a
        # comfortable chase needs fewer batters). Excluded from win predictor.
    ],
    "player_bowling": [
        "team1_bowling_economy", "team2_bowling_economy", "bowling_economy_diff",
        "team1_bowling_dot_pct", "team2_bowling_dot_pct", "dot_ball_pressure_diff",
        "team1_top_bowler_sr", "team2_top_bowler_sr",
        "team1_spin_bowling_pct", "team2_spin_bowling_pct",
    ],
}

# Score predictor: first innings score prediction
# Primary signal = batting team's strength + venue scoring environment
# Secondary signal = bowling team's defensive strength
SCORE_PREDICTOR_FEATURE_GROUPS = {
    "date": ["month", "year"],
    "venue": [
        "venue_avg_1st_inn_score", "venue_median_1st_inn_score",
        "venue_std_1st_inn_score", "matches_at_venue",
        "venue_avg_powerplay_rpo", "venue_avg_middle_rpo", "venue_avg_death_rpo",
        "is_subcontinent",
    ],
    # Batting team strength (team_batting_first = the team scoring)
    # We use team1 features as proxy (pipeline ensures team1 = batting first in context)
    # NOTE: team1_batting_depth excluded (in-match feature, see WIN_PREDICTOR note)
    "batting_team_strength": [
        "team1_batting_power",
        "team1_top3_sr_L10",
        "team1_top3_runs_L10",
        "team1_avg_boundary_pct",
    ],
    # Bowling team's defensive quality
    "bowling_team_strength": [
        "team2_bowling_economy",
        "team2_bowling_dot_pct",
        "team2_top_bowler_sr",
        "team2_spin_bowling_pct",
        "venue_spin_wicket_pct",  # venue favours spin = suppresses batting team
    ],
    # Match context
    "context": [
        "team1_form_L10",
        "team1_matches_played",
        "toss_winner_is_team1",
        "elected_to_bat",
    ],
}


def _select_features(
    df: pd.DataFrame,
    feature_groups: dict,
    require_numeric: bool = True,
) -> list[str]:
    """Select feature columns that exist in df from the defined feature groups.

    Args:
        df: DataFrame to select from.
        feature_groups: Dict of group_name → list of column names.
        require_numeric: Only include numeric columns (default True).

    Returns:
        List of column names that exist in df and pass type filter.
    """
    all_requested = [col for cols in feature_groups.values() for col in cols]

    # Filter to columns that actually exist
    available = [c for c in all_requested if c in df.columns]
    missing = set(all_requested) - set(available)

    if missing:
        logger.warning(
            f"Feature selection: {len(missing)} requested columns not found "
            f"(run pipeline with player features): {sorted(missing)[:10]}..."
            if len(missing) > 10 else f": {sorted(missing)}"
        )

    if require_numeric:
        numeric_dtypes = ("float64", "int64", "int32", "int8", "float32")
        available = [c for c in available if df[c].dtype in numeric_dtypes]

    logger.info(
        f"Feature selection: {len(available)}/{len(all_requested)} features available"
    )
    return available


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

        df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(df)} match records with {len(df.columns)} columns")

        if "team1_won" not in df.columns:
            raise ValueError(
                "'team1_won' target not found in features. "
                "Run the feature pipeline first: python scripts/train.py --pipeline"
            )

        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        # Use defined feature groups (falls back gracefully if player features absent)
        feature_cols = _select_features(df, WIN_PREDICTOR_FEATURE_GROUPS)

        # Final safety filter: remove any remaining non-numeric or excluded cols
        feature_cols = [
            c for c in feature_cols
            if c not in _ALWAYS_EXCLUDE
            and df[c].dtype in ("float64", "int64", "int32", "int8", "float32")
        ]

        logger.info(f"Win model using {len(feature_cols)} features")
        self._log_feature_groups(feature_cols, WIN_PREDICTOR_FEATURE_GROUPS)

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
        logger.info("TEST SET RESULTS (Win Predictor):")
        logger.info(f"  Accuracy:  {test_accuracy:.4f}")
        logger.info(f"  AUC-ROC:   {test_auc:.4f}")
        logger.info(f"  Log Loss:  {test_logloss:.4f}")
        logger.info("=" * 40)

        return {
            "feature_count": len(feature_cols),
            "metrics": metrics,
            "cv_results": cv_results,
            "test_accuracy": test_accuracy,
            "test_auc": test_auc,
            "test_logloss": test_logloss,
        }

    def train_score_model(self) -> dict:
        """Train the first innings score predictor.

        Uses a dedicated feature set focused on:
        - Batting team strength (primary signal)
        - Bowling team defensive quality (suppression factor)
        - Venue scoring environment (ceiling/floor)

        Target: first_innings_score (continuous, runs)
        """
        logger.info("--- Training Score Predictor ---")

        df = pd.read_parquet(self.features_path)

        if "first_innings_score" not in df.columns:
            logger.error("'first_innings_score' not found — skipping score model")
            return {"error": "Missing target variable"}

        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        # Use dedicated score predictor feature set
        feature_cols = _select_features(df, SCORE_PREDICTOR_FEATURE_GROUPS)
        feature_cols = [
            c for c in feature_cols
            if c not in _ALWAYS_EXCLUDE
            and df[c].dtype in ("float64", "int64", "int32", "int8", "float32")
        ]

        logger.info(f"Score model using {len(feature_cols)} features")
        self._log_feature_groups(feature_cols, SCORE_PREDICTOR_FEATURE_GROUPS)

        X = df[feature_cols]
        y = df["first_innings_score"]

        # Remove rows where target is 0 or missing (incomplete matches)
        valid_mask = (y > 50) & y.notna()
        X, y = X[valid_mask], y[valid_mask]
        logger.info(f"Valid matches for score model: {len(X)} (removed {(~valid_mask).sum()} incomplete)")

        # Time split (80/20 for regression — more training data benefits score model)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Split: Train={len(X_train)} | Val={len(X_val)}")

        predictor = ScorePredictor()
        metrics = predictor.train(X_train, y_train, X_val, y_val)
        predictor.save()

        logger.info("=" * 40)
        logger.info("TEST SET RESULTS (Score Predictor):")
        logger.info(f"  MAE:   {metrics.get('mae', 'N/A'):.2f} runs")
        logger.info(f"  RMSE:  {metrics.get('rmse', 'N/A'):.2f} runs")
        logger.info(f"  R²:    {metrics.get('r2', 'N/A'):.4f}")
        logger.info(f"  Mean actual:    {metrics.get('mean_actual', 0):.1f}")
        logger.info(f"  Mean predicted: {metrics.get('mean_predicted', 0):.1f}")
        logger.info("=" * 40)

        return {
            "feature_count": len(feature_cols),
            "metrics": metrics,
        }

    @staticmethod
    def _log_feature_groups(
        available_cols: list[str],
        feature_groups: dict,
    ) -> None:
        """Log which feature groups contributed columns (useful for debugging)."""
        available_set = set(available_cols)
        for group, cols in feature_groups.items():
            found = [c for c in cols if c in available_set]
            logger.info(f"  [{group}] {len(found)}/{len(cols)} features active")
