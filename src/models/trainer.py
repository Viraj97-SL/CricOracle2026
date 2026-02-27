"""Training Orchestrator — model training, evaluation, and artifact saving.

Handles the full training lifecycle:
    Load features → Permutation augmentation → Time-split → Train → Evaluate → Save

Key improvements (v2):
    - Permutation augmentation: each match is duplicated with team1/team2 swapped,
      doubling the dataset and forcing the model to be symmetric (Sankaranarayanan 2023).
    - Score predictor uses dedicated feature set focused on batting team strength.
    - batting_depth excluded (in-match leakage, 0.49 correlation with target).

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


def _augment_with_permutations(df: pd.DataFrame) -> pd.DataFrame:
    """Double the dataset by adding swapped team1/team2 rows.

    For each match (team1 vs team2, label=1), add a mirrored row
    (team2 vs team1, label=0). This forces the model to be symmetric
    w.r.t. team ordering — a key finding from Sankaranarayanan et al. (2023).

    Columns that are swapped: all team1_* ↔ team2_*, differential features
    are negated, and the target (team1_won) is flipped.
    """
    df2 = df.copy()

    # Swap team1_*/team2_* column pairs
    t1_cols = [c for c in df2.columns if c.startswith("team1_")]
    t2_cols = [c for c in df2.columns if c.startswith("team2_")]

    for t1_col in t1_cols:
        t2_col = t1_col.replace("team1_", "team2_")
        if t2_col in df2.columns:
            df2[t1_col], df2[t2_col] = df[t2_col].values, df[t1_col].values

    # Negate difference features (A-B becomes B-A)
    diff_cols = [
        "form_diff", "batting_power_diff", "top_order_form_diff",
        "bowling_economy_diff", "dot_ball_pressure_diff", "experience_diff",
        "h2h_team1_win_rate",
    ]
    for col in diff_cols:
        if col in df2.columns:
            if col == "h2h_team1_win_rate":
                df2[col] = 1.0 - df[col]
            else:
                df2[col] = -df[col]

    # Flip toss features
    if "toss_winner_is_team1" in df2.columns:
        df2["toss_winner_is_team1"] = 1 - df["toss_winner_is_team1"]

    # Flip team identity string columns
    for col in ["team1", "team2"]:
        other = "team2" if col == "team1" else "team1"
        if col in df2.columns and other in df.columns:
            df2[col] = df[other]

    # Flip target
    if "team1_won" in df2.columns:
        df2["team1_won"] = 1 - df["team1_won"]

    combined = pd.concat([df, df2], ignore_index=True)
    return combined


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
# Improvements (v2):
#   - venue_score_index: venue avg / global avg (Perera 2022 venue normalization)
#   - powerplay/death rpo as separate signals (phase-specific per Bandulasiri 2023)
#   - batting_economy_pressure: bowling team death economy (death overs suppression)
# NOTE: batting_depth excluded (in-match leakage)
SCORE_PREDICTOR_FEATURE_GROUPS = {
    "date": ["month", "year"],
    "venue": [
        "venue_avg_1st_inn_score", "venue_median_1st_inn_score",
        "venue_std_1st_inn_score", "matches_at_venue",
        "venue_avg_powerplay_rpo", "venue_avg_middle_rpo", "venue_avg_death_rpo",
        "is_subcontinent", "venue_spin_wicket_pct",
    ],
    # Batting team strength (team_batting_first = the team scoring)
    # We use team1 features as proxy (pipeline ensures team1 = batting first in context)
    "batting_team_strength": [
        "team1_batting_power",
        "team1_top3_sr_L10",
        "team1_top3_runs_L10",
        "team1_avg_boundary_pct",
        "team1_spin_bowling_pct",   # proxy for balance (all-rounder spinners bat too)
    ],
    # Bowling team's defensive quality
    "bowling_team_strength": [
        "team2_bowling_economy",
        "team2_bowling_dot_pct",
        "team2_top_bowler_sr",
        "team2_spin_bowling_pct",
        "team2_avg_boundary_pct",   # bowling team's batting depth proxy (stronger teams have better bowlers)
    ],
    # Match context
    "context": [
        "team1_form_L10",
        "team1_matches_played",
        "team2_form_L10",           # chasing team's form (pressure on batting team)
        "toss_winner_is_team1",
        "elected_to_bat",
        "batting_power_diff",       # relative strength differential
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

        # Time-based split (70/15/15) BEFORE augmentation to avoid data leakage
        n = len(df)
        train_end = int(n * settings.training.train_split)
        val_end = int(n * (settings.training.train_split + settings.training.val_split))

        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Apply permutation augmentation to training set only (Sankaranarayanan 2023)
        train_df = df.iloc[:train_end].copy()
        augmented = _augment_with_permutations(train_df)
        augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)
        X_train = augmented[feature_cols]
        y_train = augmented["team1_won"]

        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

        logger.info(
            f"Time-based split: Train={len(X_train)} (augmented ×2) | "
            f"Val={len(X_val)} | Test={len(X_test)}"
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

    def train_lstm_model(self) -> dict:
        """Train the LSTM ball-by-ball score predictor.

        Builds over-by-over sequences from the raw ball-by-ball CSV,
        then trains ScoreLSTM. Requires torch to be installed.

        Target: final innings total (regression).
        Per-over features: [runs, wickets, extras, dot_balls, boundaries,
                            current_rr, cumulative_score, wickets_total,
                            phase_encoded, over_number]
        """
        logger.info("--- Training LSTM Score Predictor ---")
        try:
            import torch
            from torch.utils.data import DataLoader
            from src.models.score_lstm import ScoreLSTM, OverByOverDataset, train_score_lstm
            from src.data.loader import CricketDataLoader
        except ImportError as e:
            logger.error(f"LSTM training requires torch: {e}")
            return {"error": str(e)}

        # Load raw ball-by-ball data
        loader = CricketDataLoader()
        df = loader.load_and_clean(modern_era_only=True)
        logger.info(f"Loaded {len(df)} ball-by-ball rows for LSTM training")

        # Build over-by-over sequences per innings
        over_sequences, targets = self._build_over_sequences(df)
        logger.info(f"Built {len(over_sequences)} innings sequences")

        if len(over_sequences) < 100:
            logger.error("Too few sequences for LSTM training — check data")
            return {"error": "Insufficient data"}

        # Split (80/20 temporal)
        split = int(len(over_sequences) * 0.8)
        train_ds = OverByOverDataset(over_sequences[:split], targets[:split])
        val_ds = OverByOverDataset(over_sequences[split:], targets[split:])
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=64)

        model = ScoreLSTM(input_dim=10, hidden_dim=128, num_layers=2, dropout=0.3)
        save_path = str(settings.paths.models / "score_lstm.pt")
        model = train_score_lstm(model, train_dl, val_dl, epochs=100, patience=15, save_path=save_path)

        logger.info(f"LSTM model saved to {save_path}")
        return {"status": "trained", "path": save_path}

    @staticmethod
    def _build_over_sequences(df: pd.DataFrame) -> tuple[list, list]:
        """Convert ball-by-ball DataFrame to per-over feature sequences."""
        sequences = []
        targets = []

        # Add over_number column (0-indexed)
        if "over" not in df.columns:
            return [], []

        innings_groups = df.groupby(["match_id", "innings"])
        for (match_id, innings), inn_df in innings_groups:
            if innings != 1:   # First innings only for pre-match score prediction
                continue

            final_score = inn_df["runs_off_bat"].fillna(0).sum() + inn_df["extras"].fillna(0).sum()
            if final_score < 50:
                continue  # Skip incomplete innings

            # Aggregate per over
            over_feats = []
            for over_num in range(1, 21):
                ov = inn_df[inn_df["over"] == over_num]
                if ov.empty:
                    over_feats.append([0.0] * 10)
                    continue

                runs = float(ov["runs_off_bat"].fillna(0).sum() + ov["extras"].fillna(0).sum())
                wickets = float(ov["wicket_type"].notna().sum()) if "wicket_type" in ov.columns else 0.0
                extras = float(ov["extras"].fillna(0).sum())
                dots = float((ov["runs_off_bat"].fillna(0) == 0).sum())
                boundaries = float(((ov["runs_off_bat"] == 4) | (ov["runs_off_bat"] == 6)).sum())

                cum_score = float(inn_df[inn_df["over"] <= over_num]["runs_off_bat"].fillna(0).sum())
                cum_wkts = float(inn_df[inn_df["over"] <= over_num]["wicket_type"].notna().sum()) if "wicket_type" in inn_df.columns else 0.0
                crr = cum_score / over_num if over_num > 0 else 0.0
                phase = 0.0 if over_num <= 6 else (1.0 if over_num <= 15 else 2.0)

                over_feats.append([
                    runs, wickets, extras, dots, boundaries,
                    crr, cum_score, cum_wkts, phase, float(over_num)
                ])

            if len(over_feats) > 0:
                sequences.append(np.array(over_feats))
                targets.append(final_score)

        return sequences, targets

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
