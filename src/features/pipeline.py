"""Feature Pipeline Orchestrator — coordinates all feature families.

Single entry point for feature engineering. Chains together:
    Raw data → Player profiles → Venue stats → Match context → Team features
             → Match player features (NEW) → Final dataset

Think of it like a supply chain assembly line:
    Raw materials (ball-by-ball) → Processing stations → Finished goods (match features)

Usage:
    from src.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    match_features = pipeline.run()
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import settings
from src.data.loader import CricketDataLoader
from src.features.player import PlayerFeatureEngine
from src.features.team import TeamFeatureEngine
from src.features.venue import VenueFeatureEngine
from src.features.match_context import MatchContextEngine
from src.features.match_player_features import MatchPlayerFeatureEngine
from src.utils.logger import logger


class FeaturePipeline:
    """Orchestrates the full feature engineering pipeline."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        include_weather: bool = False,
        modern_era_only: bool = True,
    ):
        self.data_path = data_path
        self.include_weather = include_weather
        self.modern_era_only = modern_era_only

    def run(self, save: bool = True) -> pd.DataFrame:
        """Execute the full feature pipeline.

        Steps:
        1. Load and clean raw data
        2. Build player profiles (batting + bowling + styles + roles)
        3. Build venue features (scoring, spin/pace ratios, chase rates)
        4. Build match context (teams, winner, date, toss, innings scores)
        5. Build team features (h2h, form, experience)
        6. Build match player features (team batting/bowling strength) ← NEW
        7. Merge everything into match-level dataset
        8. Save to data/processed/

        Returns:
            Match-level DataFrame with all features + target variable (team1_won).
        """
        logger.info("=" * 60)
        logger.info("FEATURE PIPELINE: Starting full run")
        logger.info("=" * 60)

        # ============================
        # Step 1: Load data
        # ============================
        loader = CricketDataLoader(self.data_path)
        df = loader.load_and_clean(modern_era_only=self.modern_era_only)

        # ============================
        # Step 2: Player features
        # ============================
        logger.info("--- Step 2: Player Features ---")
        player_engine = PlayerFeatureEngine(df)

        batting_profiles = player_engine.build_batting_profiles()
        bowling_profiles = player_engine.build_bowling_profiles()
        bowling_styles = player_engine.classify_bowling_styles()
        batting_roles = player_engine.classify_batting_roles()

        if save:
            batting_profiles.to_parquet(
                settings.paths.data_processed / "batting_profiles.parquet"
            )
            bowling_profiles.to_parquet(
                settings.paths.data_processed / "bowling_profiles.parquet"
            )
            bowling_styles.to_csv(
                settings.paths.data_processed / "bowling_styles.csv", index=False
            )
            batting_roles.to_csv(
                settings.paths.data_processed / "batting_roles.csv", index=False
            )
            logger.info("Saved player profiles to data/processed/")

        # ============================
        # Step 3: Venue features
        # ============================
        logger.info("--- Step 3: Venue Features ---")
        venue_engine = VenueFeatureEngine(df, bowler_styles=bowling_styles)
        venue_features = venue_engine.build_venue_features()

        if save:
            venue_features.to_parquet(
                settings.paths.data_processed / "venue_features.parquet"
            )

        # ============================
        # Step 4: Match context
        # ============================
        logger.info("--- Step 4: Match Context Features ---")
        context_engine = MatchContextEngine(df)
        context_features = context_engine.build_context_features()

        # ============================
        # Step 5: Team features
        # ============================
        logger.info("--- Step 5: Team Features ---")
        team_engine = TeamFeatureEngine(df, player_profiles=batting_profiles)
        team_features = team_engine.build_all_team_features()

        # ============================
        # Step 6: Match player features (NEW)
        # ============================
        logger.info("--- Step 6: Match Player Features ---")
        match_player_engine = MatchPlayerFeatureEngine(
            df=df,
            batting_profiles=batting_profiles,
            bowling_profiles=bowling_profiles,
            bowling_styles=bowling_styles,
        )
        match_player_features = match_player_engine.build_match_player_features()

        if save:
            match_player_features.to_parquet(
                settings.paths.data_processed / "match_player_features.parquet"
            )
            logger.info("Saved match player features to data/processed/")

        # ============================
        # Step 7: Merge everything
        # ============================
        logger.info("--- Step 7: Merging All Features ---")
        match_features = self._merge_all(
            context=context_features,
            venue=venue_features,
            team=team_features,
            player=match_player_features,
        )

        # ============================
        # Step 8: Save
        # ============================
        if save:
            output_path = settings.paths.processed_features
            match_features.to_parquet(output_path)
            logger.info(f"Saved match features to {output_path}")

        logger.info("=" * 60)
        logger.info(
            f"PIPELINE COMPLETE: {len(match_features)} matches × "
            f"{len(match_features.columns)} features"
        )
        logger.info(
            f"Target variable 'team1_won': "
            f"{match_features['team1_won'].sum()}/{len(match_features)} "
            f"({match_features['team1_won'].mean()*100:.1f}%)"
        )
        logger.info(f"Features: {sorted(match_features.columns.tolist())}")
        logger.info("=" * 60)

        return match_features

    def _merge_all(
        self,
        context: pd.DataFrame,
        venue: pd.DataFrame,
        team: pd.DataFrame,
        player: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Merge all feature families into a single match-level dataset.

        Context is the backbone — it has match_id, venue, teams, winner, target.
        Venue features merge on venue name.
        Team features merge on match_id.
        Player features merge on match_id.  ← NEW
        """
        result = context.copy()

        # Merge venue features on venue name
        if result["venue"].dtype.name == "category":
            result["venue"] = result["venue"].astype(str)
        if venue["venue"].dtype.name == "category":
            venue["venue"] = venue["venue"].astype(str)

        result = result.merge(venue, on="venue", how="left")

        # Merge team features on match_id
        result = result.merge(team, on="match_id", how="left")

        # Merge player features on match_id (NEW)
        if player is not None and len(player) > 0:
            result = result.merge(player, on="match_id", how="left")
            logger.info(
                f"Player features joined: {len(player.columns) - 1} new columns"
            )

        # Fill NaN in numeric columns with 0 (sensible for missing features)
        numeric_cols = result.select_dtypes(
            include=["float64", "int64", "float32"]
        ).columns
        # Don't fill target or score columns with 0
        safe_fill = [
            c for c in numeric_cols
            if c not in {
                "team1_won", "first_innings_score", "second_innings_score",
            }
        ]
        result[safe_fill] = result[safe_fill].fillna(0)

        return result
