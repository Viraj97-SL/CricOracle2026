"""Feature Pipeline Orchestrator — coordinates all feature families.

This is the single entry point for feature engineering. It chains together
player, team, venue, match context, and weather features into a single
match-level dataset ready for model training.

Think of it like a supply chain assembly line:
Raw materials (ball-by-ball) → Processing stations (feature engines) → Finished goods (match features)

Usage:
    from src.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    match_features = pipeline.run()
    match_features.to_parquet("data/processed/match_features.parquet")
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import settings
from src.data.loader import CricketDataLoader
from src.data.weather import WeatherClient
from src.features.player import PlayerFeatureEngine
from src.features.team import TeamFeatureEngine
from src.features.venue import VenueFeatureEngine
from src.features.match_context import MatchContextEngine
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
        2. Build player profiles (batting + bowling + roles + styles)
        3. Build venue features
        4. Build match context features
        5. Build team aggregate features
        6. (Optional) Add weather features
        7. Merge everything into match-level dataset
        8. Save to processed/

        Args:
            save: If True, saves output to parquet.

        Returns:
            Match-level DataFrame with all features.
        """
        logger.info("=" * 60)
        logger.info("FEATURE PIPELINE: Starting full run")
        logger.info("=" * 60)

        # Step 1: Load data
        loader = CricketDataLoader(self.data_path)
        df = loader.load_and_clean(modern_era_only=self.modern_era_only)

        # Step 2: Player features
        logger.info("--- Step 2: Player Features ---")
        player_engine = PlayerFeatureEngine(df)
        batting_profiles = player_engine.build_batting_profiles()
        bowling_profiles = player_engine.build_bowling_profiles()
        bowling_styles = player_engine.classify_bowling_styles()
        batting_roles = player_engine.classify_batting_roles()

        # Save player profiles for reference
        if save:
            batting_profiles.to_parquet(settings.paths.data_processed / "batting_profiles.parquet")
            bowling_profiles.to_parquet(settings.paths.data_processed / "bowling_profiles.parquet")
            bowling_styles.to_csv(settings.paths.data_processed / "bowling_styles.csv", index=False)
            batting_roles.to_csv(settings.paths.data_processed / "batting_roles.csv", index=False)
            logger.info("Saved player profiles to data/processed/")

        # Step 3: Venue features
        logger.info("--- Step 3: Venue Features ---")
        venue_engine = VenueFeatureEngine(df, bowler_styles=bowling_styles)
        venue_features = venue_engine.build_venue_features()

        if save:
            venue_features.to_parquet(settings.paths.data_processed / "venue_features.parquet")

        # Step 4: Match context
        logger.info("--- Step 4: Match Context Features ---")
        context_engine = MatchContextEngine(df)
        context_features = context_engine.build_context_features()

        # Step 5: Team features
        logger.info("--- Step 5: Team Features ---")
        team_engine = TeamFeatureEngine(df, player_profiles=batting_profiles)
        team_features = team_engine.build_all_team_features()

        # Step 6: Weather (optional)
        if self.include_weather:
            logger.info("--- Step 6: Weather Features ---")
            weather_features = self._build_weather_features(context_features)
        else:
            weather_features = None

        # Step 7: Merge everything
        logger.info("--- Step 7: Merging All Features ---")
        match_features = self._merge_all(
            context_features, venue_features, team_features, weather_features
        )

        # Step 8: Save
        if save:
            output_path = settings.paths.processed_features
            match_features.to_parquet(output_path)
            logger.info(f"Saved match features to {output_path}")

        logger.info("=" * 60)
        logger.info(
            f"PIPELINE COMPLETE: {len(match_features)} matches, "
            f"{len(match_features.columns)} features"
        )
        logger.info("=" * 60)

        return match_features

    def _build_weather_features(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather for all matches (with caching)."""
        client = WeatherClient()
        if not client.is_configured:
            logger.warning("Weather API not configured — skipping weather features")
            return pd.DataFrame(columns=["match_id"])

        weather_records = []
        for _, match in context_df.iterrows():
            weather = client.get_weather(
                venue=match["venue"],
                match_date=str(match["date"].date()) if hasattr(match["date"], "date") else str(match["date"]),
            )
            record = weather.to_dict()
            record["match_id"] = match["match_id"]
            weather_records.append(record)

        return pd.DataFrame(weather_records)

    def _merge_all(
        self,
        context: pd.DataFrame,
        venue: pd.DataFrame,
        team: pd.DataFrame,
        weather: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge all feature families into a single match-level dataset."""
        result = context.copy()

        # Merge venue features
        result = result.merge(venue, on="venue", how="left")

        # Merge team features
        result = result.merge(team, on="match_id", how="left")

        # Merge weather (if available)
        if weather is not None and len(weather) > 0:
            result = result.merge(weather, on="match_id", how="left")

        # Fill NaN with sensible defaults
        numeric_cols = result.select_dtypes(include=["float64", "int64"]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

        return result
