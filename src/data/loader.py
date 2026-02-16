"""Data loader and cleaner for Cricsheet T20I ball-by-ball data.

Mapped to YOUR actual CSV columns:
    match_id, date, venue, team1, team2, winner, batting_team,
    over, batter, bowler, non_striker, runs_batter, runs_extra,
    runs_total, wicket_type, player_out

Derives missing columns needed by downstream pipeline:
    bowling_team, inning_no, ball_no, is_wicket, runs_extras

Usage:
    from src.data.loader import CricketDataLoader

    loader = CricketDataLoader()
    df = loader.load_and_clean()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import settings, CricketConstants
from src.utils.logger import logger
from src.utils.validators import validate_dataframe, BallRecord


class CricketDataLoader:
    """Production-grade data loader for Cricsheet T20I data."""

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or settings.paths.ball_by_ball_csv
        self.cricket = CricketConstants()

    def load_and_clean(self, modern_era_only: bool = True) -> pd.DataFrame:
        """Full pipeline: load → standardise → derive → clean → validate."""
        logger.info(f"Loading data from {self.filepath}")

        df = self._load_raw()
        df = self._standardise_columns(df)
        df = self._parse_dates(df)
        df = self._derive_columns(df)
        df = self._create_helper_columns(df)
        df = self._deduplicate(df)

        if modern_era_only:
            df = self._filter_modern_era(df)

        df = self._optimise_memory(df)
        df, errors = validate_dataframe(df, BallRecord, sample_size=2000)

        logger.info(
            f"Data loaded: {len(df):,} deliveries | "
            f"{df['match_id'].nunique():,} matches | "
            f"{df['date'].min()} to {df['date'].max()}"
        )
        return df

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _load_raw(self) -> pd.DataFrame:
        """Load CSV with error handling."""
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.filepath}\n"
                f"Please place your Cricsheet CSV in: {settings.paths.data_raw}/"
            )
        df = pd.read_csv(self.filepath, low_memory=False)
        logger.info(f"Raw data: {len(df):,} rows, {len(df.columns)} columns")
        return df

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename your CSV columns → internal schema names."""
        rename_map = {
            "runs_extra": "runs_extras",
            "player_out": "player_dismissed",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        logger.info(f"Standardised columns: {rename_map}")
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column to datetime."""
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            null_dates = df["date"].isna().sum()
            if null_dates > 0:
                logger.warning(f"Dropped {null_dates} rows with unparseable dates")
                df = df.dropna(subset=["date"])
        return df

    def _derive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive columns missing from your CSV but needed by the pipeline.

        Derives: bowling_team, inning_no, ball_no, is_wicket
        """
        # ---- bowling_team ----
        if "bowling_team" not in df.columns:
            if {"team1", "team2", "batting_team"}.issubset(df.columns):
                df["bowling_team"] = np.where(
                    df["batting_team"] == df["team1"],
                    df["team2"],
                    df["team1"],
                )
                logger.info("Derived 'bowling_team' from team1/team2/batting_team")
            else:
                raise ValueError("Cannot derive bowling_team: need team1, team2, batting_team")

        # ---- inning_no ----
        if "inning_no" not in df.columns:
            first_batting = (
                df.groupby("match_id")["batting_team"]
                .first()
                .reset_index()
                .rename(columns={"batting_team": "first_batting_team"})
            )
            df = df.merge(first_batting, on="match_id", how="left")
            df["inning_no"] = np.where(
                df["batting_team"] == df["first_batting_team"], 1, 2
            )
            df = df.drop(columns=["first_batting_team"])
            logger.info("Derived 'inning_no' from batting order within each match")

        # ---- ball_no ----
        if "ball_no" not in df.columns:
            df["ball_no"] = (
                df.groupby(["match_id", "inning_no", "over"]).cumcount() + 1
            )
            logger.info("Derived 'ball_no' from within-over sequence")

        # ---- is_wicket ----
        if "is_wicket" not in df.columns:
            if "wicket_type" in df.columns:
                df["is_wicket"] = df["wicket_type"].notna().astype(int)
                logger.info(f"Derived 'is_wicket': {df['is_wicket'].sum():,} wickets found")
            else:
                df["is_wicket"] = 0
                logger.warning("No wicket info found — is_wicket set to 0")

        # ---- runs_extras (ensure exists and is clean) ----
        if "runs_extras" not in df.columns:
            df["runs_extras"] = (df["runs_total"] - df["runs_batter"]).clip(lower=0)
            logger.info("Derived 'runs_extras' from runs_total - runs_batter")

        # Ensure over is integer
        df["over"] = df["over"].astype(int)

        return df

    def _create_helper_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature-engineering helper columns."""
        df["is_four"] = (df["runs_batter"] == 4).astype(int)
        df["is_six"] = (df["runs_batter"] == 6).astype(int)
        df["is_dot"] = (df["runs_total"] == 0).astype(int)

        # Phase classification (overs are 0-indexed: 0-5, 6-14, 15-19)
        df["phase"] = pd.cut(
            df["over"],
            bins=[-1, 5, 14, 19],
            labels=["Powerplay", "Middle", "Death"],
        )

        logger.info("Created helper columns: is_four, is_six, is_dot, phase")
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate deliveries."""
        before = len(df)
        dedup_cols = ["match_id", "inning_no", "over", "ball_no", "batter", "bowler"]
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed:,} duplicate deliveries")
        return df

    def _filter_modern_era(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to modern T20 era (2019+)."""
        cutoff = pd.Timestamp(self.cricket.MODERN_ERA_START)
        matches_before = df["match_id"].nunique()
        df = df[df["date"] >= cutoff]
        matches_after = df["match_id"].nunique()
        logger.info(
            f"Modern era filter: {matches_before} → {matches_after} matches"
        )
        return df

    def _optimise_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory with appropriate dtypes."""
        cat_cols = ["batter", "bowler", "non_striker", "venue",
                    "batting_team", "bowling_team", "phase",
                    "winner", "team1", "team2"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        int8_cols = ["over", "ball_no", "runs_batter", "runs_extras",
                     "runs_total", "is_wicket", "is_four", "is_six", "is_dot",
                     "inning_no"]
        for col in int8_cols:
            if col in df.columns:
                df[col] = df[col].astype("int8")

        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Memory usage: {mem_mb:.1f} MB")
        return df

    # =========================================================================
    # PUBLIC HELPERS
    # =========================================================================

    def get_match_list(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get a summary of all matches (one row per match)."""
        if df is None:
            df = self.load_and_clean()

        agg_dict = {
            "date": ("date", "first"),
            "venue": ("venue", "first"),
            "total_balls": ("over", "count"),
        }
        # Add columns that exist in your data
        if "team1" in df.columns:
            agg_dict["team1"] = ("team1", "first")
        if "team2" in df.columns:
            agg_dict["team2"] = ("team2", "first")
        if "winner" in df.columns:
            agg_dict["winner"] = ("winner", "first")

        matches = (
            df.groupby("match_id")
            .agg(**agg_dict)
            .reset_index()
            .sort_values("date")
        )

        logger.info(f"Match list: {len(matches)} matches")
        return matches
