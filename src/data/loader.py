"""Data loader and cleaner for Cricsheet T20I ball-by-ball data.

This is the FIRST module that runs in the pipeline. It handles:
1. Loading raw CSV data
2. Column standardisation (Cricsheet format varies by download date)
3. Type casting and memory optimisation
4. Date parsing and filtering
5. Deduplication
6. Validation via Pydantic schemas

Usage:
    from src.data.loader import CricketDataLoader

    loader = CricketDataLoader()
    df = loader.load_and_clean()
    print(f"Loaded {len(df):,} deliveries from {df['match_id'].nunique()} matches")
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

    # Cricsheet CSV column name variations (they change between file versions)
    COLUMN_MAP = {
        # Common variations → standardised name
        "match_id": "match_id",
        "season": "season",
        "start_date": "date",
        "date": "date",
        "venue": "venue",
        "innings": "inning_no",
        "inning": "inning_no",
        "inning_no": "inning_no",
        "ball": "ball_raw",        # Original ball column (e.g., 0.1, 0.2 ... 19.6)
        "over": "over",
        "ball_no": "ball_no",
        "batting_team": "batting_team",
        "bowling_team": "bowling_team",
        "striker": "batter",
        "batter": "batter",
        "non_striker": "non_striker",
        "bowler": "bowler",
        "runs_off_bat": "runs_batter",
        "runs_batter": "runs_batter",
        "extras": "runs_extras",
        "runs_extras": "runs_extras",
        "runs_total": "runs_total",
        "wides": "wides",
        "noballs": "noballs",
        "byes": "byes",
        "legbyes": "legbyes",
        "penalty": "penalty",
        "wicket_type": "wicket_type",
        "player_dismissed": "player_dismissed",
        "is_wicket": "is_wicket",
    }

    def __init__(self, filepath: Optional[Path] = None):
        """Initialise loader.

        Args:
            filepath: Path to the CSV file. Defaults to config path.
        """
        self.filepath = filepath or settings.paths.ball_by_ball_csv
        self.cricket = CricketConstants()

    def load_and_clean(self, modern_era_only: bool = True) -> pd.DataFrame:
        """Full loading pipeline: load → standardise → clean → validate.

        Args:
            modern_era_only: If True, filter to matches from 2019 onwards.

        Returns:
            Cleaned, validated DataFrame ready for feature engineering.
        """
        logger.info(f"Loading data from {self.filepath}")

        # Step 1: Load raw CSV
        df = self._load_raw()

        # Step 2: Standardise column names
        df = self._standardise_columns(df)

        # Step 3: Parse and fix data types
        df = self._fix_dtypes(df)

        # Step 4: Parse over/ball from raw ball column if needed
        df = self._parse_overs(df)

        # Step 5: Create derived columns
        df = self._create_derived_columns(df)

        # Step 6: Handle duplicates
        df = self._deduplicate(df)

        # Step 7: Filter to modern era if requested
        if modern_era_only:
            df = self._filter_modern_era(df)

        # Step 8: Optimise memory
        df = self._optimise_memory(df)

        # Step 9: Validate a sample
        df, errors = validate_dataframe(df, BallRecord, sample_size=2000)

        logger.info(
            f"Data loaded: {len(df):,} deliveries | "
            f"{df['match_id'].nunique():,} matches | "
            f"{df['date'].min()} to {df['date'].max()}"
        )

        return df

    def _load_raw(self) -> pd.DataFrame:
        """Load CSV with error handling."""
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.filepath}\n"
                f"Please place your Cricsheet CSV in: {settings.paths.data_raw}/"
            )

        df = pd.read_csv(self.filepath, low_memory=False)
        logger.info(f"Raw data: {len(df):,} rows, {len(df.columns)} columns")
        logger.debug(f"Columns found: {list(df.columns)}")
        return df

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map column names to our standard schema."""
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in self.COLUMN_MAP:
                rename_map[col] = self.COLUMN_MAP[col_lower]

        df = df.rename(columns=rename_map)

        # Check for required columns
        required = {"match_id", "batter", "bowler", "venue", "batting_team", "bowling_team"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns after standardisation: {missing}")

        logger.info(f"Standardised {len(rename_map)} column names")
        return df

    def _fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types — dates, integers, categories."""
        # Parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            null_dates = df["date"].isna().sum()
            if null_dates > 0:
                logger.warning(f"Dropped {null_dates} rows with unparseable dates")
                df = df.dropna(subset=["date"])

        # Fix numeric columns
        numeric_cols = ["runs_batter", "runs_extras", "runs_total", "is_wicket"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        return df

    def _parse_overs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse over and ball_no from the raw ball column.

        Cricsheet uses format like 0.1, 0.2, ..., 19.6
        where integer part = over, decimal part = ball number.
        """
        if "over" in df.columns and "ball_no" in df.columns:
            # Already parsed — just ensure types
            df["over"] = df["over"].astype(int)
            df["ball_no"] = df["ball_no"].astype(int)
            return df

        if "ball_raw" in df.columns:
            # Parse from combined column (e.g., 5.3 = over 5, ball 3)
            df["over"] = df["ball_raw"].astype(float).astype(int)
            df["ball_no"] = ((df["ball_raw"] % 1) * 10).round().astype(int)
            logger.info("Parsed over/ball_no from raw ball column")
        elif "over" in df.columns:
            # Have over but not ball_no — create ball_no from within-over sequence
            df["ball_no"] = df.groupby(["match_id", "inning_no", "over"]).cumcount() + 1
            df["over"] = df["over"].astype(int)

        return df

    def _create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create essential derived columns used throughout the pipeline."""
        # is_wicket flag (if not already present)
        if "is_wicket" not in df.columns:
            if "wicket_type" in df.columns:
                df["is_wicket"] = df["wicket_type"].notna().astype(int)
            elif "player_dismissed" in df.columns:
                df["is_wicket"] = df["player_dismissed"].notna().astype(int)
            else:
                df["is_wicket"] = 0
                logger.warning("No wicket information found — setting is_wicket=0 for all")

        # Inning number (if missing)
        if "inning_no" not in df.columns:
            logger.warning("'inning_no' column missing — attempting to derive from data")
            df["inning_no"] = 1  # Fallback

        # runs_total (if missing, calculate from components)
        if "runs_total" not in df.columns:
            df["runs_total"] = df.get("runs_batter", 0) + df.get("runs_extras", 0)

        # Is boundary
        df["is_four"] = (df["runs_batter"] == 4).astype(int)
        df["is_six"] = (df["runs_batter"] == 6).astype(int)
        df["is_dot"] = (df["runs_total"] == 0).astype(int)

        # Phase classification
        df["phase"] = pd.cut(
            df["over"],
            bins=[-1, 5, 14, 19],
            labels=["Powerplay", "Middle", "Death"],
        )

        logger.info("Created derived columns: is_four, is_six, is_dot, phase")
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate deliveries."""
        before = len(df)
        dedup_cols = ["match_id", "inning_no", "over", "ball_no", "batter", "bowler"]
        existing_cols = [c for c in dedup_cols if c in df.columns]
        df = df.drop_duplicates(subset=existing_cols, keep="first")
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed:,} duplicate deliveries")
        return df

    def _filter_modern_era(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to modern T20 era (2019+).

        T20 cricket has evolved significantly — pre-2019 data adds noise.
        """
        if "date" not in df.columns:
            logger.warning("No date column — skipping modern era filter")
            return df

        cutoff = pd.Timestamp(self.cricket.MODERN_ERA_START)
        before = len(df)
        matches_before = df["match_id"].nunique()

        df = df[df["date"] >= cutoff]

        matches_after = df["match_id"].nunique()
        logger.info(
            f"Modern era filter: {matches_before} → {matches_after} matches "
            f"({before - len(df):,} rows removed)"
        )
        return df

    def _optimise_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage with appropriate dtypes."""
        # Categorical columns (low cardinality strings)
        cat_cols = [
            "batter", "bowler", "non_striker", "venue",
            "batting_team", "bowling_team", "phase",
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Small integers
        int8_cols = ["over", "ball_no", "runs_batter", "runs_extras",
                     "runs_total", "is_wicket", "is_four", "is_six", "is_dot",
                     "inning_no"]
        for col in int8_cols:
            if col in df.columns:
                df[col] = df[col].astype("int8")

        before_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Memory usage: {before_mb:.1f} MB")
        return df

    def get_match_list(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get a summary of all matches in the dataset.

        Returns:
            DataFrame with one row per match: match_id, date, venue, team1, team2, winner.
        """
        if df is None:
            df = self.load_and_clean()

        matches = (
            df.groupby("match_id")
            .agg(
                date=("date", "first"),
                venue=("venue", "first"),
                team1=("batting_team", "first"),
                team2=("bowling_team", "first"),
                total_balls=("over", "count"),
            )
            .reset_index()
            .sort_values("date")
        )

        logger.info(f"Match list: {len(matches)} matches")
        return matches
