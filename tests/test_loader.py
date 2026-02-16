"""Tests for data loading and validation."""

import pytest
import pandas as pd
from src.data.loader import CricketDataLoader


class TestCricketDataLoader:
    """Tests for the CricketDataLoader class."""

    def test_load_from_csv(self, sample_csv):
        """Should load and return a DataFrame from CSV."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns_present(self, sample_csv):
        """Loaded data should contain all required columns."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        required = {"match_id", "batter", "bowler", "venue", "batting_team", "bowling_team"}
        assert required.issubset(set(df.columns))

    def test_derived_columns_created(self, sample_csv):
        """Loader should create is_four, is_six, is_dot, phase columns."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        assert "is_four" in df.columns
        assert "is_six" in df.columns
        assert "phase" in df.columns

    def test_over_range_valid(self, sample_csv):
        """All overs should be between 0 and 19."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        assert df["over"].min() >= 0
        assert df["over"].max() <= 19

    def test_no_duplicates(self, sample_csv):
        """Should remove duplicate deliveries."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        dedup_cols = ["match_id", "inning_no", "over", "ball_no", "batter", "bowler"]
        existing = [c for c in dedup_cols if c in df.columns]
        assert df.duplicated(subset=existing).sum() == 0

    def test_file_not_found_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        loader = CricketDataLoader(filepath=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_and_clean()

    def test_match_list(self, sample_csv):
        """get_match_list should return one row per match."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        matches = loader.get_match_list(df)
        assert len(matches) == df["match_id"].nunique()
