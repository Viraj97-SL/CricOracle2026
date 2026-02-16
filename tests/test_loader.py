"""Tests for CricketDataLoader.

Tests validate the full loading pipeline including column derivation.
"""

import pytest
import pandas as pd
from src.data.loader import CricketDataLoader


class TestCricketDataLoader:
    """Test suite for the data loader."""

    def test_load_from_csv(self, sample_csv):
        """Test that CSV loads and returns non-empty DataFrame."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        assert len(df) > 0

    def test_required_columns_present(self, loaded_data):
        """Test that all required columns exist after loading."""
        required = [
            "match_id", "date", "venue", "batting_team", "bowling_team",
            "over", "batter", "bowler", "runs_batter", "runs_extras",
            "runs_total", "is_wicket", "inning_no", "ball_no",
            "is_four", "is_six", "is_dot", "phase",
        ]
        for col in required:
            assert col in loaded_data.columns, f"Missing column: {col}"

    def test_derived_bowling_team(self, loaded_data):
        """Test that bowling_team was correctly derived."""
        assert "bowling_team" in loaded_data.columns
        # Bowling team should never equal batting team
        mismatches = loaded_data[
            loaded_data["batting_team"] == loaded_data["bowling_team"]
        ]
        assert len(mismatches) == 0, "bowling_team should never equal batting_team"

    def test_derived_inning_no(self, loaded_data):
        """Test that inning_no was correctly derived as 1 or 2."""
        assert loaded_data["inning_no"].isin([1, 2]).all()

    def test_derived_is_wicket(self, loaded_data):
        """Test that is_wicket is binary 0/1."""
        assert loaded_data["is_wicket"].isin([0, 1]).all()

    def test_derived_columns_created(self, loaded_data):
        """Test that helper columns is_four, is_six, is_dot, phase are created."""
        for col in ["is_four", "is_six", "is_dot", "phase"]:
            assert col in loaded_data.columns, f"Missing derived column: {col}"

    def test_over_range_valid(self, loaded_data):
        """Test that overs are in valid T20 range (0-19)."""
        assert loaded_data["over"].min() >= 0
        assert loaded_data["over"].max() <= 19

    def test_no_duplicates(self, loaded_data):
        """Test that deduplication removes exact row duplicates."""
        dedup_cols = ["match_id", "inning_no", "over", "ball_no", "batter", "bowler"]
        dupes = loaded_data.duplicated(subset=dedup_cols).sum()
        assert dupes == 0

    def test_file_not_found_raises(self, tmp_path):
        """Test that missing file raises a clear error."""
        loader = CricketDataLoader(filepath=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_and_clean()

    def test_match_list(self, sample_csv):
        """Test match summary generation."""
        loader = CricketDataLoader(filepath=sample_csv)
        df = loader.load_and_clean(modern_era_only=False)
        matches = loader.get_match_list(df)
        assert len(matches) == 2  # Our fixture has 2 matches
        assert "winner" in matches.columns
