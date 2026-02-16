"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from src.features.player import PlayerFeatureEngine


class TestPlayerFeatureEngine:
    """Tests for player feature engineering."""

    def test_batting_profiles_returns_dataframe(self, sample_ball_data):
        """Should return a DataFrame with batting stats."""
        engine = PlayerFeatureEngine(sample_ball_data)
        profiles = engine.build_batting_profiles(min_balls=5)
        assert isinstance(profiles, pd.DataFrame)
        assert len(profiles) > 0

    def test_batting_profiles_has_key_columns(self, sample_ball_data):
        """Batting profiles should contain strike rate, average, boundary %."""
        engine = PlayerFeatureEngine(sample_ball_data)
        profiles = engine.build_batting_profiles(min_balls=5)
        expected_cols = {"batter", "strike_rate", "batting_avg", "boundary_pct"}
        assert expected_cols.issubset(set(profiles.columns))

    def test_strike_rate_is_positive(self, sample_ball_data):
        """Strike rates should be positive numbers."""
        engine = PlayerFeatureEngine(sample_ball_data)
        profiles = engine.build_batting_profiles(min_balls=5)
        assert (profiles["strike_rate"] >= 0).all()

    def test_bowling_profiles_returns_dataframe(self, sample_ball_data):
        """Should return a DataFrame with bowling stats."""
        engine = PlayerFeatureEngine(sample_ball_data)
        profiles = engine.build_bowling_profiles(min_balls=5)
        assert isinstance(profiles, pd.DataFrame)

    def test_classify_bowling_styles(self, sample_ball_data):
        """Bowling styles should be either Spin or Pace."""
        engine = PlayerFeatureEngine(sample_ball_data)
        styles = engine.classify_bowling_styles()
        if len(styles) > 0:
            assert set(styles["bowling_style"].unique()).issubset({"Spin", "Pace"})

    def test_batting_roles_classification(self, sample_ball_data):
        """Batting roles should be valid categories."""
        engine = PlayerFeatureEngine(sample_ball_data)
        roles = engine.classify_batting_roles()
        valid_roles = {"Opener", "Top Order", "Middle Order", "Finisher"}
        if len(roles) > 0:
            assert set(roles["batting_role"].unique()).issubset(valid_roles)

    def test_min_balls_filter(self, sample_ball_data):
        """Setting high min_balls should filter out players."""
        engine = PlayerFeatureEngine(sample_ball_data)
        all_players = engine.build_batting_profiles(min_balls=1)
        filtered = engine.build_batting_profiles(min_balls=500)
        assert len(filtered) <= len(all_players)


class TestVenueFeatures:
    """Tests for venue feature engineering."""

    def test_venue_features_returns_dataframe(self, sample_ball_data):
        """Should return venue-level stats."""
        from src.features.venue import VenueFeatureEngine
        engine = VenueFeatureEngine(sample_ball_data)
        features = engine.build_venue_features()
        assert isinstance(features, pd.DataFrame)
        assert "venue" in features.columns

    def test_subcontinent_detection(self, sample_ball_data):
        """Wankhede should be detected as subcontinent."""
        from src.features.venue import VenueFeatureEngine
        engine = VenueFeatureEngine(sample_ball_data)
        assert engine._is_subcontinent("Wankhede Stadium, Mumbai") is True
        assert engine._is_subcontinent("Melbourne Cricket Ground") is False
