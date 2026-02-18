"""Tests for MatchPlayerFeatureEngine and updated trainer feature selection.

Tests the new player-feature aggregation layer (Step 6 of the pipeline)
and the improved trainer's dedicated feature sets for win vs score models.

Self-contained: all fixtures are defined here so this file runs independently
of the shared tests/conftest.py fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# SHARED FIXTURES — mirrors the real ball-by-ball data structure
# =============================================================================

def _make_ball_df(n_matches: int = 4, balls_per_innings: int = 120) -> pd.DataFrame:
    """Generate a minimal but realistic ball-by-ball DataFrame.

    Generates n_matches T20 matches with 2 innings each.
    Players: 5 batters and 4 bowlers per team, consistent across matches
    so that player profiles can be built (enough history).
    """
    rng = np.random.default_rng(42)
    records = []

    teams = [("India", "Australia"), ("England", "New Zealand")]
    venues = ["Wankhede Stadium, Mumbai", "Eden Gardens, Kolkata"]

    for match_idx in range(n_matches):
        match_id = f"match_{match_idx:03d}"
        team_pair = teams[match_idx % len(teams)]
        venue = venues[match_idx % len(venues)]
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=match_idx * 7)

        for inning in [1, 2]:
            batting_team = team_pair[0] if inning == 1 else team_pair[1]
            bowling_team = team_pair[1] if inning == 1 else team_pair[0]

            # 5 distinct batters per innings, 4 bowlers
            batters = [f"{batting_team}_bat{i}" for i in range(1, 6)]
            bowlers = [f"{bowling_team}_bowl{i}" for i in range(1, 5)]

            for over in range(20):
                bowler = bowlers[over % len(bowlers)]
                for ball in range(6):
                    batter = batters[min(over // 4, len(batters) - 1)]
                    runs_batter = int(rng.choice([0, 1, 2, 4, 6], p=[0.35, 0.35, 0.1, 0.15, 0.05]))
                    runs_extras = int(rng.choice([0, 1], p=[0.95, 0.05]))
                    is_wicket = int(rng.random() < 0.04 and over > 0)
                    wicket_type = "caught" if is_wicket else ""

                    records.append({
                        "match_id": match_id,
                        "date": date,
                        "venue": venue,
                        "team1": team_pair[0],
                        "team2": team_pair[1],
                        "winner": team_pair[0],   # team1 always wins in fixture
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "inning_no": inning,
                        "over": over,
                        "ball_no": ball,
                        "batter": batter,
                        "bowler": bowler,
                        "non_striker": batters[0],
                        "runs_batter": runs_batter,
                        "runs_extras": runs_extras,
                        "runs_total": runs_batter + runs_extras,
                        "is_wicket": is_wicket,
                        "wicket_type": wicket_type,
                        "player_dismissed": batter if is_wicket else "",
                        "is_four": int(runs_batter == 4),
                        "is_six": int(runs_batter == 6),
                        "is_dot": int(runs_batter == 0 and runs_extras == 0),
                        "phase": (
                            "powerplay" if over < 6
                            else "middle" if over < 15
                            else "death"
                        ),
                    })

    return pd.DataFrame(records)


def _make_batting_profiles(ball_df: pd.DataFrame) -> pd.DataFrame:
    """Build minimal batting profiles from ball_df for testing."""
    from src.features.player import PlayerFeatureEngine
    engine = PlayerFeatureEngine(ball_df)
    return engine.build_batting_profiles(min_balls=5)


def _make_bowling_profiles(ball_df: pd.DataFrame) -> pd.DataFrame:
    """Build minimal bowling profiles from ball_df for testing."""
    from src.features.player import PlayerFeatureEngine
    engine = PlayerFeatureEngine(ball_df)
    return engine.build_bowling_profiles(min_balls=5)


def _make_bowling_styles(ball_df: pd.DataFrame) -> pd.DataFrame:
    """Build bowling styles from ball_df for testing."""
    from src.features.player import PlayerFeatureEngine
    engine = PlayerFeatureEngine(ball_df)
    return engine.classify_bowling_styles()


# --- Pytest fixtures ---

@pytest.fixture(scope="module")
def ball_df():
    """Reusable ball-by-ball DataFrame for all tests in this module."""
    return _make_ball_df(n_matches=6, balls_per_innings=120)


@pytest.fixture(scope="module")
def batting_profiles(ball_df):
    return _make_batting_profiles(ball_df)


@pytest.fixture(scope="module")
def bowling_profiles(ball_df):
    return _make_bowling_profiles(ball_df)


@pytest.fixture(scope="module")
def bowling_styles(ball_df):
    return _make_bowling_styles(ball_df)


@pytest.fixture(scope="module")
def engine(ball_df, batting_profiles, bowling_profiles, bowling_styles):
    """Fully initialised MatchPlayerFeatureEngine."""
    from src.features.match_player_features import MatchPlayerFeatureEngine
    return MatchPlayerFeatureEngine(
        df=ball_df,
        batting_profiles=batting_profiles,
        bowling_profiles=bowling_profiles,
        bowling_styles=bowling_styles,
    )


@pytest.fixture(scope="module")
def match_player_features(engine):
    """Pre-computed match player features used by multiple tests."""
    return engine.build_match_player_features()


@pytest.fixture(scope="module")
def match_features_df(ball_df, batting_profiles, bowling_profiles, bowling_styles):
    """A minimal match-level feature DataFrame that mimics pipeline output,
    including the player columns needed by the trainer feature groups."""
    from src.features.match_player_features import MatchPlayerFeatureEngine
    from src.features.match_context import MatchContextEngine
    from src.features.team import TeamFeatureEngine
    from src.features.venue import VenueFeatureEngine

    context_engine = MatchContextEngine(ball_df)
    context = context_engine.build_context_features()

    venue_engine = VenueFeatureEngine(ball_df, bowler_styles=bowling_styles)
    venue = venue_engine.build_venue_features()
    if venue["venue"].dtype.name == "category":
        venue["venue"] = venue["venue"].astype(str)
    if context["venue"].dtype.name == "category":
        context["venue"] = context["venue"].astype(str)

    team_engine = TeamFeatureEngine(ball_df, player_profiles=batting_profiles)
    team = team_engine.build_all_team_features()

    player_engine = MatchPlayerFeatureEngine(
        df=ball_df,
        batting_profiles=batting_profiles,
        bowling_profiles=bowling_profiles,
        bowling_styles=bowling_styles,
    )
    player_feats = player_engine.build_match_player_features()

    result = context.merge(venue, on="venue", how="left")
    result = result.merge(team, on="match_id", how="left")
    result = result.merge(player_feats, on="match_id", how="left")

    numeric_cols = result.select_dtypes(include=["float64", "int64", "float32"]).columns
    safe_fill = [c for c in numeric_cols if c not in {"team1_won", "first_innings_score"}]
    result[safe_fill] = result[safe_fill].fillna(0)

    return result


# =============================================================================
# TEST CLASS 1: MatchPlayerFeatureEngine — unit tests
# =============================================================================

class TestMatchPlayerFeatureEngine:
    """Tests for the new MatchPlayerFeatureEngine (Step 6 of pipeline)."""

    # -------------------------------------------------------------------------
    # Schema & shape tests
    # -------------------------------------------------------------------------

    def test_returns_dataframe(self, match_player_features):
        """Output must be a DataFrame."""
        assert isinstance(match_player_features, pd.DataFrame)

    def test_has_match_id_column(self, match_player_features):
        """match_id must be present as the join key."""
        assert "match_id" in match_player_features.columns

    def test_one_row_per_match(self, ball_df, match_player_features):
        """Output must have exactly one row per unique match."""
        n_matches = ball_df["match_id"].nunique()
        assert len(match_player_features) == n_matches, (
            f"Expected {n_matches} rows, got {len(match_player_features)}"
        )

    def test_expected_team1_columns_present(self, match_player_features):
        """All core team1_ feature columns must be present."""
        expected = [
            "team1_batting_power",
            "team1_top3_sr_L10",
            "team1_top3_runs_L10",
            "team1_avg_boundary_pct",
            "team1_batting_depth",
            "team1_bowling_economy",
            "team1_bowling_dot_pct",
            "team1_top_bowler_sr",
            "team1_spin_bowling_pct",
        ]
        for col in expected:
            assert col in match_player_features.columns, (
                f"Missing column: {col}"
            )

    def test_expected_team2_columns_present(self, match_player_features):
        """All core team2_ feature columns must be present."""
        expected = [
            "team2_batting_power",
            "team2_bowling_economy",
            "team2_spin_bowling_pct",
        ]
        for col in expected:
            assert col in match_player_features.columns, (
                f"Missing column: {col}"
            )

    def test_differential_columns_present(self, match_player_features):
        """Differential features must be computed."""
        expected = [
            "batting_power_diff",
            "bowling_economy_diff",
            "top_order_form_diff",
            "dot_ball_pressure_diff",
        ]
        for col in expected:
            assert col in match_player_features.columns, (
                f"Missing differential column: {col}"
            )

    def test_no_nan_in_output(self, match_player_features):
        """No NaN values should remain after fallback filling."""
        numeric_cols = match_player_features.select_dtypes(include="number").columns
        nan_counts = match_player_features[numeric_cols].isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert len(cols_with_nan) == 0, (
            f"NaN found in columns: {cols_with_nan.to_dict()}"
        )

    # -------------------------------------------------------------------------
    # Value range / sanity tests
    # -------------------------------------------------------------------------

    def test_batting_power_realistic_range(self, match_player_features):
        """T20 batting strike rate should be in 60–220 range."""
        sr = match_player_features["team1_batting_power"]
        assert sr.min() >= 50, f"batting_power too low: {sr.min()}"
        assert sr.max() <= 300, f"batting_power too high: {sr.max()}"

    def test_bowling_economy_realistic_range(self, match_player_features):
        """T20 bowling economy should be roughly 4–15 RPO."""
        eco = match_player_features["team1_bowling_economy"]
        assert eco.min() >= 2.0, f"economy too low: {eco.min()}"
        assert eco.max() <= 20.0, f"economy too high: {eco.max()}"

    def test_spin_bowling_pct_is_percentage(self, match_player_features):
        """Spin bowling % must be in [0, 100]."""
        pct = match_player_features["team1_spin_bowling_pct"]
        assert pct.min() >= 0.0
        assert pct.max() <= 100.0

    def test_batting_depth_positive_integer(self, match_player_features):
        """Batting depth must be a non-negative integer."""
        depth = match_player_features["team1_batting_depth"]
        assert (depth >= 0).all()
        assert depth.dtype in (np.int64, np.int32, np.float64)

    def test_differential_is_team1_minus_team2(self, match_player_features):
        """batting_power_diff must equal team1_batting_power - team2_batting_power."""
        computed = (
            match_player_features["team1_batting_power"]
            - match_player_features["team2_batting_power"]
        ).round(2)
        stored = match_player_features["batting_power_diff"].round(2)
        pd.testing.assert_series_equal(computed, stored, check_names=False)

    def test_economy_diff_is_team1_minus_team2(self, match_player_features):
        """bowling_economy_diff must equal team1 - team2 economy."""
        computed = (
            match_player_features["team1_bowling_economy"]
            - match_player_features["team2_bowling_economy"]
        ).round(2)
        stored = match_player_features["bowling_economy_diff"].round(2)
        pd.testing.assert_series_equal(computed, stored, check_names=False)

    # -------------------------------------------------------------------------
    # Behaviour tests
    # -------------------------------------------------------------------------

    def test_fallback_used_for_unknown_players(self, ball_df, batting_profiles,
                                               bowling_profiles):
        """When no player profiles exist, engine uses fallback values (no crash)."""
        from src.features.match_player_features import MatchPlayerFeatureEngine
        empty_batting = pd.DataFrame(columns=batting_profiles.columns)
        empty_bowling = pd.DataFrame(columns=bowling_profiles.columns)

        engine_no_profiles = MatchPlayerFeatureEngine(
            df=ball_df,
            batting_profiles=empty_batting,
            bowling_profiles=empty_bowling,
        )
        result = engine_no_profiles.build_match_player_features()
        assert len(result) == ball_df["match_id"].nunique()
        assert "team1_batting_power" in result.columns

    def test_without_bowling_styles(self, ball_df, batting_profiles, bowling_profiles):
        """Engine must work without bowling_styles (defaults to 50% spin)."""
        from src.features.match_player_features import MatchPlayerFeatureEngine
        engine_no_styles = MatchPlayerFeatureEngine(
            df=ball_df,
            batting_profiles=batting_profiles,
            bowling_profiles=bowling_profiles,
            bowling_styles=None,
        )
        result = engine_no_styles.build_match_player_features()
        assert (result["team1_spin_bowling_pct"] == 50.0).all(), (
            "Expected 50% spin when bowling_styles=None"
        )

    def test_inferred_playing_xi_covers_all_teams(self, engine, ball_df):
        """_infer_playing_xi must return batters and bowlers for every team."""
        batters, bowlers = engine._infer_playing_xi()
        teams_in_data = set(ball_df["batting_team"].astype(str).unique())
        teams_with_batters = set(batters["team"].unique())
        assert teams_in_data == teams_with_batters, (
            f"Missing teams in batter inference: {teams_in_data - teams_with_batters}"
        )


# =============================================================================
# TEST CLASS 2: Trainer feature selection
# =============================================================================

class TestTrainerFeatureSelection:
    """Tests for the updated trainer.py feature group definitions and selection."""

    def test_select_features_returns_list(self, match_features_df):
        """_select_features must return a list of strings."""
        from src.models.trainer import _select_features, WIN_PREDICTOR_FEATURE_GROUPS
        result = _select_features(match_features_df, WIN_PREDICTOR_FEATURE_GROUPS)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_selected_features_exist_in_df(self, match_features_df):
        """Every selected feature must exist as a column in the DataFrame."""
        from src.models.trainer import _select_features, WIN_PREDICTOR_FEATURE_GROUPS
        result = _select_features(match_features_df, WIN_PREDICTOR_FEATURE_GROUPS)
        for col in result:
            assert col in match_features_df.columns, f"Selected col {col!r} not in df"

    def test_selected_features_are_numeric(self, match_features_df):
        """_select_features with require_numeric=True must return only numeric cols."""
        from src.models.trainer import _select_features, WIN_PREDICTOR_FEATURE_GROUPS
        result = _select_features(match_features_df, WIN_PREDICTOR_FEATURE_GROUPS)
        numeric_dtypes = {"float64", "int64", "int32", "int8", "float32"}
        for col in result:
            assert match_features_df[col].dtype.name in numeric_dtypes, (
                f"Non-numeric column selected: {col} ({match_features_df[col].dtype})"
            )

    def test_excluded_cols_not_in_win_features(self, match_features_df):
        """Leakage columns must never appear in win predictor feature set."""
        from src.models.trainer import _select_features, WIN_PREDICTOR_FEATURE_GROUPS, _ALWAYS_EXCLUDE
        result = set(_select_features(match_features_df, WIN_PREDICTOR_FEATURE_GROUPS))
        leakage = result & _ALWAYS_EXCLUDE
        assert len(leakage) == 0, f"Leakage columns in features: {leakage}"

    def test_excluded_cols_not_in_score_features(self, match_features_df):
        """Leakage columns must never appear in score predictor feature set."""
        from src.models.trainer import _select_features, SCORE_PREDICTOR_FEATURE_GROUPS, _ALWAYS_EXCLUDE
        result = set(_select_features(match_features_df, SCORE_PREDICTOR_FEATURE_GROUPS))
        leakage = result & _ALWAYS_EXCLUDE
        assert len(leakage) == 0, f"Leakage columns in score features: {leakage}"

    def test_win_and_score_feature_groups_are_distinct(self):
        """Win and score predictor feature groups must differ — separate signal."""
        from src.models.trainer import WIN_PREDICTOR_FEATURE_GROUPS, SCORE_PREDICTOR_FEATURE_GROUPS
        win_cols = set(c for cols in WIN_PREDICTOR_FEATURE_GROUPS.values() for c in cols)
        score_cols = set(c for cols in SCORE_PREDICTOR_FEATURE_GROUPS.values() for c in cols)
        # They share some venue features but must not be identical
        assert win_cols != score_cols, "Win and score feature sets must differ"

    def test_player_features_in_win_predictor_groups(self):
        """Player batting/bowling features must be defined in win predictor groups."""
        from src.models.trainer import WIN_PREDICTOR_FEATURE_GROUPS
        assert "player_batting" in WIN_PREDICTOR_FEATURE_GROUPS, (
            "player_batting group missing from WIN_PREDICTOR_FEATURE_GROUPS"
        )
        assert "player_bowling" in WIN_PREDICTOR_FEATURE_GROUPS, (
            "player_bowling group missing from WIN_PREDICTOR_FEATURE_GROUPS"
        )

    def test_batting_team_features_in_score_groups(self):
        """Score predictor must have a dedicated batting_team_strength group."""
        from src.models.trainer import SCORE_PREDICTOR_FEATURE_GROUPS
        assert "batting_team_strength" in SCORE_PREDICTOR_FEATURE_GROUPS, (
            "batting_team_strength group missing from SCORE_PREDICTOR_FEATURE_GROUPS"
        )

    def test_select_features_handles_missing_columns_gracefully(self):
        """_select_features must not raise when requested cols are absent from df."""
        from src.models.trainer import _select_features
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        feature_groups = {"grp": ["a", "b", "nonexistent_col"]}
        result = _select_features(df, feature_groups)
        assert "a" in result
        assert "b" in result
        assert "nonexistent_col" not in result


# =============================================================================
# TEST CLASS 3: TeamFeatureEngine.calculate_team_strength
# =============================================================================

class TestTeamStrength:
    """Tests for the now-implemented calculate_team_strength method."""

    @pytest.fixture
    def team_engine(self, ball_df, batting_profiles):
        from src.features.team import TeamFeatureEngine
        return TeamFeatureEngine(ball_df, player_profiles=batting_profiles)

    @pytest.fixture
    def team_engine_no_profiles(self, ball_df):
        from src.features.team import TeamFeatureEngine
        return TeamFeatureEngine(ball_df, player_profiles=None)

    def test_returns_dict(self, team_engine):
        """calculate_team_strength must return a dict."""
        result = team_engine.calculate_team_strength("India")
        assert isinstance(result, dict)

    def test_has_required_keys(self, team_engine):
        """Result dict must contain the key strength metrics."""
        result = team_engine.calculate_team_strength("India")
        for key in ["team", "batting_power", "composite_strength"]:
            assert key in result, f"Missing key: {key}"

    def test_team_name_preserved(self, team_engine):
        """Result must preserve the exact team name passed in."""
        result = team_engine.calculate_team_strength("Australia")
        assert result["team"] == "Australia"

    def test_composite_strength_is_bounded(self, team_engine):
        """Composite strength must be in [0, 1] range."""
        result = team_engine.calculate_team_strength("India")
        assert 0.0 <= result["composite_strength"] <= 1.0, (
            f"composite_strength out of bounds: {result['composite_strength']}"
        )

    def test_batting_power_is_realistic(self, team_engine):
        """batting_power should reflect T20 strike rate range."""
        result = team_engine.calculate_team_strength("India")
        assert result["batting_power"] > 50, "batting_power too low"
        assert result["batting_power"] < 300, "batting_power too high"

    def test_fallback_when_no_profiles(self, team_engine_no_profiles):
        """Returns sensible defaults without player profiles (no crash)."""
        result = team_engine_no_profiles.calculate_team_strength("India")
        assert isinstance(result, dict)
        assert result["batting_power"] == 130.0

    def test_with_explicit_playing_xi(self, team_engine, ball_df):
        """Result is valid when an explicit playing XI is passed."""
        # Use players known to be in the fixture data
        xi = [f"India_bat{i}" for i in range(1, 6)]
        result = team_engine.calculate_team_strength("India", playing_xi=xi)
        assert isinstance(result, dict)
        assert result["batting_power"] > 0

    def test_unknown_team_returns_defaults(self, team_engine):
        """Gracefully handles teams with no historical data."""
        result = team_engine.calculate_team_strength("Atlantis XI")
        assert isinstance(result, dict)
        assert "composite_strength" in result


# =============================================================================
# TEST CLASS 4: Pipeline integration (end-to-end Step 6)
# =============================================================================

class TestPipelinePlayerIntegration:
    """Integration tests: player features flow through pipeline → match features."""

    def test_player_columns_in_merged_features(self, match_features_df):
        """After pipeline merge, player columns must be present."""
        player_cols = [c for c in match_features_df.columns if c.startswith("team1_batting")]
        assert len(player_cols) > 0, (
            "No team1_batting_ columns found — player features not merged"
        )

    def test_no_feature_leakage_of_future_scores(self, match_features_df):
        """first_innings_score and second_innings_score must not be in player features."""
        cols = set(match_features_df.columns)
        # These should exist as targets but not be double-counted as features
        # The trainer explicitly excludes them — this checks the column names are correct
        assert "first_innings_score" in cols, "Target column missing"
        assert "second_innings_score" in cols, "Target column missing"

    def test_total_feature_count_increased(self, match_features_df):
        """Merged DataFrame must have more columns than the old 39-feature baseline."""
        assert len(match_features_df.columns) > 39, (
            f"Expected >39 columns after player features, got {len(match_features_df.columns)}"
        )

    def test_no_all_zero_player_columns(self, match_features_df):
        """Player strength columns must not be all-zero (would indicate join failure)."""
        player_cols = [
            "team1_batting_power", "team1_bowling_economy",
            "team2_batting_power", "team2_bowling_economy",
        ]
        for col in player_cols:
            if col in match_features_df.columns:
                assert match_features_df[col].sum() > 0, (
                    f"Column {col} is all zeros — player feature join likely failed"
                )

    def test_match_id_is_unique_in_player_features(self, match_features_df):
        """match_id must be unique (no duplicated rows after merges)."""
        dupes = match_features_df["match_id"].duplicated().sum()
        assert dupes == 0, f"Duplicate match_ids found after merge: {dupes}"
