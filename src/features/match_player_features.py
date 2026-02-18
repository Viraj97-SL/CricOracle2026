"""Match-level player feature aggregation.

Bridges the gap between individual player profiles (batting/bowling stats)
and the match-level feature matrix used by the models.

Core logic (think of it like a supply chain BOM aggregation):
    Individual player stats (components) → Playing XI (assembly) → Team strength (finished good)

For each match, we:
    1. Infer the playing XI from who actually batted/bowled (ball-by-ball evidence)
    2. Look up each player's historical profile (batting avg, SR, economy, form)
    3. Aggregate into team-level batting and bowling power scores
    4. Pivot into match-level features with team1_ / team2_ prefixes

Usage:
    from src.features.match_player_features import MatchPlayerFeatureEngine

    engine = MatchPlayerFeatureEngine(
        df=ball_df,
        batting_profiles=batting_profiles,
        bowling_profiles=bowling_profiles,
        bowling_styles=bowling_styles,
    )
    match_player_feats = engine.build_match_player_features()
    # Returns: DataFrame with match_id + 16 team strength columns
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.config import CricketConstants
from src.utils.logger import logger


class MatchPlayerFeatureEngine:
    """Aggregates player profiles into match-level team strength features."""

    # How many top batters/bowlers to consider per team per match
    TOP_BATTERS = 6
    TOP_BOWLERS = 5

    def __init__(
        self,
        df: pd.DataFrame,
        batting_profiles: pd.DataFrame,
        bowling_profiles: pd.DataFrame,
        bowling_styles: Optional[pd.DataFrame] = None,
    ):
        """Initialise with ball-by-ball data and pre-built player profiles.

        Args:
            df: Cleaned ball-by-ball DataFrame from CricketDataLoader.
            batting_profiles: Output of PlayerFeatureEngine.build_batting_profiles().
            bowling_profiles: Output of PlayerFeatureEngine.build_bowling_profiles().
            bowling_styles: Output of PlayerFeatureEngine.classify_bowling_styles().
        """
        self.df = df.copy()
        self.batting_profiles = batting_profiles.copy()
        self.bowling_profiles = bowling_profiles.copy()
        self.bowling_styles = bowling_styles.copy() if bowling_styles is not None else None
        self.cricket = CricketConstants()

        # Pre-process: ensure string dtypes for join keys
        for col in ["team1", "team2"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)

        self._precompute_match_teams()

    def _precompute_match_teams(self) -> None:
        """Cache match-level team1/team2 mapping for efficient lookup."""
        self.match_teams = (
            self.df.groupby("match_id")
            .agg(
                team1=("team1", "first"),
                team2=("team2", "first"),
            )
            .reset_index()
        )
        self.match_teams["team1"] = self.match_teams["team1"].astype(str)
        self.match_teams["team2"] = self.match_teams["team2"].astype(str)

    # =========================================================================
    # PUBLIC INTERFACE
    # =========================================================================

    def build_match_player_features(self) -> pd.DataFrame:
        """Main entry point. Build all player-derived match features.

        Returns:
            DataFrame indexed by match_id with columns:
                team1_batting_power, team1_top3_sr_L10, team1_avg_boundary_pct,
                team1_bowling_economy, team1_dot_ball_pct, team1_spin_bowling_pct,
                team1_top_bowler_sr, team1_batting_depth,
                (same set for team2_),
                batting_power_diff, bowling_economy_diff
        """
        logger.info("Building match-level player features...")

        # Step 1: Who batted and bowled in each match (infer playing XI)
        match_batters, match_bowlers = self._infer_playing_xi()

        # Step 2: Aggregate batting profiles → team batting strength
        batting_feats = self._aggregate_team_batting(match_batters)

        # Step 3: Aggregate bowling profiles → team bowling strength
        bowling_feats = self._aggregate_team_bowling(match_bowlers)

        # Step 4: Merge batting + bowling per (match_id, team)
        team_feats = batting_feats.merge(bowling_feats, on=["match_id", "team"], how="outer")

        # Step 5: Pivot to match level — team1_ and team2_ columns
        result = self._pivot_to_match_level(team_feats)

        # Step 6: Differential features (team1 advantage over team2)
        result = self._add_differential_features(result)

        n_matches = len(result)
        n_features = len(result.columns) - 1  # exclude match_id
        logger.info(
            f"Built player features for {n_matches} matches × {n_features} player features"
        )
        return result

    # =========================================================================
    # STEP 1: INFER PLAYING XI
    # =========================================================================

    def _infer_playing_xi(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Infer who batted and bowled in each match from ball-by-ball evidence.

        Returns:
            match_batters: DataFrame of (match_id, team, batter, balls_faced)
            match_bowlers: DataFrame of (match_id, team, bowler, balls_bowled)
        """
        # Batters: sorted by balls faced within each match×team
        match_batters = (
            self.df.groupby(["match_id", "batting_team", "batter"])
            .agg(balls_faced=("over", "count"))
            .reset_index()
            .rename(columns={"batting_team": "team", "batter": "player"})
        )
        match_batters["team"] = match_batters["team"].astype(str)

        # Bowlers: sorted by balls bowled within each match×team
        match_bowlers = (
            self.df.groupby(["match_id", "bowling_team", "bowler"])
            .agg(balls_bowled=("over", "count"))
            .reset_index()
            .rename(columns={"bowling_team": "team", "bowler": "player"})
        )
        match_bowlers["team"] = match_bowlers["team"].astype(str)

        return match_batters, match_bowlers

    # =========================================================================
    # STEP 2: AGGREGATE BATTING FEATURES
    # =========================================================================

    def _aggregate_team_batting(self, match_batters: pd.DataFrame) -> pd.DataFrame:
        """Aggregate batting profiles into team-level batting strength per match.

        Strategy:
        - Take top N batters by balls faced (most involved = most representative)
        - Weight their stats by balls faced (more involvement = higher weight)
        - Fall back to global average for players with no profile

        Features produced:
        - batting_power: weighted avg career strike rate of top 6
        - top3_sr_L10: exponentially-weighted avg SR of top 3 batters (recent form)
        - top3_runs_L10: avg recent runs of top 3 batters
        - avg_boundary_pct: avg boundary % of top 6
        - batting_depth: count of profiled batters (team experience proxy)
        """
        # Join batting profiles
        bp = self.batting_profiles.copy()
        bp["batter"] = bp["batter"].astype(str)

        # Fallback values (global medians)
        fallback_sr = bp["strike_rate"].median() if len(bp) > 0 else 130.0
        fallback_sr_L10 = bp["batting_sr_L10"].median() if "batting_sr_L10" in bp.columns else 125.0
        fallback_runs_L10 = bp["batting_avg_L10"].median() if "batting_avg_L10" in bp.columns else 22.0
        fallback_boundary = bp["boundary_pct"].median() if len(bp) > 0 else 12.0

        records = []
        for (match_id, team), group in match_batters.groupby(["match_id", "team"]):
            # Top N batters by balls faced in this match
            top_batters = group.nlargest(self.TOP_BATTERS, "balls_faced")

            # Rename match-level 'balls_faced' → 'match_balls_faced' BEFORE merge
            # to avoid collision with career-level 'balls_faced' in batting_profiles
            top_batters = top_batters.rename(columns={"balls_faced": "match_balls_faced"})

            # Join profiles
            merged = top_batters.merge(
                bp.rename(columns={"batter": "player"}),
                on="player",
                how="left",
            )

            # Fill missing profiles with fallback
            merged["strike_rate"] = merged["strike_rate"].fillna(fallback_sr)
            if "batting_sr_L10" not in merged.columns:
                merged["batting_sr_L10"] = fallback_sr_L10
            else:
                merged["batting_sr_L10"] = merged["batting_sr_L10"].fillna(fallback_sr_L10)
            if "batting_avg_L10" not in merged.columns:
                merged["batting_avg_L10"] = fallback_runs_L10
            else:
                merged["batting_avg_L10"] = merged["batting_avg_L10"].fillna(fallback_runs_L10)
            merged["boundary_pct"] = merged["boundary_pct"].fillna(fallback_boundary)

            # Weights = match balls faced (normalized) — use renamed column
            weights = merged["match_balls_faced"].values.astype(float)
            total_weight = weights.sum()
            if total_weight == 0:
                weights = np.ones(len(merged)) / max(len(merged), 1)
            else:
                weights = weights / total_weight

            # Feature: batting power (weighted avg career SR)
            batting_power = float(np.average(merged["strike_rate"].values, weights=weights))

            # Feature: top 3 recent form
            top3 = merged.head(3)
            if len(top3) > 0:
                top3_sr_L10 = top3["batting_sr_L10"].mean() if "batting_sr_L10" in top3.columns else fallback_sr_L10
                top3_runs_L10 = top3["batting_avg_L10"].mean() if "batting_avg_L10" in top3.columns else fallback_runs_L10
            else:
                top3_sr_L10 = fallback_sr_L10
                top3_runs_L10 = fallback_runs_L10

            # Feature: avg boundary %
            avg_boundary_pct = float(merged["boundary_pct"].mean())

            # Feature: batting depth (how many top 6 have profiles = experienced XI)
            batting_depth = int(merged["strike_rate"].notna().sum())

            records.append({
                "match_id": match_id,
                "team": team,
                "batting_power": round(batting_power, 2),
                "top3_sr_L10": round(float(top3_sr_L10), 2),
                "top3_runs_L10": round(float(top3_runs_L10), 2),
                "avg_boundary_pct": round(avg_boundary_pct, 2),
                "batting_depth": batting_depth,
            })

        return pd.DataFrame(records)

    # =========================================================================
    # STEP 3: AGGREGATE BOWLING FEATURES
    # =========================================================================

    def _aggregate_team_bowling(self, match_bowlers: pd.DataFrame) -> pd.DataFrame:
        """Aggregate bowling profiles into team-level bowling strength per match.

        Features produced:
        - bowling_economy: weighted avg economy of top 5 bowlers
        - bowling_dot_pct: avg dot ball % (pressure bowling indicator)
        - top_bowler_sr: avg bowling strike rate of top 5 (wicket-taking ability)
        - spin_bowling_pct: % of bowling that is spin (venue-strategy alignment)
        """
        bp = self.bowling_profiles.copy()
        bp["bowler"] = bp["bowler"].astype(str)

        # Fallback values
        fallback_economy = bp["economy"].median() if len(bp) > 0 else 8.5
        fallback_dot_pct = bp["dot_ball_pct"].median() if len(bp) > 0 else 35.0
        fallback_bowling_sr = bp["bowling_sr"].replace(999, np.nan).median() if len(bp) > 0 else 18.0

        # Bowling styles lookup (optional)
        style_map = {}
        if self.bowling_styles is not None:
            style_map = dict(
                zip(
                    self.bowling_styles["bowler"].astype(str),
                    self.bowling_styles["bowling_style"],
                )
            )

        records = []
        for (match_id, team), group in match_bowlers.groupby(["match_id", "team"]):
            top_bowlers = group.nlargest(self.TOP_BOWLERS, "balls_bowled")

            # Rename match-level 'balls_bowled' → 'match_balls_bowled' BEFORE merge
            # to avoid collision with career-level 'balls_bowled' in bowling_profiles
            top_bowlers = top_bowlers.rename(columns={"balls_bowled": "match_balls_bowled"})

            merged = top_bowlers.merge(
                bp.rename(columns={"bowler": "player"}),
                on="player",
                how="left",
            )

            merged["economy"] = merged["economy"].fillna(fallback_economy)
            merged["dot_ball_pct"] = merged["dot_ball_pct"].fillna(fallback_dot_pct)
            # Replace 999 (no wickets) with fallback for SR
            merged["bowling_sr"] = (
                merged["bowling_sr"]
                .replace(999, np.nan)
                .fillna(fallback_bowling_sr)
            )

            # Weights = match balls bowled (use renamed column to avoid ambiguity)
            weights = merged["match_balls_bowled"].values.astype(float)
            total_weight = weights.sum()
            if total_weight == 0:
                weights = np.ones(len(merged)) / max(len(merged), 1)
            else:
                weights = weights / total_weight

            # Feature: bowling economy (weighted)
            bowling_economy = float(np.average(merged["economy"].values, weights=weights))

            # Feature: dot ball % (weighted)
            bowling_dot_pct = float(np.average(merged["dot_ball_pct"].values, weights=weights))

            # Feature: top bowler SR (avg of top 5)
            top_bowler_sr = float(merged["bowling_sr"].mean())

            # Feature: spin bowling %
            if style_map:
                spin_count = sum(
                    1 for p in merged["player"]
                    if style_map.get(str(p), "Pace") == "Spin"
                )
                spin_bowling_pct = (spin_count / max(len(merged), 1)) * 100
            else:
                spin_bowling_pct = 50.0  # Unknown = neutral

            records.append({
                "match_id": match_id,
                "team": team,
                "bowling_economy": round(bowling_economy, 2),
                "bowling_dot_pct": round(bowling_dot_pct, 2),
                "top_bowler_sr": round(top_bowler_sr, 2),
                "spin_bowling_pct": round(spin_bowling_pct, 1),
            })

        return pd.DataFrame(records)

    # =========================================================================
    # STEP 4: PIVOT TO MATCH LEVEL
    # =========================================================================

    def _pivot_to_match_level(self, team_feats: pd.DataFrame) -> pd.DataFrame:
        """Pivot team-level features into match-level columns (team1_ / team2_ prefix).

        Args:
            team_feats: DataFrame with (match_id, team, feature...) rows

        Returns:
            DataFrame with match_id + team1_<feature> + team2_<feature> columns
        """
        result = self.match_teams.copy()

        feature_cols = [c for c in team_feats.columns if c not in {"match_id", "team"}]

        # Join team1 features
        team1_feats = team_feats.rename(
            columns={c: f"team1_{c}" for c in feature_cols}
        )
        result = result.merge(
            team1_feats.rename(columns={"team": "team1"}),
            on=["match_id", "team1"],
            how="left",
        )

        # Join team2 features
        team2_feats = team_feats.rename(
            columns={c: f"team2_{c}" for c in feature_cols}
        )
        result = result.merge(
            team2_feats.rename(columns={"team": "team2"}),
            on=["match_id", "team2"],
            how="left",
        )

        # Drop team name columns (they're already in context features)
        result = result.drop(columns=["team1", "team2"])

        # Fill any remaining NaNs with neutral values
        result = result.fillna(
            {
                "team1_batting_power": 130.0,
                "team2_batting_power": 130.0,
                "team1_bowling_economy": 8.5,
                "team2_bowling_economy": 8.5,
                "team1_spin_bowling_pct": 50.0,
                "team2_spin_bowling_pct": 50.0,
                "team1_batting_depth": 3,
                "team2_batting_depth": 3,
            }
        )

        return result

    # =========================================================================
    # STEP 5: DIFFERENTIAL FEATURES
    # =========================================================================

    def _add_differential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team1 - team2 differential features.

        These capture relative advantage directly and often rank highest
        in feature importance for the win predictor.
        """
        if "team1_batting_power" in df.columns and "team2_batting_power" in df.columns:
            df["batting_power_diff"] = (
                df["team1_batting_power"] - df["team2_batting_power"]
            ).round(2)

        if "team1_bowling_economy" in df.columns and "team2_bowling_economy" in df.columns:
            # Negative = team1 bowls cheaper = team1 advantage
            df["bowling_economy_diff"] = (
                df["team1_bowling_economy"] - df["team2_bowling_economy"]
            ).round(2)

        if "team1_top3_sr_L10" in df.columns and "team2_top3_sr_L10" in df.columns:
            df["top_order_form_diff"] = (
                df["team1_top3_sr_L10"] - df["team2_top3_sr_L10"]
            ).round(2)

        if "team1_bowling_dot_pct" in df.columns and "team2_bowling_dot_pct" in df.columns:
            df["dot_ball_pressure_diff"] = (
                df["team1_bowling_dot_pct"] - df["team2_bowling_dot_pct"]
            ).round(2)

        return df
