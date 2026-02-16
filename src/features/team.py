"""Team-level feature aggregation.

Aggregates player-level features into team strength scores.
Handles: head-to-head records, team form indices, batting/bowling power ratings.

Usage:
    from src.features.team import TeamFeatureEngine

    engine = TeamFeatureEngine(ball_df, player_profiles)
    team_feats = engine.build_team_features("India", "Australia")
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.config import CricketConstants
from src.utils.logger import logger


class TeamFeatureEngine:
    """Generates team-level features from player profiles and match data."""

    def __init__(self, df: pd.DataFrame, player_profiles: Optional[pd.DataFrame] = None):
        self.df = df
        self.player_profiles = player_profiles
        self.cricket = CricketConstants()

    def build_all_team_features(self) -> pd.DataFrame:
        """Build team-level features for every match in the dataset.

        Returns:
            DataFrame with columns per match: team strengths, h2h, experience.
        """
        logger.info("Building team features...")

        matches = self.df.groupby("match_id").agg(
            team1=("batting_team", "first"),
            team2=("bowling_team", "first"),
            date=("date", "first"),
        ).reset_index()

        # TODO Phase 1 Step 2: Build these features
        # For each match, calculate:
        # 1. team_batting_power (mean SR × avg of top 7)
        # 2. team_bowling_power (mean eco × wicket rate)
        # 3. team_experience (total caps)
        # 4. team_form_index (mean player form_momentum)
        # 5. head_to_head_win_rate
        # 6. h2h_at_venue_win_rate

        features = self._calculate_head_to_head(matches)

        logger.info(f"Built team features for {len(features)} matches")
        return features

    def _calculate_head_to_head(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical head-to-head win rates."""
        # Get match results
        match_results = self._get_match_results()

        h2h_records = []
        for _, match in matches.iterrows():
            t1, t2, date = match["team1"], match["team2"], match["date"]

            # Historical matches between these teams before this date
            prior = match_results[
                (match_results["date"] < date)
                & (
                    ((match_results["team1"] == t1) & (match_results["team2"] == t2))
                    | ((match_results["team1"] == t2) & (match_results["team2"] == t1))
                )
            ]

            total_played = len(prior)
            if total_played > 0:
                t1_wins = len(prior[prior["winner"] == t1])
                h2h_win_rate = t1_wins / total_played
            else:
                h2h_win_rate = 0.5  # No history = even odds

            h2h_records.append({
                "match_id": match["match_id"],
                "h2h_matches_played": total_played,
                "h2h_team1_win_rate": round(h2h_win_rate, 3),
            })

        return pd.DataFrame(h2h_records)

    def _get_match_results(self) -> pd.DataFrame:
        """Reconstruct match results (winner) from ball-by-ball data."""
        # Calculate total scores per team per match
        scores = self.df.groupby(["match_id", "batting_team", "date"]).agg(
            total_runs=("runs_total", "sum"),
        ).reset_index()

        # For each match, determine winner (team with higher score)
        match_scores = scores.pivot_table(
            index=["match_id", "date"],
            columns="batting_team",
            values="total_runs",
            fill_value=0,
        ).reset_index()

        # Flatten columns
        teams = [c for c in match_scores.columns if c not in ("match_id", "date")]

        results = []
        for _, row in match_scores.iterrows():
            team_scores = {t: row.get(t, 0) for t in teams if pd.notna(row.get(t))}
            if len(team_scores) >= 2:
                sorted_teams = sorted(team_scores.items(), key=lambda x: x[1], reverse=True)
                results.append({
                    "match_id": row["match_id"],
                    "date": row["date"],
                    "team1": sorted_teams[0][0],
                    "team2": sorted_teams[1][0],
                    "winner": sorted_teams[0][0],
                    "team1_score": sorted_teams[0][1],
                    "team2_score": sorted_teams[1][1],
                })

        return pd.DataFrame(results)

    def calculate_team_strength(
        self,
        team_name: str,
        playing_xi: Optional[list[str]] = None,
        venue: Optional[str] = None,
    ) -> dict:
        """Calculate composite team strength score.

        This is the objective function used by the squad optimiser.

        Args:
            team_name: Name of the team.
            playing_xi: List of 11 player names (if None, uses latest available squad).
            venue: Venue name for conditions-based weighting.

        Returns:
            Dictionary with strength breakdown.
        """
        # TODO: Implement after player profiles are built
        # This will weight batting power, bowling power, and matchup advantages
        logger.warning("calculate_team_strength is a stub — implement in Phase 2")
        return {
            "team": team_name,
            "batting_power": 0.0,
            "bowling_power": 0.0,
            "composite_strength": 0.0,
        }
