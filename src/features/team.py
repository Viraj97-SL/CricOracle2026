"""Team-level feature aggregation.

Aggregates player-level features into team strength scores.
Handles: head-to-head records, team form, batting/bowling power ratings.

YOUR CSV has 'winner' column — we use it directly instead of
reconstructing winners from score comparison.

Usage:
    from src.features.team import TeamFeatureEngine

    engine = TeamFeatureEngine(ball_df)
    team_feats = engine.build_all_team_features()
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.config import CricketConstants
from src.utils.logger import logger


class TeamFeatureEngine:
    """Generates team-level features from ball-by-ball data."""

    def __init__(self, df: pd.DataFrame, player_profiles: Optional[pd.DataFrame] = None):
        self.df = df
        self.player_profiles = player_profiles
        self.cricket = CricketConstants()

    def build_all_team_features(self) -> pd.DataFrame:
        """Build team-level features for every match.

        Returns:
            DataFrame with match_id + team feature columns.
        """
        logger.info("Building team features...")

        # Get match list with teams and winner
        matches = self._get_match_list()

        # Feature 1: Head-to-head win rates
        h2h = self._calculate_head_to_head(matches)

        # Feature 2: Recent team form (win rate in last N matches)
        form = self._calculate_team_form(matches)

        # Feature 3: Team experience (matches played)
        experience = self._calculate_team_experience(matches)

        # Merge all
        features = (
            h2h
            .merge(form, on="match_id", how="left")
            .merge(experience, on="match_id", how="left")
        )

        logger.info(f"Built team features for {len(features)} matches")
        return features

    def _get_match_list(self) -> pd.DataFrame:
        """Get match list with team1, team2, winner, date."""
        agg = {"date": ("date", "first")}

        # Use original team1/team2 columns if available
        if "team1" in self.df.columns:
            agg["team1"] = ("team1", "first")
            agg["team2"] = ("team2", "first")
        else:
            agg["team1"] = ("batting_team", "first")
            agg["team2"] = ("bowling_team", "first")

        if "winner" in self.df.columns:
            agg["winner"] = ("winner", "first")

        matches = self.df.groupby("match_id").agg(**agg).reset_index()

        # Convert categoricals to string for comparison
        for col in ["team1", "team2", "winner"]:
            if col in matches.columns:
                matches[col] = matches[col].astype(str)

        matches = matches.sort_values("date").reset_index(drop=True)
        return matches

    def _calculate_head_to_head(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Head-to-head win rate for team1 vs team2 BEFORE each match."""
        logger.info("Calculating head-to-head records...")

        records = []
        for idx, match in matches.iterrows():
            t1, t2, date = match["team1"], match["team2"], match["date"]

            # All prior matches between these two teams
            prior = matches[
                (matches["date"] < date)
                & (
                    ((matches["team1"] == t1) & (matches["team2"] == t2))
                    | ((matches["team1"] == t2) & (matches["team2"] == t1))
                )
            ]

            total = len(prior)
            if total > 0 and "winner" in prior.columns:
                t1_wins = (prior["winner"] == t1).sum()
                h2h_rate = t1_wins / total
            else:
                h2h_rate = 0.5  # No history = even odds

            records.append({
                "match_id": match["match_id"],
                "h2h_matches_played": total,
                "h2h_team1_win_rate": round(h2h_rate, 3),
            })

        return pd.DataFrame(records)

    def _calculate_team_form(self, matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Rolling win rate for each team in their last N matches."""
        logger.info(f"Calculating team form (last {window} matches)...")

        records = []
        for _, match in matches.iterrows():
            t1, t2, date = match["team1"], match["team2"], match["date"]

            # Team 1 recent form
            t1_recent = matches[
                (matches["date"] < date)
                & ((matches["team1"] == t1) | (matches["team2"] == t1))
            ].tail(window)

            if len(t1_recent) > 0 and "winner" in t1_recent.columns:
                t1_form = (t1_recent["winner"] == t1).mean()
            else:
                t1_form = 0.5

            # Team 2 recent form
            t2_recent = matches[
                (matches["date"] < date)
                & ((matches["team1"] == t2) | (matches["team2"] == t2))
            ].tail(window)

            if len(t2_recent) > 0 and "winner" in t2_recent.columns:
                t2_form = (t2_recent["winner"] == t2).mean()
            else:
                t2_form = 0.5

            records.append({
                "match_id": match["match_id"],
                "team1_form_L10": round(t1_form, 3),
                "team2_form_L10": round(t2_form, 3),
                "form_diff": round(t1_form - t2_form, 3),
            })

        return pd.DataFrame(records)

    def _calculate_team_experience(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Cumulative matches played by each team before this match."""
        records = []
        for _, match in matches.iterrows():
            t1, t2, date = match["team1"], match["team2"], match["date"]

            t1_exp = len(matches[
                (matches["date"] < date)
                & ((matches["team1"] == t1) | (matches["team2"] == t1))
            ])
            t2_exp = len(matches[
                (matches["date"] < date)
                & ((matches["team1"] == t2) | (matches["team2"] == t2))
            ])

            records.append({
                "match_id": match["match_id"],
                "team1_matches_played": t1_exp,
                "team2_matches_played": t2_exp,
                "experience_diff": t1_exp - t2_exp,
            })

        return pd.DataFrame(records)

    def calculate_team_strength(
        self,
        team_name: str,
        playing_xi: Optional[list[str]] = None,
        venue: Optional[str] = None,
    ) -> dict:
        """Calculate composite team strength score (Phase 2)."""
        logger.warning("calculate_team_strength — implement in Phase 2")
        return {
            "team": team_name,
            "batting_power": 0.0,
            "bowling_power": 0.0,
            "composite_strength": 0.0,
        }
