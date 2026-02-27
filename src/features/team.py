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

        matches = self._get_match_list()

        h2h = self._calculate_head_to_head(matches)
        form = self._calculate_team_form(matches)
        experience = self._calculate_team_experience(matches)

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

        if "team1" in self.df.columns:
            agg["team1"] = ("team1", "first")
            agg["team2"] = ("team2", "first")
        else:
            agg["team1"] = ("batting_team", "first")
            agg["team2"] = ("bowling_team", "first")

        if "winner" in self.df.columns:
            agg["winner"] = ("winner", "first")

        matches = self.df.groupby("match_id").agg(**agg).reset_index()

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
                h2h_rate = 0.5

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

            t1_recent = matches[
                (matches["date"] < date)
                & ((matches["team1"] == t1) | (matches["team2"] == t1))
            ].tail(window)

            t1_form = (
                (t1_recent["winner"] == t1).mean()
                if len(t1_recent) > 0 and "winner" in t1_recent.columns
                else 0.5
            )

            t2_recent = matches[
                (matches["date"] < date)
                & ((matches["team1"] == t2) | (matches["team2"] == t2))
            ].tail(window)

            t2_form = (
                (t2_recent["winner"] == t2).mean()
                if len(t2_recent) > 0 and "winner" in t2_recent.columns
                else 0.5
            )

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
        """Calculate composite team strength score from player profiles.

        Used for inference time (e.g. API calls) where we have a known squad.
        For training, MatchPlayerFeatureEngine handles this via ball-by-ball.

        Args:
            team_name: Name of the team.
            playing_xi: List of player names in the playing XI.
            venue: Venue name (used to adjust spin weighting).

        Returns:
            Dict with batting_power, bowling_power, composite_strength.
        """
        if self.player_profiles is None:
            logger.warning(
                "calculate_team_strength called without player_profiles — "
                "returning defaults. Pass player_profiles to TeamFeatureEngine."
            )
            return {
                "team": team_name,
                "batting_power": 130.0,
                "bowling_power": 8.5,
                "composite_strength": 0.5,
            }

        if playing_xi is None:
            # Infer from historical data: most common players for this team
            playing_xi = self._infer_typical_xi(team_name)

        profiles = self.player_profiles.copy()
        team_players = profiles[profiles["batter"].isin(playing_xi)]

        if len(team_players) == 0:
            logger.warning(
                f"No profiles found for {team_name} players — using global median"
            )
            batting_power = float(profiles["strike_rate"].median())
        else:
            # Weight by career balls faced (more established players weighted higher)
            weights = team_players["balls_faced"].values.astype(float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(team_players)) / len(team_players)

            batting_power = float(
                np.average(team_players["strike_rate"].values, weights=weights)
            )

        # Bowling power from team form (proxy if no bowling profiles available)
        # A composite of recent form and batting power gives a 0-1 strength score
        team_recent_form = self._get_team_recent_form(team_name, n=10)

        # Composite: blend batting power (normalised) and recent win rate
        # normalise batting power: T20 SR range ~80-160, mid = 120
        batting_power_norm = np.clip((batting_power - 80) / 80, 0, 1)
        composite = 0.6 * team_recent_form + 0.4 * batting_power_norm

        return {
            "team": team_name,
            "batting_power": round(batting_power, 2),
            "bowling_power": 8.5,   # median economy proxy — use XIFeatureEngine for exact XI
            "recent_win_rate": round(team_recent_form, 3),
            "composite_strength": round(float(composite), 3),
        }

    def _infer_typical_xi(self, team_name: str, n: int = 11) -> list[str]:
        """Infer the most common playing XI for a team from historical data."""
        team_str = str(team_name)
        team_df = self.df[
            (self.df["batting_team"].astype(str) == team_str)
            | (self.df["bowling_team"].astype(str) == team_str)
        ]

        if len(team_df) == 0:
            return []

        # Most frequent batters for this team
        top_batters = (
            team_df[team_df["batting_team"].astype(str) == team_str]
            .groupby("batter")["match_id"]
            .nunique()
            .nlargest(n)
            .index.tolist()
        )
        return top_batters

    def _get_team_recent_form(self, team_name: str, n: int = 10) -> float:
        """Get win rate for a team in their last N matches from historical data."""
        team_str = str(team_name)
        matches = self.df.groupby("match_id").agg(
            team1=("team1", "first"),
            team2=("team2", "first"),
            winner=("winner", "first"),
            date=("date", "first"),
        ).reset_index()

        for col in ["team1", "team2", "winner"]:
            matches[col] = matches[col].astype(str)

        team_matches = matches[
            (matches["team1"] == team_str) | (matches["team2"] == team_str)
        ].sort_values("date").tail(n)

        if len(team_matches) == 0:
            return 0.5

        wins = (team_matches["winner"] == team_str).sum()
        return wins / len(team_matches)
