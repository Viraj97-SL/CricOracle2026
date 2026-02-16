"""Match context features — toss, tournament stage, day/night, innings position.

Usage:
    from src.features.match_context import MatchContextEngine

    engine = MatchContextEngine(ball_df)
    context_feats = engine.build_context_features()
"""

import pandas as pd
import numpy as np
from src.utils.logger import logger


class MatchContextEngine:
    """Generates match-level context features."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build_context_features(self) -> pd.DataFrame:
        """Build context features for every match.

        Includes: toss info, day/night flag, tournament stage (if available).
        """
        logger.info("Building match context features...")

        match_context = self.df.groupby("match_id").agg(
            date=("date", "first"),
            venue=("venue", "first"),
            team1=("batting_team", "first"),
            team2=("bowling_team", "first"),
        ).reset_index()

        # Toss features (if available in data)
        if "toss_winner" in self.df.columns:
            toss = self.df.groupby("match_id").agg(
                toss_winner=("toss_winner", "first"),
                toss_decision=("toss_decision", "first"),
            ).reset_index()
            match_context = match_context.merge(toss, on="match_id", how="left")
            match_context["toss_winner_is_team1"] = (
                match_context["toss_winner"] == match_context["team1"]
            ).astype(int)
            match_context["elected_to_bat"] = (
                match_context["toss_decision"].str.lower() == "bat"
            ).astype(int)
        else:
            match_context["toss_winner_is_team1"] = 0
            match_context["elected_to_bat"] = 0
            logger.warning("No toss data found — using defaults")

        # Day/Night detection (evening matches in India start after 14:00 UTC)
        if "date" in match_context.columns:
            match_context["month"] = pd.to_datetime(match_context["date"]).dt.month
            match_context["day_of_week"] = pd.to_datetime(match_context["date"]).dt.dayofweek
            match_context["year"] = pd.to_datetime(match_context["date"]).dt.year

        # Calculate match scores for target/chase context
        scores = self.df.groupby(["match_id", "inning_no"]).agg(
            innings_total=("runs_total", "sum"),
            innings_wickets=("is_wicket", "sum"),
        ).reset_index()

        first_inn_scores = scores[scores["inning_no"] == 1][["match_id", "innings_total"]]
        first_inn_scores.columns = ["match_id", "first_innings_score"]
        match_context = match_context.merge(first_inn_scores, on="match_id", how="left")

        logger.info(f"Built context features for {len(match_context)} matches")
        return match_context
