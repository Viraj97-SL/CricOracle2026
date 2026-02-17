"""Match context features — toss, date features, innings scores, winner.

Extracts match-level context from the ball-by-ball data.
Since YOUR CSV has winner, team1, team2 — we preserve them here
so they flow through the pipeline into the target variable.

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

        Returns one row per match with: teams, winner, date features,
        toss info, first innings score.
        """
        logger.info("Building match context features...")

        # ---- Core match info ----
        # Use team1/team2 from your CSV if available, else derive from batting/bowling
        agg_dict = {
            "date": ("date", "first"),
            "venue": ("venue", "first"),
        }

        # Preserve team1, team2, winner from your original CSV columns
        for col in ["team1", "team2", "winner"]:
            if col in self.df.columns:
                agg_dict[col] = (col, "first")

        # Always add batting_team/bowling_team as derived team identifiers
        agg_dict["team_batting_first"] = ("batting_team", "first")
        agg_dict["team_bowling_first"] = ("bowling_team", "first")

        match_context = self.df.groupby("match_id").agg(**agg_dict).reset_index()

        # If team1/team2 not in data, use batting/bowling team as fallback
        if "team1" not in match_context.columns:
            match_context["team1"] = match_context["team_batting_first"]
        if "team2" not in match_context.columns:
            match_context["team2"] = match_context["team_bowling_first"]

        # ---- Create target variable: team1_won ----
        if "winner" in match_context.columns:
            # Convert to string to handle categorical comparison
            t1 = match_context["team1"].astype(str)
            winner = match_context["winner"].astype(str)
            match_context["team1_won"] = (t1 == winner).astype(int)

            # Also flag batting_first_won
            tbf = match_context["team_batting_first"].astype(str)
            match_context["batting_first_won"] = (tbf == winner).astype(int)
            logger.info(
                f"Target variable: team1 won {match_context['team1_won'].sum()}/{len(match_context)} "
                f"({match_context['team1_won'].mean()*100:.1f}%)"
            )
        else:
            logger.warning("No 'winner' column — cannot create target variable")

        # ---- Toss features (if available) ----
        if "toss_winner" in self.df.columns:
            toss = self.df.groupby("match_id").agg(
                toss_winner=("toss_winner", "first"),
                toss_decision=("toss_decision", "first"),
            ).reset_index()
            match_context = match_context.merge(toss, on="match_id", how="left")
            match_context["toss_winner_is_team1"] = (
                match_context["toss_winner"].astype(str) == match_context["team1"].astype(str)
            ).astype(int)
            match_context["elected_to_bat"] = (
                match_context["toss_decision"].str.lower() == "bat"
            ).astype(int)
        else:
            match_context["toss_winner_is_team1"] = 0
            match_context["elected_to_bat"] = 0
            logger.info("No toss data — using defaults (0)")

        # ---- Date features ----
        date_col = pd.to_datetime(match_context["date"])
        match_context["month"] = date_col.dt.month
        match_context["day_of_week"] = date_col.dt.dayofweek
        match_context["year"] = date_col.dt.year

        # ---- First innings score (critical for score predictor target) ----
        scores = self.df.groupby(["match_id", "inning_no"]).agg(
            innings_total=("runs_total", "sum"),
            innings_wickets=("is_wicket", "sum"),
        ).reset_index()

        first_inn = scores[scores["inning_no"] == 1][
            ["match_id", "innings_total", "innings_wickets"]
        ].rename(columns={
            "innings_total": "first_innings_score",
            "innings_wickets": "first_innings_wickets",
        })
        match_context = match_context.merge(first_inn, on="match_id", how="left")

        second_inn = scores[scores["inning_no"] == 2][
            ["match_id", "innings_total", "innings_wickets"]
        ].rename(columns={
            "innings_total": "second_innings_score",
            "innings_wickets": "second_innings_wickets",
        })
        match_context = match_context.merge(second_inn, on="match_id", how="left")

        logger.info(f"Built context features for {len(match_context)} matches")
        return match_context
