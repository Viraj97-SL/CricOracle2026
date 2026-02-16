"""Venue-level feature engineering.

Calculates ground-specific stats: average scores, spin/pace wicket ratios,
chase success rates, phase-wise scoring patterns.

Usage:
    from src.features.venue import VenueFeatureEngine

    engine = VenueFeatureEngine(ball_df)
    venue_feats = engine.build_venue_features()
"""

import pandas as pd
import numpy as np
from src.config import CricketConstants
from src.utils.logger import logger


class VenueFeatureEngine:
    """Generates venue-level features from historical match data."""

    def __init__(self, df: pd.DataFrame, bowler_styles: pd.DataFrame = None):
        self.df = df
        self.bowler_styles = bowler_styles
        self.cricket = CricketConstants()

    def build_venue_features(self) -> pd.DataFrame:
        """Build comprehensive venue feature matrix.

        Returns:
            DataFrame with one row per venue containing all venue stats.
        """
        logger.info("Building venue features...")

        # 1. Average scores
        scoring = self._venue_scoring_stats()

        # 2. Spin vs Pace wicket ratios
        if self.bowler_styles is not None:
            bowling = self._venue_bowling_stats()
            scoring = scoring.merge(bowling, on="venue", how="left")

        # 3. Chase success rate
        chase = self._venue_chase_stats()
        scoring = scoring.merge(chase, on="venue", how="left")

        # 4. Subcontinent classification
        scoring["is_subcontinent"] = scoring["venue"].apply(self._is_subcontinent)

        logger.info(f"Built features for {len(scoring)} venues")
        return scoring

    def _venue_scoring_stats(self) -> pd.DataFrame:
        """Average scoring patterns per venue."""
        # First innings stats
        first_inn = self.df[self.df["inning_no"] == 1]
        first_scores = first_inn.groupby(["match_id", "venue"]).agg(
            first_inn_total=("runs_total", "sum"),
        ).reset_index()

        venue_scores = first_scores.groupby("venue").agg(
            venue_avg_1st_inn_score=("first_inn_total", "mean"),
            venue_median_1st_inn_score=("first_inn_total", "median"),
            venue_std_1st_inn_score=("first_inn_total", "std"),
            matches_at_venue=("match_id", "nunique"),
        ).reset_index()

        # Phase-wise scoring (powerplay, middle, death)
        for phase_name, over_start, over_end in [
            ("powerplay", 0, 5), ("middle", 6, 14), ("death", 15, 19)
        ]:
            phase_data = first_inn[
                (first_inn["over"] >= over_start) & (first_inn["over"] <= over_end)
            ]
            phase_scores = phase_data.groupby("venue")["runs_total"].mean().reset_index()
            phase_scores.columns = ["venue", f"venue_avg_{phase_name}_rpo"]
            venue_scores = venue_scores.merge(phase_scores, on="venue", how="left")

        return venue_scores

    def _venue_bowling_stats(self) -> pd.DataFrame:
        """Spin vs Pace wicket ratios per venue."""
        # Merge bowler styles
        df_with_style = self.df.merge(
            self.bowler_styles[["bowler", "bowling_style"]],
            on="bowler",
            how="left",
        )

        wickets = df_with_style[df_with_style["is_wicket"] == 1]

        venue_bowling = wickets.groupby(["venue", "bowling_style"]).agg(
            wickets=("is_wicket", "sum"),
        ).reset_index()

        # Pivot to get spin_wickets and pace_wickets columns
        pivoted = venue_bowling.pivot_table(
            index="venue", columns="bowling_style",
            values="wickets", fill_value=0,
        ).reset_index()

        pivoted.columns.name = None
        if "Spin" in pivoted.columns and "Pace" in pivoted.columns:
            total = pivoted["Spin"] + pivoted["Pace"]
            pivoted["venue_spin_wicket_pct"] = np.where(
                total > 0, (pivoted["Spin"] / total) * 100, 50
            )
            pivoted["venue_pace_wicket_pct"] = 100 - pivoted["venue_spin_wicket_pct"]
        else:
            pivoted["venue_spin_wicket_pct"] = 50.0
            pivoted["venue_pace_wicket_pct"] = 50.0

        return pivoted[["venue", "venue_spin_wicket_pct", "venue_pace_wicket_pct"]]

    def _venue_chase_stats(self) -> pd.DataFrame:
        """Chase success rate per venue."""
        # Determine match winners
        match_scores = self.df.groupby(["match_id", "venue", "inning_no", "batting_team"]).agg(
            total=("runs_total", "sum"),
        ).reset_index()

        chase_results = []
        for match_id, group in match_scores.groupby("match_id"):
            inn1 = group[group["inning_no"] == 1]
            inn2 = group[group["inning_no"] == 2]

            if len(inn1) > 0 and len(inn2) > 0:
                chasing_team_won = inn2.iloc[0]["total"] > inn1.iloc[0]["total"]
                chase_results.append({
                    "match_id": match_id,
                    "venue": group.iloc[0]["venue"],
                    "chase_won": int(chasing_team_won),
                })

        if not chase_results:
            return pd.DataFrame(columns=["venue", "venue_chase_win_pct"])

        chase_df = pd.DataFrame(chase_results)
        venue_chase = chase_df.groupby("venue").agg(
            venue_chase_win_pct=("chase_won", "mean"),
            venue_chase_matches=("chase_won", "count"),
        ).reset_index()

        venue_chase["venue_chase_win_pct"] = (venue_chase["venue_chase_win_pct"] * 100).round(1)

        return venue_chase

    def _is_subcontinent(self, venue: str) -> bool:
        """Check if a venue is in the subcontinent (spin-friendly conditions)."""
        venue_lower = venue.lower()
        return any(kw in venue_lower for kw in self.cricket.SUBCONTINENT_KEYWORDS)
