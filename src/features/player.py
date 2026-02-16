"""Player feature engineering — form curves, role classification, and bowling style detection.

This module replaces the hardcoded bowler dictionaries from the original notebooks
with data-driven classification using K-Means clustering.

Key features generated per player:
- Rolling batting average, strike rate, boundary %
- Rolling bowling economy, strike rate, dot ball %
- Form momentum (exponential decay weighting)
- Bowling style classification (Spin/Pace via clustering)
- Role classification (Opener/Middle/Finisher via entry point analysis)

Usage:
    from src.features.player import PlayerFeatureEngine

    engine = PlayerFeatureEngine(ball_df)
    batter_profiles = engine.build_batting_profiles()
    bowler_profiles = engine.build_bowling_profiles()
    bowler_styles = engine.classify_bowling_styles()
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Optional

from src.config import settings, CricketConstants
from src.utils.logger import logger


class PlayerFeatureEngine:
    """Generates player-level features from ball-by-ball data."""

    def __init__(self, df: pd.DataFrame):
        """Initialise with ball-by-ball DataFrame.

        Args:
            df: Cleaned ball-by-ball DataFrame from CricketDataLoader.
        """
        self.df = df.copy()
        self.cricket = CricketConstants()
        self._ensure_required_columns()

    def _ensure_required_columns(self) -> None:
        """Verify required columns exist."""
        required = {"match_id", "batter", "bowler", "runs_batter", "runs_total",
                     "is_wicket", "over", "date", "venue", "batting_team"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================================
    # BATTING PROFILES
    # =========================================================================

    def build_batting_profiles(
        self,
        min_balls: Optional[int] = None,
    ) -> pd.DataFrame:
        """Build comprehensive batting profiles for all batters.

        Returns:
            DataFrame with one row per batter: career stats + rolling form.
        """
        min_balls = min_balls or self.cricket.MIN_BALLS_BATTER
        logger.info("Building batting profiles...")

        # Career aggregates
        career = self._batting_career_stats()

        # Phase-wise stats (Powerplay, Middle, Death)
        phase_stats = self._batting_phase_stats()

        # Rolling form (last N innings)
        form = self._batting_rolling_form()

        # Merge all
        profiles = (
            career
            .merge(phase_stats, on="batter", how="left")
            .merge(form, on="batter", how="left")
        )

        # Filter minimum sample size
        profiles = profiles[profiles["balls_faced"] >= min_balls]

        logger.info(f"Built profiles for {len(profiles)} batters (min {min_balls} balls)")
        return profiles

    def _batting_career_stats(self) -> pd.DataFrame:
        """Aggregate career batting statistics."""
        stats = self.df.groupby("batter").agg(
            matches=("match_id", "nunique"),
            innings=("match_id", "nunique"),  # Approximate
            balls_faced=("over", "count"),
            runs_scored=("runs_batter", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            dots_faced=("is_dot", "sum"),
            dismissals=("is_wicket", "sum"),
        ).reset_index()

        # Derived metrics
        stats["batting_avg"] = np.where(
            stats["dismissals"] > 0,
            stats["runs_scored"] / stats["dismissals"],
            stats["runs_scored"]  # Not out = runs as average
        )
        stats["strike_rate"] = (stats["runs_scored"] / stats["balls_faced"]) * 100
        stats["boundary_pct"] = (
            (stats["fours"] + stats["sixes"]) / stats["balls_faced"]
        ) * 100
        stats["dot_ball_pct"] = (stats["dots_faced"] / stats["balls_faced"]) * 100

        return stats

    def _batting_phase_stats(self) -> pd.DataFrame:
        """Batting stats broken down by phase (Powerplay/Middle/Death)."""
        if "phase" not in self.df.columns:
            logger.warning("No 'phase' column — skipping phase stats")
            return pd.DataFrame(columns=["batter"])

        phase_agg = self.df.groupby(["batter", "phase"]).agg(
            phase_runs=("runs_batter", "sum"),
            phase_balls=("over", "count"),
            phase_dots=("is_dot", "sum"),
            phase_boundaries=("is_four", "sum"),
            phase_sixes=("is_six", "sum"),
        ).reset_index()

        phase_agg["phase_sr"] = np.where(
            phase_agg["phase_balls"] > 0,
            (phase_agg["phase_runs"] / phase_agg["phase_balls"]) * 100,
            0,
        )

        # Pivot: one column per phase
        pivoted = phase_agg.pivot_table(
            index="batter",
            columns="phase",
            values=["phase_sr", "phase_runs", "phase_balls"],
            fill_value=0,
        )
        pivoted.columns = [f"{col[0]}_{col[1].lower()}" for col in pivoted.columns]
        pivoted = pivoted.reset_index()

        return pivoted

    def _batting_rolling_form(self) -> pd.DataFrame:
        """Calculate rolling batting form using last N innings.

        Uses exponential decay weighting: recent innings matter more.
        """
        logger.info("Calculating rolling batting form...")

        # Get per-innings stats
        innings = self.df.groupby(["batter", "match_id", "date"]).agg(
            runs=("runs_batter", "sum"),
            balls=("over", "count"),
            boundaries=("is_four", "sum"),
            sixes=("is_six", "sum"),
        ).reset_index().sort_values(["batter", "date"])

        innings["sr"] = np.where(
            innings["balls"] > 0, (innings["runs"] / innings["balls"]) * 100, 0
        )

        form_records = []

        for batter, group in innings.groupby("batter"):
            if len(group) < 3:  # Need at least 3 innings for rolling stats
                continue

            for window in [
                self.cricket.FORM_WINDOW_SHORT,
                self.cricket.FORM_WINDOW_MEDIUM,
            ]:
                recent = group.tail(window)

                # Exponential decay weights (most recent = highest weight)
                n = len(recent)
                weights = np.exp(np.linspace(-1, 0, n))
                weights /= weights.sum()

                weighted_sr = np.average(recent["sr"].values, weights=weights)
                weighted_runs = np.average(recent["runs"].values, weights=weights)

                form_records.append({
                    "batter": batter,
                    f"batting_sr_L{window}": round(weighted_sr, 2),
                    f"batting_avg_L{window}": round(weighted_runs, 2),
                    f"boundary_pct_L{window}": round(
                        (recent["boundaries"].sum() + recent["sixes"].sum())
                        / max(recent["balls"].sum(), 1) * 100, 2
                    ),
                })

        if not form_records:
            return pd.DataFrame(columns=["batter"])

        form_df = pd.DataFrame(form_records)
        # Merge rows for same batter (different window sizes)
        form_df = form_df.groupby("batter").first().reset_index()

        return form_df

    # =========================================================================
    # BOWLING PROFILES
    # =========================================================================

    def build_bowling_profiles(
        self,
        min_balls: Optional[int] = None,
    ) -> pd.DataFrame:
        """Build comprehensive bowling profiles for all bowlers."""
        min_balls = min_balls or self.cricket.MIN_BALLS_BOWLER
        logger.info("Building bowling profiles...")

        stats = self.df.groupby("bowler").agg(
            matches=("match_id", "nunique"),
            balls_bowled=("over", "count"),
            runs_conceded=("runs_total", "sum"),
            wickets=("is_wicket", "sum"),
            dots_bowled=("is_dot", "sum"),
            fours_conceded=("is_four", "sum"),
            sixes_conceded=("is_six", "sum"),
        ).reset_index()

        # Derived metrics
        stats["overs_bowled"] = stats["balls_bowled"] / 6
        stats["economy"] = np.where(
            stats["overs_bowled"] > 0,
            stats["runs_conceded"] / stats["overs_bowled"],
            0,
        )
        stats["bowling_sr"] = np.where(
            stats["wickets"] > 0,
            stats["balls_bowled"] / stats["wickets"],
            999,  # No wickets = very high strike rate
        )
        stats["bowling_avg"] = np.where(
            stats["wickets"] > 0,
            stats["runs_conceded"] / stats["wickets"],
            stats["runs_conceded"],
        )
        stats["dot_ball_pct"] = (stats["dots_bowled"] / stats["balls_bowled"]) * 100

        # Filter minimum sample
        stats = stats[stats["balls_bowled"] >= min_balls]

        logger.info(f"Built profiles for {len(stats)} bowlers (min {min_balls} balls)")
        return stats

    # =========================================================================
    # BOWLING STYLE CLASSIFICATION (Replaces hardcoded dictionary!)
    # =========================================================================

    def classify_bowling_styles(self, n_clusters: int = 2) -> pd.DataFrame:
        """Classify bowlers as Spin or Pace using K-Means clustering.

        Instead of maintaining a manual dictionary of 150+ bowlers,
        we use the data itself. Key insight:
        - Spinners bowl more in overs 7-15 (middle overs)
        - Pacers bowl more in overs 1-6 and 16-20 (powerplay + death)
        - Spinners typically have lower economy in middle overs
        - Pacers have more dot balls in death overs

        Returns:
            DataFrame with columns: [bowler, bowling_style, cluster_confidence]
        """
        logger.info("Classifying bowling styles using K-Means...")

        # Calculate clustering features per bowler
        bowler_features = self.df.groupby("bowler").agg(
            avg_over_bowled=("over", "mean"),
            median_over_bowled=("over", "median"),
            balls_bowled=("over", "count"),
            economy=("runs_total", lambda x: x.sum() / max(len(x) / 6, 1)),
            pct_middle_overs=("over", lambda x: ((x >= 7) & (x <= 14)).mean()),
            pct_death_overs=("over", lambda x: (x >= 16).mean()),
        ).reset_index()

        # Filter: need enough data for reliable classification
        min_balls = self.cricket.MIN_BALLS_BOWLER // 2  # Slightly relaxed for classification
        bowler_features = bowler_features[bowler_features["balls_bowled"] >= min_balls]

        if len(bowler_features) < 10:
            logger.warning(f"Only {len(bowler_features)} bowlers with enough data — "
                           "using fallback classification")
            return self._fallback_bowling_classification()

        # Features for clustering
        cluster_features = bowler_features[
            ["avg_over_bowled", "pct_middle_overs", "pct_death_overs"]
        ].values

        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_features)

        # K-Means with 2 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        bowler_features["cluster"] = kmeans.fit_predict(scaled)

        # Identify which cluster is Spin:
        # Spinners have HIGHER pct_middle_overs
        cluster_profiles = bowler_features.groupby("cluster")["pct_middle_overs"].mean()
        spin_cluster = cluster_profiles.idxmax()

        bowler_features["bowling_style"] = bowler_features["cluster"].map({
            spin_cluster: "Spin",
            1 - spin_cluster: "Pace",
        })

        # Confidence: distance from cluster center (normalised)
        distances = kmeans.transform(scaled)
        bowler_features["cluster_confidence"] = 1 - (
            distances.min(axis=1) / distances.max(axis=1)
        )

        result = bowler_features[["bowler", "bowling_style", "cluster_confidence"]]

        spin_count = (result["bowling_style"] == "Spin").sum()
        pace_count = (result["bowling_style"] == "Pace").sum()
        logger.info(
            f"Classified {len(result)} bowlers: {spin_count} Spin, {pace_count} Pace"
        )

        return result

    def _fallback_bowling_classification(self) -> pd.DataFrame:
        """Fallback: classify by average over bowled (simple threshold)."""
        stats = self.df.groupby("bowler").agg(
            avg_over=("over", "mean"),
            balls=("over", "count"),
        ).reset_index()

        # Simple heuristic: if average over > 10, more likely spin
        stats["bowling_style"] = np.where(stats["avg_over"] > 10, "Spin", "Pace")
        stats["cluster_confidence"] = 0.5  # Low confidence for fallback
        return stats[["bowler", "bowling_style", "cluster_confidence"]]

    # =========================================================================
    # ROLE CLASSIFICATION
    # =========================================================================

    def classify_batting_roles(self) -> pd.DataFrame:
        """Classify batters into roles based on entry point analysis.

        Roles:
        - Opener: Median entry over 0-1
        - Top Order: Median entry over 2-5
        - Middle Order: Median entry over 6-12
        - Finisher: Median entry over 13+

        This replaces the manual role assignment from Notebook 49.
        """
        logger.info("Classifying batting roles...")

        # Find the first ball each batter faced in each match
        entry_points = (
            self.df.groupby(["match_id", "batter"])["over"]
            .min()
            .reset_index()
            .rename(columns={"over": "entry_over"})
        )

        # Median entry point per batter
        median_entry = (
            entry_points.groupby("batter")["entry_over"]
            .agg(["median", "mean", "std", "count"])
            .reset_index()
            .rename(columns={
                "median": "median_entry_over",
                "mean": "mean_entry_over",
                "std": "entry_over_std",
                "count": "innings_count",
            })
        )

        # Classify role based on median entry over
        conditions = [
            median_entry["median_entry_over"] <= 1,
            median_entry["median_entry_over"] <= 5,
            median_entry["median_entry_over"] <= 12,
        ]
        choices = ["Opener", "Top Order", "Middle Order"]
        median_entry["batting_role"] = np.select(conditions, choices, default="Finisher")

        # Add consistency metric (low std = consistent role)
        median_entry["role_consistency"] = 1 / (1 + median_entry["entry_over_std"].fillna(5))

        logger.info(
            f"Roles: {(median_entry['batting_role'] == 'Opener').sum()} Openers, "
            f"{(median_entry['batting_role'] == 'Top Order').sum()} Top Order, "
            f"{(median_entry['batting_role'] == 'Middle Order').sum()} Middle Order, "
            f"{(median_entry['batting_role'] == 'Finisher').sum()} Finishers"
        )

        return median_entry
