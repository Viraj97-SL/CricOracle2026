"""Playing XI Feature Engine — compute team strength from a known playing XI.

When the exact playing XI is known (pre-match or from squad selection), this
module computes role-bucket team strength features that feed directly into
the win predictor and score predictor.

Role-bucket approach (Narayanan et al. 2024, Sankaranarayanan et al. 2023):
    Top-order (pos 1-3)     → top3_batting_rating
    Middle-order (pos 4-6)  → mid_batting_rating
    Finisher (pos 7-8)      → finisher_batting_rating
    Pace bowlers            → pace_bowling_rating, pace_economy
    Spin bowlers            → spin_bowling_rating, spin_economy
    All-rounders bonus      → all_rounder_count (≥2 = bonus)

These replace the historical-average team features with exact-XI signal.

Usage:
    from src.features.xi_features import XIFeatureEngine

    engine = XIFeatureEngine()
    feats = engine.compute_xi_features(
        players=["Rohit Sharma", "Virat Kohli", ...],
        batting_profiles=batting_df,
        bowling_profiles=bowling_df,
        bowling_styles=styles_df,
    )
    # Returns dict of team1_ or team2_ prefixed features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.logger import logger


# Batting order role → bucket mapping
_ROLE_TO_BUCKET = {
    "Opener":       "top_order",
    "Top Order":    "top_order",
    "Middle Order": "middle_order",
    "Lower Order":  "lower_order",
    "Finisher":     "lower_order",
    "Tail":         "tail",
}

_BATTING_ROLE_TO_BUCKET = {
    "Opener":             "top_order",
    "Top order bat":      "top_order",
    "Top Order":          "top_order",
    "Middle order bat":   "middle_order",
    "Middle Order":       "middle_order",
    "Hard-hitting lower": "lower_order",
    "Lower Order":        "lower_order",
    "Tail":               "tail",
}


class XIFeatureEngine:
    """Compute match-level features from a specific playing XI."""

    # Global fallbacks (population medians from training data)
    _FALLBACK = {
        "batting_sr":    130.0,
        "batting_sr_L10": 125.0,
        "batting_avg_L10": 22.0,
        "boundary_pct":   12.0,
        "economy":        8.5,
        "dot_ball_pct":   35.0,
        "bowling_sr":     20.0,
    }

    def __init__(
        self,
        batting_profiles: Optional[pd.DataFrame] = None,
        bowling_profiles: Optional[pd.DataFrame] = None,
        bowling_styles: Optional[pd.DataFrame] = None,
        batting_roles: Optional[pd.DataFrame] = None,
    ):
        self.batting_profiles = batting_profiles
        self.bowling_profiles = bowling_profiles
        self.bowling_styles = bowling_styles
        self.batting_roles = batting_roles

        # Pre-build lookup maps for speed
        self._bat_map: dict = {}
        self._bowl_map: dict = {}
        self._style_map: dict = {}
        self._role_map: dict = {}

        if batting_profiles is not None:
            bname = "batter" if "batter" in batting_profiles.columns else "player_name"
            for _, row in batting_profiles.iterrows():
                self._bat_map[str(row[bname])] = row.to_dict()

        if bowling_profiles is not None:
            bname = "bowler" if "bowler" in bowling_profiles.columns else "player_name"
            for _, row in bowling_profiles.iterrows():
                self._bowl_map[str(row[bname])] = row.to_dict()

        if bowling_styles is not None:
            bname = "bowler" if "bowler" in bowling_styles.columns else "player_name"
            for _, row in bowling_styles.iterrows():
                self._style_map[str(row[bname])] = str(row.get("bowling_style", "Pace"))

        if batting_roles is not None:
            bname = "batter" if "batter" in batting_roles.columns else "player_name"
            for _, row in batting_roles.iterrows():
                self._role_map[str(row[bname])] = str(row.get("batting_role", "Middle Order"))

        # Update fallbacks from actual data
        if batting_profiles is not None and len(batting_profiles) > 0:
            bp = batting_profiles
            for col, fallback_key in [
                ("strike_rate", "batting_sr"),
                ("batting_sr_L10", "batting_sr_L10"),
                ("batting_avg_L10", "batting_avg_L10"),
                ("boundary_pct", "boundary_pct"),
            ]:
                if col in bp.columns:
                    med = float(bp[col].median())
                    if not np.isnan(med):
                        self._FALLBACK[fallback_key] = med

        if bowling_profiles is not None and len(bowling_profiles) > 0:
            bp = bowling_profiles
            for col, fallback_key in [
                ("economy", "economy"),
                ("dot_ball_pct", "dot_ball_pct"),
                ("bowling_sr", "bowling_sr"),
            ]:
                if col in bp.columns:
                    vals = bp[col].replace(999, np.nan).dropna()
                    if len(vals) > 0:
                        self._FALLBACK[fallback_key] = float(vals.median())

    # =========================================================================
    # Public API
    # =========================================================================

    def compute_xi_features(
        self,
        players: list[str],
        squad_meta: Optional[list[dict]] = None,
        prefix: str = "team1",
    ) -> dict:
        """Compute team-strength features for a given playing XI.

        Args:
            players: List of player names (the playing XI).
            squad_meta: Optional list of squad dicts from wc2026_squads.json with
                        role, batting_order, bowling_style per player.
            prefix: Feature name prefix — 'team1' or 'team2'.

        Returns:
            Dictionary of features matching the win/score predictor feature sets.
        """
        if not players:
            return self._fallback_features(prefix)

        player_data = [self._resolve_player(p, squad_meta) for p in players]

        # ── Batting features ─────────────────────────────────────────────────
        batters = sorted(player_data, key=lambda x: x["bat_order_rank"])

        # Top-3 batting rating (exponentially weighted SR of top order)
        top3 = batters[:3]
        top3_sr = float(np.mean([p["batting_sr_L10"] for p in top3]))
        top3_runs = float(np.mean([p["batting_avg_L10"] for p in top3]))

        # Batting power = weighted avg SR of top 6
        top6 = batters[:6]
        weights = np.array([max(p["balls_faced"], 1) for p in top6], dtype=float)
        weights /= weights.sum()
        batting_power = float(np.average([p["batting_sr"] for p in top6], weights=weights))

        # Boundary %
        avg_boundary_pct = float(np.mean([p["boundary_pct"] for p in top6]))

        # ── Bowling features ──────────────────────────────────────────────────
        bowlers = [p for p in player_data if p.get("can_bowl", False)]
        if len(bowlers) < 4:
            # Everyone might bowl in emergencies
            bowlers = player_data

        # Sort by bowling rating (lower economy = better)
        bowlers = sorted(bowlers, key=lambda x: x["economy"])[:5]

        avg_economy = float(np.mean([p["economy"] for p in bowlers]))
        avg_dot_pct = float(np.mean([p["dot_ball_pct"] for p in bowlers]))
        top_bowler_sr = float(np.mean([p["bowling_sr"] for p in bowlers[:3]]))

        spinners = [p for p in bowlers if p.get("is_spin", False)]
        spin_pct = (len(spinners) / max(len(bowlers), 1)) * 100.0

        # ── All-rounder bonus ─────────────────────────────────────────────────
        all_rounders = [p for p in player_data if p.get("is_allrounder", False)]

        # ── Assemble result ───────────────────────────────────────────────────
        return {
            f"{prefix}_batting_power":     round(batting_power, 2),
            f"{prefix}_top3_sr_L10":       round(top3_sr, 2),
            f"{prefix}_top3_runs_L10":     round(top3_runs, 2),
            f"{prefix}_avg_boundary_pct":  round(avg_boundary_pct, 2),
            f"{prefix}_bowling_economy":   round(avg_economy, 2),
            f"{prefix}_bowling_dot_pct":   round(avg_dot_pct, 2),
            f"{prefix}_top_bowler_sr":     round(top_bowler_sr, 2),
            f"{prefix}_spin_bowling_pct":  round(spin_pct, 2),
            "_all_rounder_count":          len(all_rounders),
        }

    def compute_differential_features(self, t1_feats: dict, t2_feats: dict) -> dict:
        """Compute difference features from two teams' XI features.

        Sankaranarayanan et al. (2023): difference features significantly
        improve symmetry and reduce order-dependence in two-team sport models.
        """
        return {
            "batting_power_diff":       t1_feats.get("team1_batting_power", 130) - t2_feats.get("team2_batting_power", 130),
            "top_order_form_diff":      t1_feats.get("team1_top3_sr_L10", 125) - t2_feats.get("team2_top3_sr_L10", 125),
            "bowling_economy_diff":     t1_feats.get("team1_bowling_economy", 8.5) - t2_feats.get("team2_bowling_economy", 8.5),
            "dot_ball_pressure_diff":   t1_feats.get("team1_bowling_dot_pct", 35) - t2_feats.get("team2_bowling_dot_pct", 35),
        }

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _resolve_player(self, name: str, squad_meta: Optional[list[dict]]) -> dict:
        """Look up a player's stats from profiles, falling back to squad meta or defaults."""
        # Try batting profile
        bat = self._bat_map.get(name, {})
        bowl = self._bowl_map.get(name, {})

        # Squad meta (from JSON) provides role info
        meta = {}
        if squad_meta:
            for m in squad_meta:
                if m.get("name") == name:
                    meta = m
                    break

        # Batting order rank (lower = bats earlier)
        bat_order_str = meta.get("batting_order") or self._role_map.get(name, "Middle Order")
        bat_order_rank = self._batting_order_rank(bat_order_str)

        # Bowling style
        bowl_style = meta.get("bowling_style") or self._style_map.get(name)
        is_spin = bowl_style == "Spin" if bowl_style else False
        can_bowl = bowl_style is not None or bool(bowl)

        # Is all-rounder?
        role = meta.get("role", "Batsman")
        is_allrounder = "All-rounder" in role or "all-round" in role.lower()

        return {
            "name":           name,
            "batting_sr":     self._get(bat, "strike_rate", "batting_sr"),
            "batting_sr_L10": self._get(bat, "batting_sr_L10", "batting_sr_L10"),
            "batting_avg_L10": self._get(bat, "batting_avg_L10", "batting_avg_L10"),
            "boundary_pct":   self._get(bat, "boundary_pct", "boundary_pct"),
            "balls_faced":    float(bat.get("balls_faced", 100)),
            "economy":        self._get(bowl, "economy", "economy"),
            "dot_ball_pct":   self._get(bowl, "dot_ball_pct", "dot_ball_pct"),
            "bowling_sr":     self._get_bowling_sr(bowl),
            "bat_order_rank": bat_order_rank,
            "is_spin":        is_spin,
            "can_bowl":       can_bowl or is_allrounder,
            "is_allrounder":  is_allrounder,
        }

    def _get(self, profile: dict, col: str, fallback_key: str) -> float:
        v = profile.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return self._FALLBACK[fallback_key]
        return float(v)

    def _get_bowling_sr(self, bowl: dict) -> float:
        v = bowl.get("bowling_sr")
        if v is None or (isinstance(v, float) and np.isnan(v)) or v >= 999:
            return self._FALLBACK["bowling_sr"]
        return float(v)

    @staticmethod
    def _batting_order_rank(order_str: str) -> int:
        mapping = {
            "Opener": 1, "Top Order": 2, "Top order bat": 2,
            "Middle Order": 3, "Middle order bat": 3,
            "Lower Order": 4, "Hard-hitting lower": 4, "Finisher": 4,
            "Tail": 5,
        }
        return mapping.get(order_str, 3)

    def _fallback_features(self, prefix: str) -> dict:
        return {
            f"{prefix}_batting_power":    self._FALLBACK["batting_sr"],
            f"{prefix}_top3_sr_L10":      self._FALLBACK["batting_sr_L10"],
            f"{prefix}_top3_runs_L10":    self._FALLBACK["batting_avg_L10"],
            f"{prefix}_avg_boundary_pct": self._FALLBACK["boundary_pct"],
            f"{prefix}_bowling_economy":  self._FALLBACK["economy"],
            f"{prefix}_bowling_dot_pct":  self._FALLBACK["dot_ball_pct"],
            f"{prefix}_top_bowler_sr":    self._FALLBACK["bowling_sr"],
            f"{prefix}_spin_bowling_pct": 40.0,
            "_all_rounder_count":         2,
        }