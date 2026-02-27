"""ModelService — singleton that loads trained models and serves real predictions.

Wires together:
    - WinPredictor (XGBoost)  → P(team1 wins)
    - ScorePredictor (XGBoost) → first innings score
    - SHAP explainability      → key factors for each prediction
    - XIFeatureEngine          → exact player-level features when XI is known

Feature construction strategy:
    Given team1, team2, venue (all pre-match):
    1. If playing XI provided → use XIFeatureEngine for exact player-level features
    2. Otherwise → look up most recent match records for each team (rolling averages)
    3. Look up venue stats from venue_features.parquet
    4. Construct H2H features from historical matches
    5. Build feature vector matching the model's expected columns

Usage:
    from src.models.model_service import get_model_service

    svc = get_model_service()
    result = svc.predict_match("India", "Australia", "Wankhede Stadium, Mumbai")
    result_xi = svc.predict_match(
        "India", "Australia", "Wankhede Stadium, Mumbai",
        team1_xi=["Rohit Sharma", "Virat Kohli", ...],
        team2_xi=["Travis Head", "David Warner", ...],
    )
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import logger

# Number of recent matches per team to average for team-strength snapshot
_RECENT_MATCHES = 10


class ModelService:
    """Loads models once and serves predictions. Instantiate via get_model_service()."""

    def __init__(self):
        self._win_model = None
        self._score_model = None
        self._win_features: list[str] = []
        self._score_features: list[str] = []
        self._match_df: Optional[pd.DataFrame] = None
        self._venue_df: Optional[pd.DataFrame] = None
        self._xi_engine = None
        self._squads: dict = {}
        self._loaded = False

    # =========================================================================
    # Startup
    # =========================================================================

    def load(self) -> None:
        """Load all models and reference data. Called once at API startup."""
        if self._loaded:
            return

        logger.info("ModelService: loading models and reference data...")

        # Load trained models
        win_data = joblib.load(settings.paths.models / "win_predictor_xgb.pkl")
        self._win_model = win_data["model"]
        self._win_features = win_data["feature_names"]

        # Load Platt calibrator (optional — falls back to raw probabilities)
        self._platt = None
        cal_path = settings.paths.models / "win_predictor_calibrator.pkl"
        if cal_path.exists():
            cal_data = joblib.load(cal_path)
            self._platt = cal_data["platt"]
            logger.info("Platt calibrator loaded")

        score_data = joblib.load(settings.paths.models / "score_predictor_xgb.pkl")
        self._score_model = score_data["model"]
        self._score_features = score_data["feature_names"]

        # Load processed match features (reference data for team lookups)
        feat_path = settings.paths.processed_features
        if feat_path.exists():
            self._match_df = pd.read_parquet(feat_path)
            if "date" in self._match_df.columns:
                self._match_df = self._match_df.sort_values("date").reset_index(drop=True)
            # Normalise team names to str
            for col in ("team1", "team2"):
                if col in self._match_df.columns:
                    self._match_df[col] = self._match_df[col].astype(str)
            if "venue" in self._match_df.columns:
                self._match_df["venue"] = self._match_df["venue"].astype(str)
            logger.info(f"Loaded {len(self._match_df)} match records for feature lookup")
        else:
            logger.warning("match_features.parquet not found — using global fallbacks")

        # Load venue features for venue lookup
        venue_path = settings.paths.data_processed / "venue_features.parquet"
        if venue_path.exists():
            self._venue_df = pd.read_parquet(venue_path)
            if "venue" in self._venue_df.columns:
                self._venue_df["venue"] = self._venue_df["venue"].astype(str)
            logger.info(f"Loaded {len(self._venue_df)} venue records")

        # Load player profiles for XI-based feature engine
        self._xi_engine = self._build_xi_engine()

        # Load WC 2026 squad data
        squads_path = settings.paths.root / "data" / "squads" / "wc2026_squads.json"
        if squads_path.exists():
            with open(squads_path) as f:
                data = json.load(f)
            self._squads = data.get("teams", {})
            logger.info(f"Loaded squads for {len(self._squads)} teams")

        self._loaded = True
        logger.info("ModelService: ready")

    def _build_xi_engine(self):
        """Build XIFeatureEngine from available player profiles."""
        try:
            from src.features.xi_features import XIFeatureEngine
            bp_path = settings.paths.data_processed / "batting_profiles.parquet"
            bowl_path = settings.paths.data_processed / "bowling_profiles.parquet"
            roles_path = settings.paths.data_processed / "batting_roles.csv"
            styles_path = settings.paths.data_processed / "bowling_styles.csv"
            bp = pd.read_parquet(bp_path) if bp_path.exists() else None
            bowl = pd.read_parquet(bowl_path) if bowl_path.exists() else None
            roles = pd.read_csv(roles_path) if roles_path.exists() else None
            styles = pd.read_csv(styles_path) if styles_path.exists() else None
            engine = XIFeatureEngine(
                batting_profiles=bp,
                bowling_profiles=bowl,
                bowling_styles=styles,
                batting_roles=roles,
            )
            logger.info("XIFeatureEngine loaded")
            return engine
        except Exception as e:
            logger.warning(f"Could not build XIFeatureEngine: {e}")
            return None

    # =========================================================================
    # Public API
    # =========================================================================

    def predict_match(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: Optional[str] = None,
        toss_decision: Optional[str] = None,
        date: Optional[datetime.date] = None,
        team1_xi: Optional[list[str]] = None,
        team2_xi: Optional[list[str]] = None,
    ) -> dict:
        """Predict match outcome probability.

        Args:
            team1_xi: Optional playing XI for team1 — uses exact player profiles
                      for stronger prediction vs. rolling historical averages.
            team2_xi: Optional playing XI for team2.

        Returns:
            {
                team1_win_prob: float,
                team2_win_prob: float,
                confidence: str,
                key_factors: list[str],
                xi_used: bool,
            }
        """
        self._ensure_loaded()
        date = date or datetime.date.today()

        row = self._build_win_features(
            team1, team2, venue, toss_winner, toss_decision, date,
            team1_xi=team1_xi, team2_xi=team2_xi,
        )
        X = pd.DataFrame([row])[self._win_features]

        raw_prob = float(self._win_model.predict_proba(X)[:, 1][0])
        if self._platt is not None:
            prob_team1 = float(self._platt.predict_proba([[raw_prob]])[:, 1][0])
        else:
            prob_team1 = raw_prob
        prob_team2 = 1.0 - prob_team1

        confidence = self._confidence_label(prob_team1)
        shap_info = self._explain(self._win_model, X, self._win_features, f"{team1} vs {team2}")

        return {
            "team1_win_prob": round(prob_team1, 4),
            "team2_win_prob": round(prob_team2, 4),
            "confidence": confidence,
            "key_positive_factors": shap_info.get("key_positive_factors", []),
            "key_negative_factors": shap_info.get("key_negative_factors", []),
            "top_features": shap_info.get("top_features", []),
            "xi_used": bool(team1_xi or team2_xi),
        }

    def predict_score(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
        innings: int = 1,
        toss_winner: Optional[str] = None,
        toss_decision: Optional[str] = None,
        date: Optional[datetime.date] = None,
    ) -> dict:
        """Predict first innings score.

        Returns:
            {
                predicted_score: int,
                score_range_low: int,
                score_range_high: int,
                mae_estimate: float,
            }
        """
        self._ensure_loaded()
        date = date or datetime.date.today()

        # Score model uses team1=batting_team, team2=bowling_team convention
        row = self._build_score_features(batting_team, bowling_team, venue, toss_winner, toss_decision, date)
        X = pd.DataFrame([row])[self._score_features]

        predicted = float(self._score_model.predict(X)[0])
        mae = 25.6  # Current model MAE — used as uncertainty estimate

        return {
            "predicted_score": int(round(predicted)),
            "score_range_low": int(round(predicted - mae)),
            "score_range_high": int(round(predicted + mae)),
            "mae_estimate": mae,
        }

    def get_team_list(self) -> list[str]:
        """Return WC 2026 teams first, then other historical teams."""
        wc_teams = list(self._squads.keys()) if self._squads else list(settings.cricket.WC_2026_TEAMS)
        if self._match_df is not None:
            hist = set(self._match_df["team1"].tolist()) | set(self._match_df["team2"].tolist())
            others = sorted(t for t in hist if t and t != "nan" and t not in wc_teams)
            return wc_teams + others
        return sorted(wc_teams)

    def get_venue_list(self) -> list[str]:
        """Return WC 2026 venues first, then all historical venues."""
        wc_venues = list(settings.cricket.WC_2026_VENUES)
        if self._match_df is not None:
            hist = self._match_df["venue"].dropna().unique().tolist()
            others = sorted(str(v) for v in hist if str(v) not in wc_venues)
            return wc_venues + others
        return sorted(wc_venues)

    def get_squad(self, team: str) -> list[dict]:
        """Return the WC 2026 squad for a team (from squads JSON)."""
        if team in self._squads:
            return self._squads[team].get("squad", [])
        # Fall back to historical most-frequent players
        if self._match_df is not None:
            return []
        return []

    def get_all_squads(self) -> dict:
        """Return all WC 2026 squads."""
        return {
            team: {
                "captain": info.get("captain"),
                "squad": [
                    {
                        "name": p["name"],
                        "role": p["role"],
                        "batting_order": p.get("batting_order"),
                        "bowling_style": p.get("bowling_style"),
                        "is_keeper": p.get("is_keeper", False),
                        "is_captain": p.get("is_captain", False),
                        "injury_status": p.get("injury_status", "fit"),
                    }
                    for p in info.get("squad", [])
                ]
            }
            for team, info in self._squads.items()
        }

    def health(self) -> dict:
        """Check if models and data are loaded."""
        return {
            "models_loaded": self._loaded,
            "win_model_features": len(self._win_features),
            "score_model_features": len(self._score_features),
            "match_records": len(self._match_df) if self._match_df is not None else 0,
            "venue_records": len(self._venue_df) if self._venue_df is not None else 0,
        }

    # =========================================================================
    # Feature construction helpers
    # =========================================================================

    def _build_win_features(
        self,
        team1: str,
        team2: str,
        venue: str,
        toss_winner: Optional[str],
        toss_decision: Optional[str],
        date: datetime.date,
        team1_xi: Optional[list[str]] = None,
        team2_xi: Optional[list[str]] = None,
    ) -> dict:
        """Construct win-predictor feature row from team + venue names.

        If playing XIs are provided, uses XIFeatureEngine for exact player-level
        features. Otherwise falls back to rolling historical team averages.
        """
        row: dict = {}

        # Date features
        row["month"] = date.month
        row["day_of_week"] = date.weekday()
        row["year"] = date.year

        # Toss features
        toss_is_team1 = 1 if (toss_winner and toss_winner.lower() == team1.lower()) else 0
        row["toss_winner_is_team1"] = toss_is_team1
        row["elected_to_bat"] = 1 if (toss_decision and toss_decision.lower() == "bat") else 0

        # Venue features
        vf = self._lookup_venue(venue)
        row.update(vf)

        # H2H features
        h2h = self._lookup_h2h(team1, team2)
        row.update(h2h)

        # Team features — use XI engine if available and XI provided
        if self._xi_engine is not None and (team1_xi or team2_xi):
            # Get squad meta for role info
            t1_meta = self._get_squad_meta(team1, team1_xi)
            t2_meta = self._get_squad_meta(team2, team2_xi)

            if team1_xi:
                xi_t1 = self._xi_engine.compute_xi_features(team1_xi, t1_meta, prefix="team1")
                row.update(xi_t1)
            else:
                t1 = self._lookup_team_features(team1, side="team1")
                row.update(t1)

            if team2_xi:
                xi_t2 = self._xi_engine.compute_xi_features(team2_xi, t2_meta, prefix="team2")
                row.update(xi_t2)
            else:
                t2 = self._lookup_team_features(team2, side="team2")
                row.update(t2)

            # Form and experience still come from historical data
            hist_t1 = self._lookup_team_features(team1, side="team1")
            hist_t2 = self._lookup_team_features(team2, side="team2")
            row["team1_form_L10"] = hist_t1.get("team1_form_L10", 0.5)
            row["team2_form_L10"] = hist_t2.get("team2_form_L10", 0.5)
            row["team1_matches_played"] = hist_t1.get("team1_matches_played", 50)
            row["team2_matches_played"] = hist_t2.get("team2_matches_played", 50)
        else:
            # Historical rolling averages
            t1 = self._lookup_team_features(team1, side="team1")
            t2 = self._lookup_team_features(team2, side="team2")
            row.update(t1)
            row.update(t2)

        # Differentials (always recompute from whatever t1/t2 features ended up in row)
        row["form_diff"] = row.get("team1_form_L10", 0.5) - row.get("team2_form_L10", 0.5)
        row["experience_diff"] = row.get("team1_matches_played", 0) - row.get("team2_matches_played", 0)
        row["batting_power_diff"] = row.get("team1_batting_power", 130) - row.get("team2_batting_power", 130)
        row["top_order_form_diff"] = row.get("team1_top3_sr_L10", 125) - row.get("team2_top3_sr_L10", 125)
        row["bowling_economy_diff"] = row.get("team1_bowling_economy", 8.5) - row.get("team2_bowling_economy", 8.5)
        row["dot_ball_pressure_diff"] = row.get("team1_bowling_dot_pct", 35) - row.get("team2_bowling_dot_pct", 35)

        # Fill any remaining expected features with 0
        for feat in self._win_features:
            if feat not in row:
                row[feat] = 0.0

        return row

    def _get_squad_meta(self, team: str, xi: Optional[list[str]]) -> Optional[list[dict]]:
        """Get squad metadata from wc2026_squads.json for role/style info."""
        if team not in self._squads:
            return None
        squad = self._squads[team].get("squad", [])
        if xi:
            return [p for p in squad if p["name"] in xi]
        return squad

    def _build_score_features(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
        toss_winner: Optional[str],
        toss_decision: Optional[str],
        date: datetime.date,
    ) -> dict:
        """Construct score-predictor feature row."""
        row: dict = {}

        row["month"] = date.month
        row["year"] = date.year

        vf = self._lookup_venue(venue)
        row.update(vf)

        # Score model: team1 = batting, team2 = bowling
        t1 = self._lookup_team_features(batting_team, side="team1")
        row.update(t1)

        t2 = self._lookup_team_features(bowling_team, side="team2")
        row.update(t2)

        toss_is_batting = 1 if (toss_winner and toss_winner.lower() == batting_team.lower()) else 0
        row["toss_winner_is_team1"] = toss_is_batting
        row["elected_to_bat"] = 1 if (toss_decision and toss_decision.lower() == "bat") else 0

        for feat in self._score_features:
            if feat not in row:
                row[feat] = 0.0

        return row

    def _lookup_venue(self, venue: str) -> dict:
        """Look up venue features. Falls back to global average if venue unknown."""
        defaults = {
            "venue_avg_1st_inn_score": 155.0,
            "venue_median_1st_inn_score": 153.0,
            "venue_std_1st_inn_score": 25.0,
            "matches_at_venue": 10,
            "venue_avg_powerplay_rpo": 7.5,
            "venue_avg_middle_rpo": 7.0,
            "venue_avg_death_rpo": 10.0,
            "venue_spin_wicket_pct": 35.0,
            "venue_pace_wicket_pct": 65.0,
            "venue_chase_win_pct": 50.0,
            "venue_chase_matches": 10,
            "is_subcontinent": 0,
        }

        if self._venue_df is not None:
            # Try exact match first, then substring
            match = self._venue_df[self._venue_df["venue"].str.lower() == venue.lower()]
            if match.empty:
                match = self._venue_df[
                    self._venue_df["venue"].str.lower().str.contains(
                        venue.lower()[:15], na=False
                    )
                ]

            if not match.empty:
                row = match.iloc[0]
                for key in defaults:
                    if key in row.index and pd.notna(row[key]):
                        defaults[key] = float(row[key])

        # Subcontinent classification from venue name
        venue_lower = venue.lower()
        is_sub = any(kw in venue_lower for kw in settings.cricket.SUBCONTINENT_KEYWORDS)
        defaults["is_subcontinent"] = int(is_sub)

        return defaults

    def _lookup_h2h(self, team1: str, team2: str) -> dict:
        """Compute head-to-head record from historical matches."""
        defaults = {"h2h_matches_played": 5, "h2h_team1_win_rate": 0.5}
        if self._match_df is None:
            return defaults

        df = self._match_df
        # All matches where these two teams played (either orientation)
        h2h_mask = (
            ((df["team1"] == team1) & (df["team2"] == team2)) |
            ((df["team1"] == team2) & (df["team2"] == team1))
        )
        h2h = df[h2h_mask]
        if h2h.empty:
            return defaults

        n = len(h2h)
        # Wins where team1 (our team1) was on winning side
        wins_as_team1 = ((h2h["team1"] == team1) & (h2h["team1_won"] == 1)).sum()
        wins_as_team2 = ((h2h["team2"] == team1) & (h2h["team1_won"] == 0)).sum()
        total_wins = wins_as_team1 + wins_as_team2

        return {
            "h2h_matches_played": n,
            "h2h_team1_win_rate": round(total_wins / n, 4) if n > 0 else 0.5,
        }

    def _lookup_team_features(self, team: str, side: str) -> dict:
        """Extract recent team features from match history.

        Averages the last _RECENT_MATCHES matches where this team appeared as 'side'.
        Falls back to global averages if team not found.
        """
        prefix = side  # "team1" or "team2"
        cols_of_interest = [
            f"{prefix}_form_L10",
            f"{prefix}_matches_played",
            f"{prefix}_batting_power",
            f"{prefix}_top3_sr_L10",
            f"{prefix}_top3_runs_L10",
            f"{prefix}_avg_boundary_pct",
            f"{prefix}_bowling_economy",
            f"{prefix}_bowling_dot_pct",
            f"{prefix}_top_bowler_sr",
            f"{prefix}_spin_bowling_pct",
        ]
        global_defaults = {
            f"{prefix}_form_L10": 0.5,
            f"{prefix}_matches_played": 50,
            f"{prefix}_batting_power": 130.0,
            f"{prefix}_top3_sr_L10": 125.0,
            f"{prefix}_top3_runs_L10": 22.0,
            f"{prefix}_avg_boundary_pct": 12.0,
            f"{prefix}_bowling_economy": 8.5,
            f"{prefix}_bowling_dot_pct": 35.0,
            f"{prefix}_top_bowler_sr": 18.0,
            f"{prefix}_spin_bowling_pct": 40.0,
        }

        if self._match_df is None:
            return global_defaults

        df = self._match_df
        # Look in both team1 and team2 columns — use latest matches regardless of side
        mask_t1 = df["team1"] == team
        mask_t2 = df["team2"] == team

        # Get most recent records from both sides
        t1_rows = df[mask_t1].tail(_RECENT_MATCHES)
        t2_rows = df[mask_t2].tail(_RECENT_MATCHES)

        # Prefer same-side rows (better feature alignment); fall back to opposite
        same_side = df[df[prefix] == team] if prefix in df.columns else pd.DataFrame()

        # Try: rows where team appears as the requested side
        if prefix == "team1":
            own_rows = t1_rows
        else:
            own_rows = t2_rows

        if own_rows.empty:
            # Team appears on the other side — map features across
            other_side = "team2" if prefix == "team1" else "team1"
            other_rows = t2_rows if prefix == "team1" else t1_rows
            if other_rows.empty:
                return global_defaults
            # Remap: team's features are stored under other_side_ prefix
            result = {}
            for col in cols_of_interest:
                other_col = col.replace(prefix, other_side)
                if other_col in other_rows.columns:
                    val = other_rows[other_col].mean()
                    if pd.notna(val):
                        result[col] = round(float(val), 4)
                    else:
                        result[col] = global_defaults[col]
                else:
                    result[col] = global_defaults[col]
            return result

        # Normal case: team appears as the same side
        result = {}
        for col in cols_of_interest:
            if col in own_rows.columns:
                val = own_rows[col].mean()
                result[col] = round(float(val), 4) if pd.notna(val) else global_defaults[col]
            else:
                result[col] = global_defaults[col]

        return result

    # =========================================================================
    # SHAP explainability
    # =========================================================================

    def _explain(
        self,
        model,
        X: pd.DataFrame,
        feature_names: list[str],
        label: str = "Match",
    ) -> dict:
        """Generate SHAP explanation. Returns empty dict on failure (SHAP optional)."""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            sv = shap_values[0] if hasattr(shap_values, "__len__") else shap_values[0]
            importance = pd.DataFrame({
                "feature": feature_names,
                "shap_value": sv,
                "abs_shap": np.abs(sv),
            }).sort_values("abs_shap", ascending=False)

            positive = importance[importance["shap_value"] > 0].head(3)
            negative = importance[importance["shap_value"] < 0].head(3)

            return {
                "key_positive_factors": self._humanise(positive["feature"].tolist(), positive["shap_value"].tolist()),
                "key_negative_factors": self._humanise(negative["feature"].tolist(), negative["shap_value"].tolist()),
                "top_features": importance.head(8).to_dict("records"),
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed ({e}) — returning feature importance fallback")
            return self._feature_importance_fallback(model, feature_names)

    def _feature_importance_fallback(self, model, feature_names: list[str]) -> dict:
        """Fall back to XGBoost gain-based importance when SHAP unavailable."""
        imp = model.feature_importances_
        ranked = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        top = [{"feature": f, "shap_value": float(v), "abs_shap": float(v)} for f, v in ranked[:8]]
        positive = [self._feature_label(f) for f, _ in ranked[:3]]
        return {
            "key_positive_factors": positive,
            "key_negative_factors": [],
            "top_features": top,
        }

    _FEATURE_LABELS = {
        "batting_power_diff": "Team batting strength advantage",
        "team1_batting_power": "Team 1 batting power",
        "team2_batting_power": "Team 2 batting power",
        "team1_form_L10": "Team 1 recent form (last 10)",
        "team2_form_L10": "Team 2 recent form (last 10)",
        "form_diff": "Recent form advantage",
        "venue_chase_win_pct": "Venue chasing success rate",
        "h2h_team1_win_rate": "Historical head-to-head record",
        "team1_avg_boundary_pct": "Team 1 boundary hitting %",
        "team2_avg_boundary_pct": "Team 2 boundary hitting %",
        "team1_bowling_economy": "Team 1 bowling economy",
        "team2_bowling_economy": "Team 2 bowling economy",
        "bowling_economy_diff": "Bowling economy advantage",
        "experience_diff": "International experience gap",
        "team1_spin_bowling_pct": "Team 1 spin bowling proportion",
        "team2_spin_bowling_pct": "Team 2 spin bowling proportion",
        "venue_avg_1st_inn_score": "Typical venue first innings score",
        "is_subcontinent": "Subcontinent spin-friendly conditions",
        "toss_winner_is_team1": "Toss advantage",
        "elected_to_bat": "Elected to bat first",
        "team1_top3_runs_L10": "Team 1 top-order run scoring form",
        "team2_top3_runs_L10": "Team 2 top-order run scoring form",
    }

    def _feature_label(self, feature: str) -> str:
        return self._FEATURE_LABELS.get(feature, feature.replace("_", " ").title())

    def _humanise(self, features: list[str], values: list[float]) -> list[str]:
        return [
            f"{self._feature_label(f)} ({'+' if v > 0 else ''}{v:.3f})"
            for f, v in zip(features, values)
        ]

    # =========================================================================
    # Utility
    # =========================================================================

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @staticmethod
    def _confidence_label(prob: float) -> str:
        margin = abs(prob - 0.5)
        if margin >= 0.25:
            return "High"
        if margin >= 0.12:
            return "Medium"
        return "Low"


# ---------------------------------------------------------------------------
# Singleton accessor — import this in routers
# ---------------------------------------------------------------------------

_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Return the global ModelService, loading it on first call."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelService()
        _service_instance.load()
    return _service_instance
