"""Squad optimisation API endpoints — wired to GA-based SquadOptimiser."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class SquadRequest(BaseModel):
    team: str
    squad: list[str]
    venue: str
    opponent: Optional[str] = None


class SquadResponse(BaseModel):
    team: str
    playing_xi: list[str]
    bench: list[str]
    team_strength: float
    balance_score: float
    constraint_violations: list[str]
    reasoning: list[str]


@router.post("/optimise", response_model=SquadResponse)
async def optimise_squad(request: SquadRequest):
    """Select optimal Playing XI from a 15-man squad using genetic algorithm.

    Constraints enforced:
    - Exactly 11 players
    - ≥5 batters, ≥1 wicketkeeper
    - ≥1 spinner for subcontinent venues
    - Balance between batting and bowling strength

    Player profiles must be in the historical dataset for best results.
    Unknown players are scored with fallback averages.
    """
    if len(request.squad) < 11:
        raise HTTPException(
            status_code=422,
            detail=f"Squad must have at least 11 players, got {len(request.squad)}",
        )

    try:
        import pandas as pd
        from src.config import settings
        from src.models.squad_optimiser import SquadOptimiser

        # Load player profiles
        profiles_path = settings.paths.data_processed / "batting_profiles.parquet"
        bowling_path = settings.paths.data_processed / "bowling_profiles.parquet"
        roles_path = settings.paths.data_processed / "batting_roles.csv"
        styles_path = settings.paths.data_processed / "bowling_styles.csv"

        if not profiles_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Player profiles not found. Run the feature pipeline first.",
            )

        batting_profiles = pd.read_parquet(profiles_path)
        bowling_profiles = pd.read_parquet(bowling_path)

        # Build merged player profile with required columns
        profiles = _build_squad_profiles(
            request.squad, batting_profiles, bowling_profiles,
            roles_path, styles_path
        )

        optimiser = SquadOptimiser(player_profiles=profiles)
        result = optimiser.select_xi(
            squad=request.squad,
            venue=request.venue,
            opponent=request.opponent,
            population_size=300,
            generations=100,
        )

        reasoning = [
            f"Team strength score: {result.team_strength_score:.1f}",
            f"Batting score: {result.batting_score:.1f}",
            f"Bowling score: {result.bowling_score:.1f}",
            f"Squad balance: {result.balance_score:.2f}",
        ]
        if result.constraint_violations:
            reasoning.extend([f"Warning: {v}" for v in result.constraint_violations])

        return SquadResponse(
            team=request.team,
            playing_xi=result.playing_xi,
            bench=result.bench,
            team_strength=result.team_strength_score,
            balance_score=result.balance_score,
            constraint_violations=result.constraint_violations,
            reasoning=reasoning,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Squad optimisation failed: {str(e)}")


def _build_squad_profiles(
    squad: list[str],
    batting_profiles,
    bowling_profiles,
    roles_path,
    styles_path,
) -> "pd.DataFrame":
    """Merge batting + bowling profiles into a single player DataFrame."""
    import pandas as pd
    import numpy as np

    batting_profiles = batting_profiles.copy()
    bowling_profiles = bowling_profiles.copy()

    # Normalise player name column
    if "batter" in batting_profiles.columns:
        batting_profiles = batting_profiles.rename(columns={"batter": "player_name"})
    if "bowler" in bowling_profiles.columns:
        bowling_profiles = bowling_profiles.rename(columns={"bowler": "player_name"})

    batting_profiles["player_name"] = batting_profiles["player_name"].astype(str)
    bowling_profiles["player_name"] = bowling_profiles["player_name"].astype(str)

    # Compute batting_score = strike_rate (normalised)
    if "strike_rate" in batting_profiles.columns:
        batting_profiles["batting_score"] = batting_profiles["strike_rate"].fillna(
            batting_profiles["strike_rate"].median()
        )

    # Compute bowling_score = inverse of economy (lower economy = better)
    if "economy" in bowling_profiles.columns:
        median_eco = bowling_profiles["economy"].median()
        bowling_profiles["bowling_score"] = (
            20.0 / bowling_profiles["economy"].fillna(median_eco).clip(lower=1.0)
        )

    # Merge
    merged = batting_profiles[["player_name", "batting_score"]].merge(
        bowling_profiles[["player_name", "bowling_score"]],
        on="player_name",
        how="outer",
    )
    merged["batting_score"] = merged["batting_score"].fillna(merged["batting_score"].median())
    merged["bowling_score"] = merged["bowling_score"].fillna(merged["bowling_score"].median())

    # Attach batting roles
    if roles_path.exists():
        roles = pd.read_csv(roles_path)
        if "batter" in roles.columns:
            roles = roles.rename(columns={"batter": "player_name"})
        merged = merged.merge(roles[["player_name", "batting_role"]], on="player_name", how="left")
    if "batting_role" not in merged.columns:
        merged["batting_role"] = "Middle Order"

    # Attach bowling styles
    if styles_path.exists():
        styles = pd.read_csv(styles_path)
        if "bowler" in styles.columns:
            styles = styles.rename(columns={"bowler": "player_name"})
        merged = merged.merge(styles[["player_name", "bowling_style"]], on="player_name", how="left")
    if "bowling_style" not in merged.columns:
        merged["bowling_style"] = "Pace"

    merged["is_wicketkeeper"] = False

    # Add fallback rows for squad members not in profiles
    missing = [p for p in squad if p not in merged["player_name"].values]
    if missing:
        fallback_bat = merged["batting_score"].median() if len(merged) > 0 else 130.0
        fallback_bowl = merged["bowling_score"].median() if len(merged) > 0 else 2.3
        fallback_rows = pd.DataFrame({
            "player_name": missing,
            "batting_score": fallback_bat,
            "bowling_score": fallback_bowl,
            "batting_role": "Middle Order",
            "bowling_style": "Pace",
            "is_wicketkeeper": False,
        })
        merged = pd.concat([merged, fallback_rows], ignore_index=True)

    # Filter to squad only
    merged = merged[merged["player_name"].isin(squad)].reset_index(drop=True)
    return merged
