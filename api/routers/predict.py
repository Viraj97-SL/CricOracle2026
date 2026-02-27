"""Prediction API endpoints — wired to real trained models."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional
import datetime

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class MatchPredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None
    team1_xi: Optional[list[str]] = None   # Playing XI for team1 (11 names)
    team2_xi: Optional[list[str]] = None   # Playing XI for team2 (11 names)

    @field_validator("toss_decision")
    @classmethod
    def validate_toss(cls, v):
        if v and v.lower() not in ("bat", "field"):
            raise ValueError("toss_decision must be 'bat' or 'field'")
        return v


class MatchPredictionResponse(BaseModel):
    team1: str
    team2: str
    team1_win_probability: float
    team2_win_probability: float
    predicted_score_team1: Optional[int] = None
    predicted_score_team2: Optional[int] = None
    confidence: str
    key_factors: list[str]
    xi_used: bool = False
    model_version: str


class SquadInfoResponse(BaseModel):
    team: str
    captain: Optional[str]
    squad: list[dict]


class ScorePredictionRequest(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str
    innings: int = 1
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None


class ScorePredictionResponse(BaseModel):
    batting_team: str
    venue: str
    predicted_score: int
    score_range_low: int
    score_range_high: int
    model_version: str


class TeamListResponse(BaseModel):
    teams: list[str]
    venues: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/teams", response_model=TeamListResponse)
async def list_teams():
    """List all available teams and venues for the prediction UI."""
    from src.models.model_service import get_model_service
    svc = get_model_service()
    return TeamListResponse(teams=svc.get_team_list(), venues=svc.get_venue_list())


@router.get("/squads", summary="All WC 2026 squads")
async def get_all_squads():
    """Return WC 2026 squad for all teams (names, roles, injury status)."""
    from src.models.model_service import get_model_service
    svc = get_model_service()
    return svc.get_all_squads()


@router.get("/squads/{team}", response_model=SquadInfoResponse)
async def get_squad(team: str):
    """Return WC 2026 squad for a specific team."""
    from src.models.model_service import get_model_service
    from fastapi import HTTPException
    svc = get_model_service()
    squads = svc.get_all_squads()
    if team not in squads:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found in WC 2026 squads")
    info = squads[team]
    return SquadInfoResponse(team=team, captain=info.get("captain"), squad=info.get("squad", []))


@router.post("/match", response_model=MatchPredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    """Predict match outcome (win probability) with SHAP-derived key factors.

    Uses a trained XGBoost model with 44 pre-match features covering:
    team form, batting/bowling strength, venue stats, head-to-head record,
    and toss information.
    """
    from src.models.model_service import get_model_service
    try:
        svc = get_model_service()
        result = svc.predict_match(
            team1=request.team1,
            team2=request.team2,
            venue=request.venue,
            toss_winner=request.toss_winner,
            toss_decision=request.toss_decision,
            team1_xi=request.team1_xi,
            team2_xi=request.team2_xi,
        )

        # Also get score estimates
        score1 = svc.predict_score(
            batting_team=request.team1,
            bowling_team=request.team2,
            venue=request.venue,
            toss_winner=request.toss_winner,
            toss_decision=request.toss_decision,
        )
        score2 = svc.predict_score(
            batting_team=request.team2,
            bowling_team=request.team1,
            venue=request.venue,
        )

        key_factors = (
            result.get("key_positive_factors", []) +
            result.get("key_negative_factors", [])
        )[:5]
        if not key_factors:
            key_factors = ["Prediction based on historical match data"]

        return MatchPredictionResponse(
            team1=request.team1,
            team2=request.team2,
            team1_win_probability=result["team1_win_prob"],
            team2_win_probability=result["team2_win_prob"],
            predicted_score_team1=score1["predicted_score"],
            predicted_score_team2=score2["predicted_score"],
            confidence=result["confidence"],
            key_factors=key_factors,
            xi_used=result.get("xi_used", False),
            model_version="1.0.0",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/score", response_model=ScorePredictionResponse)
async def predict_score(request: ScorePredictionRequest):
    """Predict first innings score given batting team, bowling team and venue."""
    from src.models.model_service import get_model_service
    try:
        svc = get_model_service()
        result = svc.predict_score(
            batting_team=request.batting_team,
            bowling_team=request.bowling_team,
            venue=request.venue,
            innings=request.innings,
            toss_winner=request.toss_winner,
            toss_decision=request.toss_decision,
        )
        return ScorePredictionResponse(
            batting_team=request.batting_team,
            venue=request.venue,
            predicted_score=result["predicted_score"],
            score_range_low=result["score_range_low"],
            score_range_high=result["score_range_high"],
            model_version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Score prediction failed: {str(e)}")
