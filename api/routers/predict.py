"""Prediction API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional

router = APIRouter()


class MatchPredictionRequest(BaseModel):
    """Request schema for match prediction."""

    team1: str
    team2: str
    venue: str
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None
    team1_xi: Optional[list[str]] = None
    team2_xi: Optional[list[str]] = None

    @field_validator("toss_decision")
    @classmethod
    def validate_toss(cls, v):
        if v and v.lower() not in ("bat", "field"):
            raise ValueError("toss_decision must be 'bat' or 'field'")
        return v


class MatchPredictionResponse(BaseModel):
    """Response schema for match prediction."""

    team1: str
    team2: str
    team1_win_probability: float
    team2_win_probability: float
    predicted_score_team1: Optional[int] = None
    predicted_score_team2: Optional[int] = None
    confidence: str
    key_factors: list[str]
    model_version: str


class ScorePredictionRequest(BaseModel):
    """Request schema for score prediction."""

    batting_team: str
    bowling_team: str
    venue: str
    innings: int = 1


class ScorePredictionResponse(BaseModel):
    """Response schema for score prediction."""

    batting_team: str
    venue: str
    predicted_score: int
    score_range_low: int
    score_range_high: int
    model_version: str


@router.post("/match", response_model=MatchPredictionResponse)
async def predict_match(request: MatchPredictionRequest):
    """Predict match outcome (win probability).

    Returns calibrated win probabilities for both teams,
    predicted scores, and SHAP-derived key factors.
    """
    # TODO: Load model and generate real predictions
    # For now, return placeholder to verify API structure works
    return MatchPredictionResponse(
        team1=request.team1,
        team2=request.team2,
        team1_win_probability=0.55,
        team2_win_probability=0.45,
        predicted_score_team1=165,
        predicted_score_team2=158,
        confidence="Medium",
        key_factors=[
            "Home advantage at venue",
            "Strong recent batting form",
            "Head-to-head record favours team1",
        ],
        model_version="0.1.0-placeholder",
    )


@router.post("/score", response_model=ScorePredictionResponse)
async def predict_score(request: ScorePredictionRequest):
    """Predict first innings score."""
    # TODO: Load model and generate real predictions
    return ScorePredictionResponse(
        batting_team=request.batting_team,
        venue=request.venue,
        predicted_score=162,
        score_range_low=148,
        score_range_high=176,
        model_version="0.1.0-placeholder",
    )
