"""Squad optimisation API endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class SquadRequest(BaseModel):
    """Request schema for squad optimisation."""

    team: str
    squad: list[str]
    venue: str
    opponent: Optional[str] = None


class SquadResponse(BaseModel):
    """Response schema for squad optimisation."""

    team: str
    playing_xi: list[str]
    bench: list[str]
    team_strength: float
    reasoning: list[str]


@router.post("/optimise", response_model=SquadResponse)
async def optimise_squad(request: SquadRequest):
    """Select optimal Playing XI from a 15-man squad."""
    # TODO: Load models and run genetic algorithm
    return SquadResponse(
        team=request.team,
        playing_xi=request.squad[:11],
        bench=request.squad[11:],
        team_strength=78.5,
        reasoning=["Placeholder — implement GA optimiser"],
    )
