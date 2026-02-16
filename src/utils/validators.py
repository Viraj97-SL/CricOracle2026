"""Pydantic validation schemas for data integrity.

These models validate every row of incoming data before it enters the pipeline.
Think of this like quality control on a production line — bad data gets caught
here instead of silently corrupting your model.

Usage:
    from src.utils.validators import BallRecord, validate_dataframe

    df = pd.read_csv("data.csv")
    clean_df, errors = validate_dataframe(df, BallRecord)
"""

from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, Any
import pandas as pd
from src.utils.logger import logger


class BallRecord(BaseModel):
    """Schema for a single ball-by-ball record from Cricsheet data.

    Validates types, ranges, and basic business rules.
    """

    model_config = {"arbitrary_types_allowed": True}

    match_id: int
    inning_no: int
    over: int
    ball_no: int
    batter: str
    bowler: str
    non_striker: str
    runs_batter: int
    runs_extras: int
    runs_total: int
    is_wicket: int
    batting_team: str
    bowling_team: str
    venue: str
    date: Any  # Accepts str, datetime, Timestamp

    @field_validator("inning_no")
    @classmethod
    def valid_innings(cls, v: int) -> int:
        """T20s have exactly 2 innings (ignore super overs for now)."""
        if v not in (1, 2):
            raise ValueError(f"inning_no must be 1 or 2, got {v}")
        return v

    @field_validator("over")
    @classmethod
    def valid_over(cls, v: int) -> int:
        """Overs range from 0 to 19 in Cricsheet format."""
        if not 0 <= v <= 19:
            raise ValueError(f"over must be 0-19, got {v}")
        return v

    @field_validator("runs_batter", "runs_extras", "runs_total")
    @classmethod
    def non_negative_runs(cls, v: int) -> int:
        """Runs can never be negative."""
        if v < 0:
            raise ValueError(f"runs cannot be negative, got {v}")
        return v

    @field_validator("is_wicket")
    @classmethod
    def valid_wicket_flag(cls, v: int) -> int:
        """Wicket is binary: 0 or 1."""
        if v not in (0, 1):
            raise ValueError(f"is_wicket must be 0 or 1, got {v}")
        return v

    @model_validator(mode="after")
    def teams_are_different(self) -> "BallRecord":
        """Batting and bowling teams must be different."""
        if self.batting_team == self.bowling_team:
            raise ValueError(
                f"batting_team and bowling_team cannot be the same: {self.batting_team}"
            )
        return self

    @model_validator(mode="after")
    def batter_not_bowler(self) -> "BallRecord":
        """Same player can't bat and bowl simultaneously."""
        if self.batter == self.bowler:
            raise ValueError(f"batter and bowler cannot be the same person: {self.batter}")
        return self


class MatchPredictionInput(BaseModel):
    """Schema for match prediction API requests."""

    team1: str
    team2: str
    venue: str
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None
    team1_xi: Optional[list[str]] = None
    team2_xi: Optional[list[str]] = None

    @field_validator("toss_decision")
    @classmethod
    def valid_toss_decision(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in ("bat", "field"):
            raise ValueError(f"toss_decision must be 'bat' or 'field', got '{v}'")
        return v.lower() if v else v

    @model_validator(mode="after")
    def teams_different(self) -> "MatchPredictionInput":
        if self.team1 == self.team2:
            raise ValueError("team1 and team2 must be different")
        return self

    @model_validator(mode="after")
    def toss_winner_is_playing(self) -> "MatchPredictionInput":
        if self.toss_winner and self.toss_winner not in (self.team1, self.team2):
            raise ValueError(
                f"toss_winner '{self.toss_winner}' must be one of the playing teams"
            )
        return self


def validate_dataframe(
    df: pd.DataFrame,
    schema: type[BaseModel] = BallRecord,
    sample_size: int = 5000,
) -> tuple[pd.DataFrame, list[dict]]:
    """Validate a DataFrame against a Pydantic schema.

    Args:
        df: Input DataFrame to validate.
        schema: Pydantic model class to validate against.
        sample_size: Number of rows to validate (for performance on large datasets).

    Returns:
        Tuple of (clean_df, list_of_errors).
        clean_df has invalid rows removed.
        errors list contains {row_index, error_message} dicts.
    """
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    errors = []
    invalid_indices = []

    for idx, row in sample.iterrows():
        try:
            schema(**row.to_dict())
        except Exception as e:
            errors.append({"row_index": idx, "error": str(e)})
            invalid_indices.append(idx)

    error_rate = len(errors) / len(sample) * 100

    if errors:
        logger.warning(
            f"Validation found {len(errors)} errors in {len(sample)} rows "
            f"({error_rate:.1f}% error rate)"
        )
        # Show first 5 errors for debugging
        for err in errors[:5]:
            logger.debug(f"  Row {err['row_index']}: {err['error'][:200]}")
    else:
        logger.info(f"Validation passed: {len(sample)} rows checked, 0 errors")

    # Remove invalid rows from full dataset
    clean_df = df.drop(index=invalid_indices, errors="ignore")

    return clean_df, errors
