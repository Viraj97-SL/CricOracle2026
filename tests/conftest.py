"""Pytest fixtures shared across all tests.

These create sample data that mirrors the structure of real Cricsheet data,
so tests can run without needing the actual 3,000-match CSV.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_ball_data() -> pd.DataFrame:
    """Create a minimal ball-by-ball DataFrame for testing.

    Simulates 2 matches with realistic T20 data structure.
    """
    np.random.seed(42)
    records = []

    teams_matches = [
        {"match_id": 1, "date": "2024-06-01", "venue": "Wankhede Stadium, Mumbai",
         "teams": [("India", "Australia")]},
        {"match_id": 2, "date": "2024-06-05", "venue": "Eden Gardens, Kolkata",
         "teams": [("England", "South Africa")]},
    ]

    for match in teams_matches:
        for inning_no, (batting, bowling) in enumerate(
            [(match["teams"][0][0], match["teams"][0][1]),
             (match["teams"][0][1], match["teams"][0][0])], start=1
        ):
            batters = [f"{batting}_Batter_{i}" for i in range(1, 8)]
            bowlers = [f"{bowling}_Bowler_{i}" for i in range(1, 6)]

            for over in range(20):
                for ball in range(1, 7):
                    runs = np.random.choice([0, 0, 0, 1, 1, 2, 4, 6], p=[0.3, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.05])
                    is_wicket = 1 if np.random.random() < 0.03 else 0

                    records.append({
                        "match_id": match["match_id"],
                        "date": match["date"],
                        "venue": match["venue"],
                        "inning_no": inning_no,
                        "over": over,
                        "ball_no": ball,
                        "batter": np.random.choice(batters),
                        "bowler": np.random.choice(bowlers),
                        "non_striker": np.random.choice(batters),
                        "batting_team": batting,
                        "bowling_team": bowling,
                        "runs_batter": runs,
                        "runs_extras": np.random.choice([0, 0, 0, 1], p=[0.85, 0.05, 0.05, 0.05]),
                        "runs_total": runs + np.random.choice([0, 1], p=[0.9, 0.1]),
                        "is_wicket": is_wicket,
                        "is_four": int(runs == 4),
                        "is_six": int(runs == 6),
                        "is_dot": int(runs == 0),
                    })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Add phase
    df["phase"] = pd.cut(df["over"], bins=[-1, 5, 14, 19],
                          labels=["Powerplay", "Middle", "Death"])
    return df


@pytest.fixture
def sample_csv(tmp_path, sample_ball_data) -> Path:
    """Save sample data to a CSV file for loader tests."""
    csv_path = tmp_path / "test_data.csv"
    sample_ball_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_player_profiles() -> pd.DataFrame:
    """Sample player profiles for model tests."""
    return pd.DataFrame({
        "player_name": [f"Player_{i}" for i in range(15)],
        "batting_role": ["Opener"] * 3 + ["Top Order"] * 4 + ["Middle Order"] * 4 + ["Finisher"] * 4,
        "bowling_style": ["Pace"] * 8 + ["Spin"] * 7,
        "is_wicketkeeper": [False] * 13 + [True] * 2,
        "batting_score": np.random.uniform(30, 90, 15),
        "bowling_score": np.random.uniform(20, 80, 15),
    })
