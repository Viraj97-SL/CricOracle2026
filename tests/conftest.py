"""Shared pytest fixtures for the CricOracle2026 test suite.

These fixtures create synthetic but realistic T20 data that matches
YOUR actual Cricsheet CSV format:
    match_id, date, venue, team1, team2, winner, batting_team,
    over, batter, bowler, non_striker, runs_batter, runs_extra,
    runs_total, wicket_type, player_out
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_ball_data() -> pd.DataFrame:
    """Create synthetic ball-by-ball data matching your actual CSV schema.

    Generates 2 complete matches (2 innings each, 20 overs, 6 balls).
    Total: 2 matches × 2 innings × 20 overs × 6 balls = 480 rows.
    """
    np.random.seed(42)
    rows = []

    matches = [
        {
            "match_id": 9999001,
            "date": "2024-06-01",
            "venue": "Wankhede Stadium, Mumbai",
            "team1": "India",
            "team2": "Australia",
            "winner": "India",
        },
        {
            "match_id": 9999002,
            "date": "2024-06-02",
            "venue": "Eden Gardens, Kolkata",
            "team1": "England",
            "team2": "South Africa",
            "winner": "England",
        },
    ]

    batters = {
        "India": ["Rohit Sharma", "Virat Kohli", "SKY", "Hardik Pandya"],
        "Australia": ["Travis Head", "David Warner", "Glenn Maxwell", "Mitchell Marsh"],
        "England": ["Jos Buttler", "Phil Salt", "Harry Brook", "Liam Livingstone"],
        "South Africa": ["Quinton de Kock", "Aiden Markram", "David Miller", "Heinrich Klaasen"],
    }

    bowlers = {
        "India": ["Jasprit Bumrah", "Mohammed Siraj"],
        "Australia": ["Pat Cummins", "Mitchell Starc"],
        "England": ["Jofra Archer", "Mark Wood"],
        "South Africa": ["Anrich Nortje", "Kagiso Rabada"],
    }

    for match in matches:
        # Innings 1: team1 bats, team2 bowls
        # Innings 2: team2 bats, team1 bowls
        innings_config = [
            (match["team1"], match["team2"]),
            (match["team2"], match["team1"]),
        ]

        for batting_team, bowling_team in innings_config:
            for over in range(20):
                for ball in range(6):
                    batter = np.random.choice(batters[batting_team])
                    bowler = np.random.choice(bowlers[bowling_team])
                    non_striker = np.random.choice(
                        [b for b in batters[batting_team] if b != batter]
                    )

                    runs_batter = np.random.choice([0, 0, 0, 1, 1, 2, 4, 6], p=[0.3, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.05])
                    runs_extra = np.random.choice([0, 0, 0, 0, 1], p=[0.7, 0.1, 0.05, 0.05, 0.1])
                    runs_total = runs_batter + runs_extra

                    # ~5% chance of wicket
                    is_wkt = np.random.random() < 0.05
                    wicket_type = np.random.choice(["bowled", "caught", "lbw", "run out"]) if is_wkt else np.nan
                    player_out = batter if is_wkt else np.nan

                    rows.append({
                        "match_id": match["match_id"],
                        "date": match["date"],
                        "venue": match["venue"],
                        "team1": match["team1"],
                        "team2": match["team2"],
                        "winner": match["winner"],
                        "batting_team": batting_team,
                        "inning_no": innings_config.index((batting_team, bowling_team)) + 1,
                        "over": over,
                        "batter": batter,
                        "bowler": bowler,
                        "non_striker": non_striker,
                        "runs_batter": int(runs_batter),
                        "runs_extra": int(runs_extra),
                        "runs_total": int(runs_total),
                        "is_dot": int(runs_batter == 0),
                        "is_four": int(runs_batter == 4),
                        "is_six": int(runs_batter == 6),
                        "is_wicket": int(is_wkt),
                        "wicket_type": wicket_type,
                        "player_out": player_out,
                    })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(sample_ball_data, tmp_path) -> Path:
    """Save sample data as CSV — mimics your actual file."""
    csv_path = tmp_path / "test_data.csv"
    sample_ball_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def loaded_data(sample_csv):
    """Load and clean sample data through the full pipeline."""
    from src.data.loader import CricketDataLoader
    loader = CricketDataLoader(filepath=sample_csv)
    return loader.load_and_clean(modern_era_only=False)


@pytest.fixture
def sample_player_profiles() -> pd.DataFrame:
    """Pre-built player profiles for feature/model testing."""
    return pd.DataFrame({
        "batter": [
            "Rohit Sharma", "Virat Kohli", "SKY", "Jos Buttler",
            "Travis Head", "Phil Salt", "Quinton de Kock",
            "Glenn Maxwell", "Harry Brook", "Liam Livingstone",
            "David Miller", "Heinrich Klaasen", "Hardik Pandya",
            "Mitchell Marsh", "Aiden Markram",
        ],
        "batting_team": [
            "India", "India", "India", "England",
            "Australia", "England", "South Africa",
            "Australia", "England", "England",
            "South Africa", "South Africa", "India",
            "Australia", "South Africa",
        ],
        "matches": [50, 55, 40, 45, 35, 30, 48, 42, 28, 32, 50, 38, 44, 36, 40],
        "innings": [48, 52, 38, 43, 33, 29, 46, 40, 26, 30, 48, 36, 42, 34, 38],
        "total_runs": [1400, 1600, 1200, 1350, 1050, 900, 1300, 1100, 850, 950, 1250, 1050, 1150, 950, 1000],
        "balls_faced": [1050, 1200, 850, 950, 750, 620, 980, 780, 580, 650, 950, 720, 830, 700, 760],
        "strike_rate": [133.3, 133.3, 141.2, 142.1, 140.0, 145.2, 132.7, 141.0, 146.6, 146.2, 131.6, 145.8, 138.6, 135.7, 131.6],
        "avg": [29.2, 30.8, 31.6, 31.4, 31.8, 31.0, 28.3, 27.5, 32.7, 31.7, 26.0, 29.2, 27.4, 27.9, 26.3],
    })
