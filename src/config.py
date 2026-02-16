"""Central configuration for the CricOracle 2026 project.

All hyperparameters, file paths, constants, and API keys are defined here.
Import this module everywhere instead of hardcoding values.

Usage:
    from src.config import settings, Paths, ModelParams
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


# ===========================
# File Paths
# ===========================
PROJECT_ROOT = Path(__file__).parent.parent  # CricOracle2026/


@dataclass(frozen=True)
class Paths:
    """All file paths used in the project. Frozen = immutable after creation."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    data_external: Path = PROJECT_ROOT / "data" / "external"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    models: Path = PROJECT_ROOT / "models"
    outputs: Path = PROJECT_ROOT / "outputs"
    logs: Path = PROJECT_ROOT / "logs"

    # Key data files
    ball_by_ball_csv: Path = PROJECT_ROOT / "data" / "raw" / "t20_ball_by_ball_v2.csv"
    processed_features: Path = PROJECT_ROOT / "data" / "processed" / "match_features.parquet"
    player_profiles: Path = PROJECT_ROOT / "data" / "processed" / "player_profiles.parquet"
    venue_features: Path = PROJECT_ROOT / "data" / "processed" / "venue_features.parquet"

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for attr_name in [
            "data_raw", "data_processed", "data_external",
            "data_interim", "models", "outputs", "logs",
        ]:
            getattr(self, attr_name).mkdir(parents=True, exist_ok=True)


# ===========================
# API Keys (loaded from .env)
# ===========================
@dataclass(frozen=True)
class APIKeys:
    """External API keys. Loaded from .env file — never hardcode these."""

    openweather: str = os.getenv("OPENWEATHER_API_KEY", "")
    news_api: str = os.getenv("NEWS_API_KEY", "")
    twitter: str = os.getenv("TWITTER_BEARER_TOKEN", "")


# ===========================
# Cricket Domain Constants
# ===========================
@dataclass(frozen=True)
class CricketConstants:
    """Domain-specific constants for T20 cricket."""

    # Phase definitions (overs)
    POWERPLAY_START: int = 1
    POWERPLAY_END: int = 6
    MIDDLE_OVERS_START: int = 7
    MIDDLE_OVERS_END: int = 15
    DEATH_OVERS_START: int = 16
    DEATH_OVERS_END: int = 20

    # Rolling window sizes for player form
    FORM_WINDOW_SHORT: int = 5     # Last 5 innings (hot form)
    FORM_WINDOW_MEDIUM: int = 10   # Last 10 innings (consistent form)
    FORM_WINDOW_LONG: int = 15     # Last 15 innings (baseline)

    # Minimum sample sizes (to avoid noisy stats)
    MIN_BALLS_BATTER: int = 60     # At least 10 overs faced
    MIN_BALLS_BOWLER: int = 120    # At least 20 overs bowled
    MIN_MATCHES_PLAYER: int = 10   # Minimum matches for reliable stats

    # Modern era filter (T20 cricket has evolved significantly)
    MODERN_ERA_START: str = "2019-01-01"

    # T20 World Cup 2026 venues
    WC_2026_VENUES: tuple = (
        "Narendra Modi Stadium, Ahmedabad",
        "M.A.Chidambaram Stadium, Chennai",
        "Arun Jaitley Stadium, Delhi",
        "Eden Gardens, Kolkata",
        "Wankhede Stadium, Mumbai",
        "R.Premadasa Stadium, Colombo",
        "Sinhalese Sports Club Ground, Colombo",
        "Pallekele International Cricket Stadium",
    )

    # Venue coordinates for weather API
    VENUE_COORDS: dict = field(default_factory=lambda: {
        "Narendra Modi Stadium, Ahmedabad": (23.0916, 72.5970),
        "M.A.Chidambaram Stadium, Chennai": (13.0629, 80.2792),
        "Arun Jaitley Stadium, Delhi": (28.6370, 77.2433),
        "Eden Gardens, Kolkata": (22.5646, 88.3433),
        "Wankhede Stadium, Mumbai": (19.0448, 72.8224),
        "R.Premadasa Stadium, Colombo": (6.9157, 79.8636),
        "Sinhalese Sports Club Ground, Colombo": (6.9053, 79.8636),
        "Pallekele International Cricket Stadium": (7.2709, 80.6356),
    })

    # T20 WC 2026 teams (20 teams)
    WC_2026_TEAMS: tuple = (
        "India", "Pakistan", "Australia", "England", "New Zealand",
        "South Africa", "Sri Lanka", "Afghanistan", "West Indies",
        "Scotland", "Ireland", "Netherlands", "Nepal", "Namibia",
        "Oman", "Zimbabwe", "Canada", "United States of America",
        "United Arab Emirates", "Italy",
    )

    # Subcontinent venues (spin-friendly classification)
    SUBCONTINENT_KEYWORDS: tuple = (
        "colombo", "kandy", "galle", "hambantota", "pallekele", "dambulla",
        "mumbai", "wankhede", "delhi", "kotla", "jaitley",
        "kolkata", "eden gardens", "chennai", "chepauk", "chidambaram",
        "bangalore", "bengaluru", "hyderabad", "ahmedabad", "motera",
        "lucknow", "pune", "nagpur", "indore", "rajkot", "ranchi",
        "dhaka", "chittagong", "sylhet",
        "lahore", "karachi", "rawalpindi", "multan",
        "dubai", "abu dhabi", "sharjah",
    )


# ===========================
# Model Hyperparameters
# ===========================
@dataclass
class XGBoostParams:
    """XGBoost hyperparameters. Mutable — Optuna will tune these."""

    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 50
    random_state: int = 42


@dataclass
class LSTMParams:
    """LSTM hyperparameters for score prediction."""

    input_dim: int = 10       # Features per over
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    max_overs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    patience: int = 15        # Early stopping patience
    grad_clip: float = 1.0


@dataclass
class TrainingConfig:
    """General training configuration."""

    train_split: float = 0.70    # 70% train
    val_split: float = 0.15      # 15% validation
    test_split: float = 0.15     # 15% test
    random_state: int = 42
    use_time_split: bool = True  # CRITICAL: Always True for cricket (temporal data)
    optuna_trials: int = 200     # Number of Bayesian optimisation trials


# ===========================
# Global Settings Object
# ===========================
@dataclass
class Settings:
    """Master settings object. Import this everywhere."""

    PROJECT_NAME: str = "CricOracle2026"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    paths: Paths = field(default_factory=Paths)
    api_keys: APIKeys = field(default_factory=APIKeys)
    cricket: CricketConstants = field(default_factory=CricketConstants)
    xgboost: XGBoostParams = field(default_factory=XGBoostParams)
    lstm: LSTMParams = field(default_factory=LSTMParams)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Singleton instance — import this
settings = Settings()

# Ensure directories exist on import
settings.paths.ensure_dirs()
