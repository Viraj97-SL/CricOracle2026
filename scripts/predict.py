"""CLI script for making predictions.

Usage:
    python scripts/predict.py --team1 "India" --team2 "Australia" --venue "Wankhede Stadium, Mumbai"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="CricOracle 2026 — Match Predictor")
    parser.add_argument("--team1", required=True, help="Team 1 name")
    parser.add_argument("--team2", required=True, help="Team 2 name")
    parser.add_argument("--venue", required=True, help="Venue name")
    parser.add_argument("--toss-winner", help="Toss winner")
    parser.add_argument("--toss-decision", choices=["bat", "field"], help="Toss decision")
    args = parser.parse_args()

    logger.info(f"Predicting: {args.team1} vs {args.team2} at {args.venue}")

    # TODO: Load trained model and make prediction
    # from src.models.win_predictor import WinPredictor
    # predictor = WinPredictor()
    # predictor.load()
    # result = predictor.predict(...)

    logger.info("Prediction complete (placeholder — train models first)")


if __name__ == "__main__":
    main()
