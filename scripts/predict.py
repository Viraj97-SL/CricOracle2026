"""CLI script for making predictions.

Usage:
    python scripts/predict.py --team1 "India" --team2 "Australia" --venue "Wankhede Stadium, Mumbai"
    python scripts/predict.py --team1 "India" --team2 "Australia" --venue "Wankhede Stadium, Mumbai" \\
        --toss-winner "India" --toss-decision bat
    python scripts/predict.py --score --batting "India" --bowling "Australia" --venue "Wankhede Stadium, Mumbai"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger
from src.models.model_service import get_model_service


def predict_match(args) -> None:
    svc = get_model_service()
    result = svc.predict_match(
        team1=args.team1,
        team2=args.team2,
        venue=args.venue,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
    )

    print()
    print("=" * 55)
    print(f"  {args.team1} vs {args.team2}")
    print(f"  @ {args.venue}")
    print("=" * 55)
    print(f"  {args.team1:30s}  {result['team1_win_prob']*100:5.1f}%")
    print(f"  {args.team2:30s}  {result['team2_win_prob']*100:5.1f}%")
    print(f"  Confidence: {result['confidence']}")
    print()
    print("  Key factors:")
    for factor in result.get("key_positive_factors", []):
        print(f"    + {factor}")
    for factor in result.get("key_negative_factors", []):
        print(f"    - {factor}")
    print("=" * 55)


def predict_score(args) -> None:
    svc = get_model_service()
    result = svc.predict_score(
        batting_team=args.batting,
        bowling_team=args.bowling,
        venue=args.venue,
        toss_winner=args.toss_winner,
        toss_decision=args.toss_decision,
    )

    print()
    print("=" * 55)
    print(f"  Score Prediction")
    print(f"  {args.batting} batting vs {args.bowling}")
    print(f"  @ {args.venue}")
    print("=" * 55)
    print(f"  Predicted score:  {result['predicted_score']}")
    print(f"  Range:            {result['score_range_low']}–{result['score_range_high']}")
    print(f"  Model MAE:        ±{result['mae_estimate']:.0f} runs")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="CricOracle 2026 — Prediction CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Match prediction (default)
    match_parser = parser.add_argument_group("Match prediction")
    parser.add_argument("--team1", help="Team 1 name")
    parser.add_argument("--team2", help="Team 2 name")
    parser.add_argument("--venue", required=True, help="Venue name")
    parser.add_argument("--toss-winner", dest="toss_winner", help="Toss winner")
    parser.add_argument("--toss-decision", dest="toss_decision",
                        choices=["bat", "field"], help="Toss decision")

    # Score prediction mode
    parser.add_argument("--score", action="store_true", help="Predict score instead of match outcome")
    parser.add_argument("--batting", help="Batting team (for --score mode)")
    parser.add_argument("--bowling", help="Bowling team (for --score mode)")

    args = parser.parse_args()

    if args.score:
        if not args.batting or not args.bowling:
            parser.error("--score requires --batting and --bowling")
        predict_score(args)
    else:
        if not args.team1 or not args.team2:
            parser.error("Match prediction requires --team1 and --team2")
        predict_match(args)


if __name__ == "__main__":
    main()
