"""CLI script for training models.

Usage:
    python scripts/train.py --model all
    python scripts/train.py --model win
    python scripts/train.py --model score
    python scripts/train.py --pipeline   # Run feature pipeline first
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="CricOracle 2026 — Model Training")
    parser.add_argument(
        "--model",
        choices=["all", "win", "score", "lstm"],
        default="all",
        help="Which model to train",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run feature pipeline before training",
    )
    parser.add_argument(
        "--no-modern-filter",
        action="store_true",
        help="Use ALL historical data (not just modern era 2019+)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CricOracle 2026 — Training Pipeline")
    logger.info("=" * 60)

    # Step 1: Run feature pipeline if requested
    if args.pipeline:
        logger.info("Running feature pipeline...")
        from src.features.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(modern_era_only=not args.no_modern_filter)
        pipeline.run(save=True)

    # Step 2: Train models
    from src.models.trainer import ModelTrainer
    trainer = ModelTrainer()

    if args.model == "all":
        trainer.train_all()
    elif args.model == "win":
        trainer.train_win_model()
    elif args.model == "score":
        trainer.train_score_model()
    elif args.model == "lstm":
        logger.info("LSTM score predictor training — requires ball-by-ball over sequences")
        from src.models.trainer import ModelTrainer
        trainer_lstm = ModelTrainer()
        trainer_lstm.train_lstm_model()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
