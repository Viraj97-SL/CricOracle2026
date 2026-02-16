"""Centralised logging configuration using Loguru.

Usage:
    from src.utils.logger import logger

    logger.info("Loading data...")
    logger.warning("Missing 42 rows")
    logger.error("Model training failed", exc_info=True)
"""

import sys
from loguru import logger as _logger
from src.config import settings

# Remove default handler
_logger.remove()

# Console handler (colourful, human-readable)
_logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# File handler (structured, for debugging)
_logger.add(
    settings.paths.logs / "cricoracle_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
)

# Export as 'logger'
logger = _logger
