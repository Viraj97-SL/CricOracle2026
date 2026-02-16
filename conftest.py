"""Root conftest.py — ensures `src` package is importable in tests.

This file exists at the project root so pytest can discover the src package
without needing `pip install -e .` during development.
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))
