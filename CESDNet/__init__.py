"""Allow `python -m CESDNet.<command>` from a standalone repository checkout."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1])]
