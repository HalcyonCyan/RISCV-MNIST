"""Resolve bundled files relative to this project, independent of the cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
