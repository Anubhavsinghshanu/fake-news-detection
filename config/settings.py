"""Portable paths and project-wide configuration."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def configured_path(env_name: str, default: str) -> Path:
    """Return a configured path; relative values are anchored at project root."""
    raw = os.environ.get(env_name, default).strip()
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

DATA_DIR = configured_path("FAKE_NEWS_DATA_DIR", "data/raw")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = configured_path("FAKE_NEWS_MODEL_DIR", "models")
ARTIFACT_DIR = configured_path("FAKE_NEWS_ARTIFACT_DIR", "artifacts")
RANDOM_SEED = 42
TEXT_COLUMN_CANDIDATES = ("text", "content", "article", "news", "body")
LABEL_COLUMN_CANDIDATES = ("label", "class", "target", "fake", "is_fake")
