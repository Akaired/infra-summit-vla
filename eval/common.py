"""Shared helpers: repository-root path resolution and YAML loading.

Every path this package reads is resolved against the repository root, never
against the current working directory, so the harness runs identically no
matter where it is launched from. Same convention as ``data/dataloader.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    """Resolve ``value`` against the repository root unless already absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file addressed relative to the repository root."""
    with resolve_path(path).open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a top-level mapping, got {type(loaded).__name__}")
    return loaded
