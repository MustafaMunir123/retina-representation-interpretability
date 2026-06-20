from __future__ import annotations

from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an output path and return the path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_dir(path: str | Path) -> Path:
    """Create an output directory and return it."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
