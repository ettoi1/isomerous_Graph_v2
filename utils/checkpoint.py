"""Checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Persist a checkpoint to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint from disk."""

    return torch.load(path, map_location="cpu")
