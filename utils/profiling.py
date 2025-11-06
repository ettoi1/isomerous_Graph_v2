"""Lightweight profiling helpers."""
from __future__ import annotations

from typing import Dict

import torch


def model_parameter_count(model: torch.nn.Module) -> Dict[str, float]:
    """Return parameter counts in total and trainable subsets."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": float(total), "trainable": float(trainable)}
