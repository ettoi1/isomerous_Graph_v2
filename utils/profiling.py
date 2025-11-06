"""Lightweight profiling helpers."""
from __future__ import annotations

from typing import Dict

import torch
from torch.nn.parameter import UninitializedParameter


def _is_initialized(param: torch.nn.Parameter) -> bool:
    return not isinstance(param, UninitializedParameter)


def model_parameter_count(model: torch.nn.Module) -> Dict[str, float]:
    """Return parameter counts in total and trainable subsets."""

    total = sum(p.numel() for p in model.parameters() if _is_initialized(p))
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad and _is_initialized(p)
    )
    return {"total": float(total), "trainable": float(trainable)}
