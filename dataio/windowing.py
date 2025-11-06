"""Temporal windowing helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WindowConfig:
    """Sliding window configuration."""

    window_size: int
    stride: int


def build_time_windows(roi_ts: torch.Tensor, config: WindowConfig) -> torch.Tensor:
    """Return time windows extracted from ``roi_ts``.

    Args:
        roi_ts: Tensor of shape ``[R, T]`` or ``[R, T, F]``.
        config: Sliding window configuration.

    Returns:
        Tensor of shape ``[num_windows, R, window_size, F]`` if the input has a
        frequency channel, otherwise ``[num_windows, R, window_size]``.
    """

    if roi_ts.dim() == 3:
        roi_ts = roi_ts.unsqueeze(0)
    R, T = roi_ts.shape[0], roi_ts.shape[1]
    window_size = config.window_size
    stride = config.stride
    windows = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(roi_ts[:, start:end, ...])
    stacked = torch.stack(windows) if windows else torch.empty(0)
    return stacked
