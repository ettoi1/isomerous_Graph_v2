"""Utility positional encodings for temporal and graph aware modules."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PositionalEncodingConfig:
    """Configuration describing the type of positional encoding to build."""

    dim: int
    max_len: int = 5000
    use_learnable: bool = False


class SinusoidalPositionalEncoding(torch.nn.Module):
    """Classic sinusoidal encoding compatible with Transformer inputs."""

    def __init__(self, config: PositionalEncodingConfig) -> None:
        super().__init__()
        self.config = config
        position = torch.arange(0, config.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.dim)
        )
        pe = torch.zeros(config.max_len, config.dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        if config.use_learnable:
            self.learnable = torch.nn.Parameter(torch.zeros_like(self.pe))
        else:
            self.learnable = None

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Add positional encoding to ``x`` starting from index ``start``.

        Args:
            x: Tensor of shape ``[..., seq_len, dim]``.
            start: Optional offset to shift the encoding window.

        Returns:
            Tensor with positional encodings added.
        """

        seq_len = x.size(-2)
        encoding = self.pe[:, start : start + seq_len]
        if self.learnable is not None:
            encoding = encoding + self.learnable[:, :seq_len]
        return x + encoding


def rotary_positional_embedding(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a simple rotary positional embedding to query/key tensors."""

    rotary_dim = q.size(-1)
    device = q.device
    seq_len = q.size(-2)
    theta = torch.arange(0, rotary_dim, 2, device=device).float()
    theta = 1.0 / (10000 ** (theta / rotary_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    angles = torch.einsum("i,j->ij", seq_idx, theta)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    q_rot = _apply_rotary(q, cos, sin)
    k_rot = _apply_rotary(k, cos, sin)
    return q_rot, k_rot


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.zeros_like(x)
    x_rot[..., ::2] = x1 * cos - x2 * sin
    x_rot[..., 1::2] = x1 * sin + x2 * cos
    return x_rot


def anatomical_system_encoding(num_roi: int, dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Generate a deterministic anatomical/system encoding for each ROI.

    The function assigns ROIs to two pseudo systems (left/right hemisphere) by
    alternating indices. The returned tensor can be added to node features
    before graph level attention is applied.
    """

    device = device or torch.device("cpu")
    half = math.ceil(num_roi / 2)
    encoding = torch.zeros(num_roi, dim, device=device)
    encoding[:half, 0] = 1.0
    encoding[half:, 1] = 1.0
    if dim > 2:
        encoding[:, 2] = torch.linspace(-1.0, 1.0, steps=num_roi, device=device)
    return encoding
