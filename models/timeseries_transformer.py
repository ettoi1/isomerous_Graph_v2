"""Temporal Transformer backbone for ROI time-series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from layers.positional_encodings import PositionalEncodingConfig, SinusoidalPositionalEncoding


@dataclass
class TimeSeriesTransformerConfig:
    """Configuration parameters for :class:`TimeSeriesTransformer`."""

    input_channels: int
    hidden_dim: int
    num_heads: int
    dropout: float
    use_frequency: bool = False
    use_rope: bool = False
    window_size: Optional[int] = None


class TimeSeriesTransformer(torch.nn.Module):
    """Encode ROI time-series into node embeddings using a Transformer encoder.

    Expected tensor shapes are documented explicitly to keep consistency across
    the project:

    - ``roi_ts``: ``[B, R, T]`` or ``[B, R, T, F]``
    - ``time_states``: optional tensor ``[B, R, T, D]`` containing intermediate
      temporal states used by downstream modules such as the router or the
      community attention mechanism.
    """

    def __init__(self, config: TimeSeriesTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = torch.nn.Linear(config.input_channels, config.hidden_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.positional = SinusoidalPositionalEncoding(
            PositionalEncodingConfig(dim=config.hidden_dim)
        )

    def forward(self, roi_ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ROI time-series.

        Args:
            roi_ts: Tensor shaped ``[B, R, T]`` or ``[B, R, T, F]``.

        Returns:
            ``node_embeddings``: ``[B, R, D]`` with node representations.
            ``time_states``: ``[B, R, T, D]`` with temporal hidden states.
        """

        if roi_ts.dim() == 3:
            roi_ts = roi_ts.unsqueeze(-1)
        B, R, T, F = roi_ts.shape
        tokens = roi_ts.reshape(B * R, T, F)
        tokens = self.input_proj(tokens)
        tokens = self.positional(tokens)
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        encoded = self.encoder(tokens)
        cls_state = encoded[:, 0]
        time_states = encoded[:, 1:]
        node_embeddings = cls_state.reshape(B, R, -1)
        time_states = time_states.reshape(B, R, T, -1)
        return node_embeddings, time_states
