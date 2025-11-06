"""Attention utilities shared across the project."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .positional_encodings import rotary_positional_embedding


class MultiheadAttentionWithRope(torch.nn.Module):
    """Multi-head attention supporting optional rotary positional encoding."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.use_rope = use_rope

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass applying optional rotary positional encoding."""

        if self.use_rope:
            query, key = rotary_positional_embedding(query, key)
        attn_output, attn_weights = self.mha(
            query,
            key,
            value,
            attn_mask=attn_mask,
            need_weights=True,
        )
        return attn_output, attn_weights


def sparse_attention_mask(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Create an attention mask that only allows attending to connected nodes."""

    device = edge_index.device
    mask = torch.full((num_nodes, num_nodes), float("-inf"), device=device)
    mask.fill_diagonal_(0.0)
    if edge_index.numel() == 0:
        return mask
    mask[edge_index[0], edge_index[1]] = 0.0
    return mask


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return a mask selecting the top ``k`` entries per row of ``scores``."""

    topk = torch.topk(scores, k=min(k, scores.size(-1)), dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, topk, True)
    return mask


def softmax_with_temperature(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature controlled softmax used by routing components."""

    scaled = scores / max(temperature, 1e-6)
    return F.softmax(scaled, dim=-1)
