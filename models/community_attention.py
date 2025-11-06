"""Dynamic community discovery using slot attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class CommunityAttentionConfig:
    """Configuration for :class:`CommunityAttention`."""

    hidden_dim: int
    num_slots: int
    iterations: int
    dropout: float


class CommunityAttention(torch.nn.Module):
    """Slot attention module producing soft community assignments."""

    def __init__(self, config: CommunityAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.slot_mu = torch.nn.Parameter(torch.randn(config.num_slots, config.hidden_dim))
        self.slot_sigma = torch.nn.Parameter(torch.ones(config.num_slots, config.hidden_dim))
        self.input_proj = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.gru = torch.nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.norm = torch.nn.LayerNorm(config.hidden_dim)
        self.attn = torch.nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        time_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute community assignments.

        Args:
            node_embeddings: Tensor ``[B, R, D]``.
            time_states: Optional tensor ``[B, R, T, D]``.

        Returns:
            ``assign_soft``: ``[B, R, C]`` soft community memberships.
            ``centroids``: ``[B, C, D]`` slot representations.
            ``stats``: Dictionary with auxiliary metrics.
        """

        B, R, D = node_embeddings.shape
        slots = self.slot_mu.unsqueeze(0) + torch.randn(
            B, self.config.num_slots, D, device=node_embeddings.device
        ) * torch.nn.functional.softplus(self.slot_sigma)
        inputs = self.input_proj(node_embeddings)
        if time_states is not None:
            temporal_pool = time_states.mean(dim=2)
            inputs = inputs + temporal_pool
        inputs = self.norm(inputs)
        for _ in range(self.config.iterations):
            attn_logits = torch.einsum("brd,bcd->brc", self.attn(inputs), slots)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.dropout(attn_weights)
            updates = torch.einsum("brc,brd->bcd", attn_weights, inputs)
            slots = slots.reshape(-1, D)
            updates = updates.reshape(-1, D)
            slots = self.gru(updates, slots)
            slots = self.norm(slots)
            slots = slots.reshape(B, self.config.num_slots, D)
        assign_soft = torch.softmax(
            torch.einsum("brd,bcd->brc", inputs, slots), dim=-1
        )
        stats = {
            "entropy": -(assign_soft * assign_soft.log().clamp_min(1e-6)).sum(dim=-1).mean(),
        }
        return assign_soft, slots, stats
