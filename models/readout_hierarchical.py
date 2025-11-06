"""Hierarchical attention based readout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class HierarchicalReadoutConfig:
    """Configuration for :class:`HierarchicalReadout`."""

    hidden_dim: int
    num_classes: int
    temperature: float = 1.0


class HierarchicalReadout(torch.nn.Module):
    """Aggregate representations using hierarchical attention blocks."""

    def __init__(self, config: HierarchicalReadoutConfig) -> None:
        super().__init__()
        self.config = config
        self.node_attn = torch.nn.Linear(config.hidden_dim, 1)
        self.community_attn = torch.nn.Linear(config.hidden_dim, 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        community_assign: torch.Tensor,
        community_centers: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform hierarchical pooling.

        Args:
            node_embeddings: ``[B, R, D]`` node features.
            community_assign: ``[B, R, C]`` soft community memberships.
            community_centers: Optional ``[B, C, D]`` centroids.

        Returns:
            ``logits``: ``[B, num_classes]`` raw predictions.
            ``probs``: calibrated probabilities.
            ``attn_info``: diagnostic dictionary.
        """

        node_scores = torch.softmax(
            self.node_attn(node_embeddings) / max(self.config.temperature, 1e-6), dim=1
        )
        node_summary = (node_scores.transpose(1, 2) @ node_embeddings).squeeze(1)
        community_summary = torch.einsum("brc,brd->bcd", community_assign, node_embeddings)
        if community_centers is not None:
            community_summary = (community_summary + community_centers) / 2.0
        community_scores = torch.softmax(
            self.community_attn(community_summary) / max(self.config.temperature, 1e-6),
            dim=1,
        )
        graph_summary = (community_scores.transpose(1, 2) @ community_summary).squeeze(1)
        fused = (graph_summary + node_summary) / 2.0
        logits = self.classifier(fused)
        probs = torch.softmax(logits, dim=-1)
        attn_info = {
            "node_scores": node_scores.squeeze(-1),
            "community_scores": community_scores.squeeze(-1),
        }
        return logits, probs, attn_info
