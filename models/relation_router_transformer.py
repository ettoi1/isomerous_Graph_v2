"""Mixture-of-experts style relation routing for graph construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from layers.attention_utils import softmax_with_temperature


@dataclass
class RelationRouterConfig:
    """Configuration for :class:`RelationRouterTransformer`."""

    hidden_dim: int
    num_relations: int
    temperature: float
    keep_threshold: float


class RelationRouterTransformer(torch.nn.Module):
    """Select relation types for candidate edges prior to message passing."""

    def __init__(self, config: RelationRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.LazyLinear(config.hidden_dim),
            torch.nn.GELU(),
        )
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, config.num_relations),
        )
        self.keep_head = torch.nn.Linear(config.hidden_dim * 2, 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Route edges to relation experts.

        Args:
            node_embeddings: Tensor ``[B, R, D]`` representing node features.
            edge_bank: Tensor ``[B, |E|, M]`` storing candidate edge descriptors.

        Returns:
            ``relation_id``: ``[B, |E|]`` hard assignment per edge.
            ``keep_mask``: boolean tensor selecting active edges.
            ``stats``: dictionary with auxiliary routing statistics.
        """

        B, num_edges, _ = edge_bank.shape
        pooled_nodes = node_embeddings.mean(dim=1)
        pooled = pooled_nodes.unsqueeze(1).expand(-1, num_edges, -1)
        edge_encoded = self.edge_encoder(edge_bank)
        fused = torch.cat([pooled, edge_encoded], dim=-1)
        fused = torch.nn.functional.layer_norm(fused, fused.shape[-1:])
        relation_logits = self.edge_mlp(fused)
        relation_probs = softmax_with_temperature(
            relation_logits, self.config.temperature
        )
        relation_id = relation_probs.argmax(dim=-1)
        keep_scores = torch.sigmoid(self.keep_head(fused)).squeeze(-1)
        keep_mask = keep_scores > self.config.keep_threshold
        stats = {
            "prob_mean": relation_probs.mean(dim=(0, 1)),
            "keep_ratio": keep_mask.float().mean(),
        }
        return relation_id, keep_mask, stats
