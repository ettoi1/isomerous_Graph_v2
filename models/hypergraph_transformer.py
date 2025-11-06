"""Hypergraph Transformer encoder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class HypergraphTransformerConfig:
    """Configuration for :class:`HypergraphTransformer`."""

    hidden_dim: int
    num_heads: int
    num_iters: int
    dropout: float


class HypergraphTransformer(torch.nn.Module):
    """Hypergraph encoder exchanging information between nodes and hyperedges."""

    def __init__(self, config: HypergraphTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.node_to_edge = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.edge_to_node = torch.nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.edge_proj = torch.nn.Linear(config.hidden_dim, config.hidden_dim)
        self.node_norm = torch.nn.LayerNorm(config.hidden_dim)
        self.edge_norm = torch.nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        hypergraph: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run hypergraph message passing.

        Args:
            node_embeddings: Tensor ``[B, R, D]``.
            hypergraph: Dictionary containing ``incidence`` ``[B, R, E]`` and
                optional ``weights`` ``[B, E]``.

        Returns:
            Updated node embeddings, hyperedge representations and statistics.
        """

        incidence = hypergraph["incidence"]
        weights = hypergraph.get("weights")
        B, R, E = incidence.shape
        edges = []
        for b in range(B):
            node_mask = incidence[b].transpose(0, 1)
            pooled = torch.matmul(node_mask, node_embeddings[b]) / (
                node_mask.sum(dim=-1, keepdim=True) + 1e-6
            )
            edges.append(pooled)
        edge_feats = torch.stack(edges)
        if weights is not None:
            edge_feats = edge_feats * weights.unsqueeze(-1)
        edge_feats = self.edge_proj(edge_feats)
        node_out, _ = self.node_to_edge(node_embeddings, edge_feats, edge_feats)
        node_embeddings = self.node_norm(node_embeddings + node_out)
        edge_out, _ = self.edge_to_node(edge_feats, node_embeddings, node_embeddings)
        edge_feats = self.edge_norm(edge_feats + edge_out)
        stats = {"edge_mean": edge_feats.mean()}
        return node_embeddings, edge_feats, stats
