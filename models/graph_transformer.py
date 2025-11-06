"""Graph Transformer operating on routed typed edges."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from layers.attention_utils import MultiheadAttentionWithRope, sparse_attention_mask


@dataclass
class GraphTransformerConfig:
    """Configuration for :class:`GraphTransformer`."""

    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    use_rope: bool = False
    max_relations: int = 32


class GraphTransformerLayer(torch.nn.Module):
    """Single block combining attention with feedforward updates."""

    def __init__(self, config: GraphTransformerConfig) -> None:
        super().__init__()
        self.attn = MultiheadAttentionWithRope(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_rope=config.use_rope,
        )
        self.attn_norm = torch.nn.LayerNorm(config.hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )
        self.ffn_norm = torch.nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.attn_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x


class GraphTransformer(torch.nn.Module):
    """Graph Transformer module using relation-aware attention masks."""

    def __init__(self, config: GraphTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(
            [GraphTransformerLayer(config) for _ in range(config.num_layers)]
        )
        self.relation_bias = torch.nn.Embedding(max(config.max_relations, 32), 1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        relation_id: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply relation-aware message passing.

        Args:
            node_embeddings: ``[B, R, D]`` node features.
            edge_index: ``[B, 2, |E|]`` edge endpoints.
            relation_id: ``[B, |E|]`` relation identifiers.
            keep_mask: ``[B, |E|]`` boolean mask of active edges.

        Returns:
            Updated node embeddings of shape ``[B, R, D]``.
        """

        B, R, _ = node_embeddings.shape
        outputs = []
        for b in range(B):
            x = node_embeddings[b]
            mask = self._build_mask(R, edge_index[b], relation_id[b], keep_mask[b])
            for layer in self.layers:
                x = layer(x, mask)
            outputs.append(x)
        return torch.stack(outputs)

    def _build_mask(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        relation_id: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        if keep_mask is not None:
            edge_index = edge_index[:, keep_mask]
            relation_id = relation_id[keep_mask]
        if edge_index.numel() == 0:
            return sparse_attention_mask(torch.empty(2, 0, dtype=torch.long, device=edge_index.device), num_nodes)
        base_mask = sparse_attention_mask(edge_index, num_nodes)
        if relation_id.numel() == 0:
            return base_mask
        bias = self.relation_bias(relation_id).squeeze(-1)
        for idx in range(edge_index.shape[1]):
            src = edge_index[0, idx]
            dst = edge_index[1, idx]
            base_mask[src, dst] += bias[idx]
        return base_mask
