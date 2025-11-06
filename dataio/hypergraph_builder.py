"""Hypergraph construction utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def build_hypergraph_from_assignments(
    community_assign: torch.Tensor,
    top_k: int = 3,
) -> Dict[str, torch.Tensor]:
    """Construct a toy hypergraph based on soft community assignments.

    Args:
        community_assign: Tensor of shape ``[B, R, C]`` with soft assignments.
        top_k: Number of nodes to include per hyperedge.

    Returns:
        Dictionary containing the incidence matrix ``H`` of shape ``[B, R, E]``
        and the hyperedge weights ``w`` of shape ``[B, E]``.
    """

    B, R, C = community_assign.shape
    num_edges = C * top_k
    H = torch.zeros(B, R, num_edges, device=community_assign.device)
    weights = torch.zeros(B, num_edges, device=community_assign.device)
    edge_idx = 0
    for c in range(C):
        scores = community_assign[..., c]
        top_nodes = scores.topk(top_k, dim=-1).indices
        for k in range(top_k):
            node_idx = top_nodes[:, k]
            batch_idx = torch.arange(B, device=community_assign.device)
            H[batch_idx, node_idx, edge_idx] = 1.0
            weights[:, edge_idx] = scores[batch_idx, node_idx]
            edge_idx += 1
    return {"incidence": H, "weights": weights}
