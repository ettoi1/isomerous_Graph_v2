"""Loss utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the main and auxiliary losses.

    Args:
        outputs: Dictionary produced by the model forward pass.
        batch: Mini-batch dictionary containing the ground-truth labels.
        config: Sub-dictionary from ``config['losses']``.

    Returns:
        Tuple ``(total_loss, losses_dict)``.
    """

    logits = outputs["logits"]
    labels = batch["label"]
    losses = {}
    losses["main"] = F.cross_entropy(logits, labels)

    router_weight = config.get("router_balance_weight", 0.0)
    if router_weight > 0 and "router_stats" in outputs:
        keep_ratio = outputs["router_stats"].get("keep_ratio")
        if keep_ratio is not None:
            losses["router_balance"] = router_weight * (keep_ratio - 0.5).abs()

    community_weight = config.get("community_compact_weight", 0.0)
    if community_weight > 0 and "community_assign" in outputs:
        assign = outputs["community_assign"]
        compact = (assign * assign).sum(dim=-1).mean()
        losses["community_compact"] = community_weight * (1.0 - compact)

    hyper_weight = config.get("hypergraph_consistency_weight", 0.0)
    if hyper_weight > 0 and "hyper_edges" in outputs:
        hyper = outputs["hyper_edges"]
        losses["hypergraph_consistency"] = hyper_weight * hyper.std()

    total = sum(losses.values())
    return total, losses
