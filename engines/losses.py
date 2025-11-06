"""Loss utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    base_config: Dict[str, float],
    loss_weights: Dict[str, float],
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

    config = base_config or {}
    weights = loss_weights or {}

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

    lambda_balance = weights.get("lambda_balance", 0.0)
    lambda_entropy = weights.get("lambda_entropy", 0.0)
    lambda_edge = weights.get("lambda_edge_budget", 0.0)
    lambda_time = weights.get("lambda_time_smooth", 0.0)
    lambda_comm = weights.get("lambda_comm", 0.0)
    router_stats = outputs.get("router_stats", {}) if lambda_balance or lambda_entropy or lambda_edge or lambda_time or lambda_comm else {}
    if lambda_balance > 0 and router_stats:
        usage = router_stats.get("expert_usage")
        target = router_stats.get("target_usage")
        if usage is not None and target is not None:
            losses["router_balance_reg"] = lambda_balance * (usage - target).abs().mean()
    if lambda_entropy > 0 and router_stats:
        entropy = router_stats.get("gating_entropy")
        if entropy is not None:
            losses["router_entropy_reg"] = lambda_entropy * entropy
    if lambda_edge > 0 and router_stats:
        density = router_stats.get("edge_density")
        budget = router_stats.get("edge_budget_target")
        if density is not None and budget is not None:
            losses["router_edge_budget"] = lambda_edge * (density - budget).abs()
    if lambda_time > 0 and router_stats:
        smooth = router_stats.get("time_smooth_penalty")
        if smooth is not None:
            losses["router_time_smooth"] = lambda_time * smooth
    if lambda_comm > 0 and router_stats:
        comm = router_stats.get("comm_cross_penalty")
        if comm is not None:
            losses["router_comm"] = lambda_comm * comm

    total = sum(losses.values())
    return total, losses
