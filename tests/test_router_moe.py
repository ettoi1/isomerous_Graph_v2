"""Unit tests for the relation router MoE."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from models.relation_router_moe import RelationRouterMoE, RelationRouterMoEConfig


def _build_inputs(batch: int = 2, num_roi: int = 16, num_edges: int = 120) -> tuple:
    node_x = torch.randn(batch, num_roi, 48)
    edge_index = torch.randint(0, num_roi, (batch, 2, num_edges))
    edge_bank = torch.randn(batch, num_edges, 12)
    community_ctx = torch.softmax(torch.randn(batch, num_roi, 3), dim=-1)
    meta = [{"site_id": f"site_{i}"} for i in range(batch)]
    return node_x, edge_index, edge_bank, community_ctx, meta


def test_router_outputs_and_budget() -> None:
    config = RelationRouterMoEConfig(
        hidden_dim=48,
        num_relations=6,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        router_type="transformer",
        edge_budget_target=0.1,
    )
    router = RelationRouterMoE(config)
    router.set_epoch(0, 8)
    node_x, edge_index, edge_bank, community_ctx, meta = _build_inputs()
    typed_index, relation_id, keep_mask, stats = router(
        node_x, edge_index, edge_bank, community_ctx=community_ctx, meta_ctx=meta
    )
    assert typed_index.shape[0] == node_x.shape[0]
    assert relation_id.shape == keep_mask.shape
    assert keep_mask.dtype == torch.bool
    assert torch.all(keep_mask.sum(dim=1) >= 1)
    active_rel = relation_id[keep_mask]
    assert torch.unique(active_rel).numel() > 1
    density = stats["edge_density"].item()
    target = stats["edge_budget_target"].item()
    assert density <= target + 0.15
    stats["gating_entropy"].backward()


def test_router_hard_path_eval_mode() -> None:
    config = RelationRouterMoEConfig(
        hidden_dim=48,
        num_relations=4,
        d_model=48,
        num_heads=2,
        num_layers=1,
        dropout=0.1,
        router_type="mlp",
        edge_budget_target=0.2,
    )
    router = RelationRouterMoE(config)
    router.set_epoch(config.hard_after_epochs + 1, config.hard_after_epochs + 5)
    router.eval()
    node_x, edge_index, edge_bank, community_ctx, meta = _build_inputs()
    typed_index, relation_id, keep_mask, stats = router(
        node_x, edge_index, edge_bank, community_ctx=community_ctx, meta_ctx=meta
    )
    assert keep_mask.dtype == torch.bool
    assert torch.all(keep_mask.sum(dim=1) >= 1)
    assert int(stats["topk"].item()) == 1
