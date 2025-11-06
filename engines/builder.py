"""Model builder assembling the end-to-end pipeline."""
from __future__ import annotations

from typing import Any, Dict

import torch

from dataio.hypergraph_builder import build_hypergraph_from_assignments
from models import (
    CommunityAttention,
    GraphTransformer,
    HypergraphTransformer,
    RelationRouterTransformer,
    TimeSeriesTransformer,
    HierarchicalReadout,
)
from models.community_attention import CommunityAttentionConfig
from models.graph_transformer import GraphTransformerConfig
from models.hypergraph_transformer import HypergraphTransformerConfig
from models.readout_hierarchical import HierarchicalReadoutConfig
from models.relation_router_transformer import RelationRouterConfig
from models.timeseries_transformer import TimeSeriesTransformerConfig


class DynamicGraphModel(torch.nn.Module):
    """Full pipeline combining temporal, graph, hypergraph and readout modules."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = model_cfg["hidden_dim"]
        num_heads = model_cfg["num_heads"]
        dropout = model_cfg["dropout"]
        ablation = config.get("ablation", {})

        self.timeseries = None
        if not ablation.get("disable_timeseries", False) and model_cfg["timeseries"]["enabled"]:
            ts_cfg = TimeSeriesTransformerConfig(
                input_channels=model_cfg["timeseries"].get("input_channels", 1),
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_frequency=model_cfg["timeseries"].get("use_frequency", False),
                use_rope=model_cfg["timeseries"].get("rope", False),
                window_size=model_cfg["timeseries"].get("window_size"),
            )
            self.timeseries = TimeSeriesTransformer(ts_cfg)

        self.router = None
        if not ablation.get("disable_router", False) and model_cfg["router"]["enabled"]:
            router_cfg = RelationRouterConfig(
                hidden_dim=hidden_dim,
                num_relations=model_cfg["router"].get("num_relations", 4),
                temperature=model_cfg["router"].get("temperature", 1.0),
                keep_threshold=model_cfg["router"].get("keep_threshold", 0.5),
            )
            self.router = RelationRouterTransformer(router_cfg)

        self.graph_tf = None
        if not ablation.get("disable_graph_tf", False) and model_cfg["graph_transformer"][
            "enabled"
        ]:
            graph_cfg = GraphTransformerConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=model_cfg["graph_transformer"].get("num_layers", 1),
                dropout=dropout,
            )
            self.graph_tf = GraphTransformer(graph_cfg)

        self.community = None
        if not ablation.get("disable_community", False) and model_cfg["community"]["enabled"]:
            community_cfg = CommunityAttentionConfig(
                hidden_dim=hidden_dim,
                num_slots=model_cfg["community"].get("num_slots", 4),
                iterations=model_cfg["community"].get("iterations", 3),
                dropout=dropout,
            )
            self.community = CommunityAttention(community_cfg)

        self.hypergraph = None
        if not ablation.get("disable_hyper_tf", False) and model_cfg["hypergraph"]["enabled"]:
            hyper_cfg = HypergraphTransformerConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_iters=model_cfg["hypergraph"].get("num_iters", 1),
                dropout=dropout,
            )
            self.hypergraph = HypergraphTransformer(hyper_cfg)

        self.readout = None
        if not ablation.get("disable_hier_readout", False) and model_cfg["readout"]["enabled"]:
            readout_cfg = HierarchicalReadoutConfig(
                hidden_dim=hidden_dim,
                num_classes=model_cfg["readout"].get("num_classes", 2),
                temperature=model_cfg["readout"].get("temperature", 1.0),
            )
            self.readout = HierarchicalReadout(readout_cfg)

        self.hidden_dim = hidden_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        roi_ts = batch["roi_ts"].float()
        edge_bank = batch["edge_bank"].float()
        edge_index = batch["edge_index"].long()

        if self.timeseries is not None:
            node_embeddings, time_states = self.timeseries(roi_ts)
            outputs["time_states"] = time_states
        else:
            node_embeddings = roi_ts.mean(dim=-1) if roi_ts.dim() == 4 else roi_ts
            node_embeddings = node_embeddings.mean(dim=-1, keepdim=True)
            time_states = None
        outputs["node_embeddings_initial"] = node_embeddings

        if self.router is not None:
            relation_id, keep_mask, router_stats = self.router(node_embeddings, edge_bank)
        else:
            B, _, _ = edge_bank.shape
            relation_id = torch.zeros(B, edge_bank.shape[1], dtype=torch.long, device=roi_ts.device)
            keep_mask = torch.ones_like(relation_id, dtype=torch.bool)
            router_stats = {"keep_ratio": torch.tensor(1.0)}
        outputs["router_stats"] = router_stats

        if self.graph_tf is not None:
            node_embeddings = self.graph_tf(node_embeddings, edge_index, relation_id, keep_mask)
        outputs["node_embeddings_graph"] = node_embeddings

        hypergraph_data = None
        if self.community is not None:
            assign_soft, centers, community_stats = self.community(node_embeddings, time_states)
        else:
            B, R, _ = node_embeddings.shape
            assign_soft = torch.full(
                (B, R, 1),
                1.0,
                device=node_embeddings.device,
            )
            centers = node_embeddings.mean(dim=1, keepdim=True)
            community_stats = {}
        outputs["community_assign"] = assign_soft
        outputs["community_centers"] = centers
        outputs["community_stats"] = community_stats

        if self.hypergraph is not None:
            hypergraph_data = build_hypergraph_from_assignments(assign_soft)
            node_embeddings, hyper_edges, hyper_stats = self.hypergraph(
                node_embeddings, hypergraph_data
            )
            outputs["hyper_edges"] = hyper_edges
            outputs["hyper_stats"] = hyper_stats
        outputs["node_embeddings_final"] = node_embeddings

        if self.readout is not None:
            logits, probs, attn_info = self.readout(node_embeddings, assign_soft, centers)
            outputs["logits"] = logits
            outputs["probs"] = probs
            outputs["attn_info"] = attn_info
        else:
            pooled = node_embeddings.mean(dim=1)
            outputs["logits"] = pooled
            outputs["probs"] = pooled

        return outputs


def build_model(config: Dict[str, Any]) -> DynamicGraphModel:
    """Factory returning the assembled pipeline."""

    return DynamicGraphModel(config)
