"""Model builder assembling the end-to-end pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from dataio.hypergraph_builder import build_hypergraph_from_assignments
from models import (
    CommunityAttention,
    GraphTransformer,
    HypergraphTransformer,
    RelationRouterMoE,
    TimeSeriesTransformer,
    HierarchicalReadout,
)
from models.community_attention import CommunityAttentionConfig
from models.graph_transformer import GraphTransformerConfig
from models.hypergraph_transformer import HypergraphTransformerConfig
from models.readout_hierarchical import HierarchicalReadoutConfig
from models.relation_router_moe import RelationRouterMoEConfig
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
            router_cfg = RelationRouterMoEConfig(
                hidden_dim=hidden_dim,
                num_relations=model_cfg["router"].get("num_relations", 4),
                d_model=model_cfg["router"].get("d_model", hidden_dim),
                num_heads=model_cfg["router"].get("num_heads", num_heads),
                num_layers=model_cfg["router"].get("num_layers", 2),
                dropout=model_cfg["router"].get("dropout", dropout),
                router_type=model_cfg["router"].get("type", "transformer"),
                topk=model_cfg["router"].get("topk", 2),
                hard_after_epochs=model_cfg["router"].get("hard_after_epochs", 5),
                tau_start=model_cfg["router"].get("tau_start", 2.0),
                tau_end=model_cfg["router"].get("tau_end", 0.5),
                capacity_factor=model_cfg["router"].get("capacity_factor", 1.25),
                capacity_final=model_cfg["router"].get("capacity_final", 1.0),
                edge_budget_target=model_cfg["router"].get("edge_budget_target", 0.1),
                edge_budget_warmup=model_cfg["router"].get("edge_budget_warmup"),
                use_relation_clusters=model_cfg["router"].get("use_relation_clusters", True),
                use_domain_prompt=model_cfg["router"].get("use_domain_prompt", True),
                residual_enabled=model_cfg["router"].get("residual_enabled", True),
                prune_enabled=model_cfg["router"].get("prune_enabled", True),
                domain_vocab=model_cfg["router"].get("domain_vocab", 64),
            )
            self.router = RelationRouterMoE(router_cfg)

        self.graph_tf = None
        if not ablation.get("disable_graph_tf", False) and model_cfg["graph_transformer"][
            "enabled"
        ]:
            graph_cfg = GraphTransformerConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=model_cfg["graph_transformer"].get("num_layers", 1),
                dropout=dropout,
                max_relations=max(
                    32,
                    (self.router.num_experts if self.router is not None else 0) + 4,
                ),
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
            typed_edge_index, relation_id, keep_mask, router_stats = self.router(
                node_embeddings,
                edge_index,
                edge_bank,
                community_ctx=None,
                meta_ctx=batch.get("metadata"),
            )
        else:
            B, _, num_edges = edge_bank.shape
            typed_edge_index = edge_index
            relation_id = torch.zeros(B, num_edges, dtype=torch.long, device=roi_ts.device)
            keep_mask = torch.ones(B, num_edges, dtype=torch.bool, device=roi_ts.device)
            router_stats = {
                "keep_ratio": torch.tensor(1.0, device=roi_ts.device),
                "edge_density": torch.tensor(1.0, device=roi_ts.device),
            }
        outputs["typed_edge_index"] = typed_edge_index
        outputs["typed_relation_id"] = relation_id
        outputs["edge_mask"] = keep_mask
        outputs["router_stats"] = router_stats

        if self.graph_tf is not None:
            node_embeddings = self.graph_tf(node_embeddings, typed_edge_index, relation_id, keep_mask)
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

        if "router_stats" in outputs:
            router_stats = outputs["router_stats"]
            router_stats.update(
                self._community_routing_stats(
                    outputs.get("typed_edge_index"),
                    outputs.get("typed_relation_id"),
                    outputs.get("edge_mask"),
                    assign_soft,
                )
            )

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

    def _community_routing_stats(
        self,
        edge_index: Optional[torch.Tensor],
        relation_id: Optional[torch.Tensor],
        edge_mask: Optional[torch.Tensor],
        assign_soft: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if edge_index is None or relation_id is None or edge_mask is None:
            return {}
        if assign_soft is None:
            return {}
        hard_assign = assign_soft.argmax(dim=-1)
        device = assign_soft.device
        intra = []
        inter = []
        for b in range(edge_index.shape[0]):
            mask = edge_mask[b]
            if not bool(mask.any()):
                intra.append(torch.tensor(0.0, device=device))
                inter.append(torch.tensor(0.0, device=device))
                continue
            idx = edge_index[b][:, mask]
            rel = relation_id[b][mask]
            src = idx[0]
            dst = idx[1]
            same = (hard_assign[b, src] == hard_assign[b, dst]).float()
            intra.append(same.mean())
            inter.append((1.0 - same).mean())
        intra_ratio = torch.stack(intra).mean()
        inter_ratio = torch.stack(inter).mean()
        comm_penalty = torch.clamp(inter_ratio - 0.2, min=0.0)
        return {
            "intra_community_ratio": intra_ratio,
            "inter_community_ratio": inter_ratio,
            "comm_cross_penalty": comm_penalty,
        }

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        if self.router is not None and hasattr(self.router, "set_epoch"):
            self.router.set_epoch(epoch, total_epochs)

def build_model(config: Dict[str, Any]) -> DynamicGraphModel:
    """Factory returning the assembled pipeline."""

    return DynamicGraphModel(config)
