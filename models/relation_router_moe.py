"""Switch-Transformer relation router with mixture-of-experts gating."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RelationRouterMoEConfig:
    """Configuration for :class:`RelationRouterMoE`."""

    hidden_dim: int
    num_relations: int
    d_model: int
    num_heads: int
    num_layers: int
    dropout: float
    router_type: str = "transformer"
    topk: int = 2
    hard_after_epochs: int = 5
    tau_start: float = 2.0
    tau_end: float = 0.5
    capacity_factor: float = 1.25
    capacity_final: float = 1.0
    edge_budget_target: float = 0.08
    edge_budget_warmup: Optional[float] = None
    use_relation_clusters: bool = True
    use_domain_prompt: bool = True
    residual_enabled: bool = True
    prune_enabled: bool = True
    domain_vocab: int = 64


class DomainPromptEncoder(torch.nn.Module):
    """Encode optional domain/site metadata into conditioning prompts."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)

    def forward(
        self,
        meta_ctx: Optional[Sequence[Dict]],
        num_edges: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not meta_ctx:
            return torch.zeros(batch_size, num_edges, self.embedding.embedding_dim, device=device)
        indices: List[int] = []
        for metadata in meta_ctx:
            token = (
                str(metadata.get("site_id"))
                or str(metadata.get("domain"))
                or str(metadata.get("subject_id"))
                or "default"
            )
            indices.append(hash(token) % self.embedding.num_embeddings)
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        if idx_tensor.shape[0] != batch_size:
            idx_tensor = idx_tensor[:batch_size]
            if idx_tensor.numel() < batch_size:
                pad = torch.zeros(batch_size - idx_tensor.shape[0], dtype=torch.long, device=device)
                idx_tensor = torch.cat([idx_tensor, pad], dim=0)
        embed = self.embedding(idx_tensor).unsqueeze(1).repeat(1, num_edges, 1)
        return embed


class RelationRouterMoE(torch.nn.Module):
    """Route candidate edges to relation experts before message passing.

    Args:
        config: :class:`RelationRouterMoEConfig` instance.

    Inputs:
        node_x: ``[B, R, D]`` embeddings from :mod:`timeseries_transformer`.
        edge_index: ``[B, 2, |E_c|]`` candidate edge indices.
        edge_bank: ``[B, |E_c|, M]`` multi-channel edge descriptors.
        community_ctx: optional ``[B, R, C]`` soft community hints.
        meta_ctx: optional metadata sequence (site id, age bucket, ...).

    Returns:
        typed_edge_index: ``[B, 2, |E_t|max]`` padded typed edges with mask.
        relation_id: ``[B, |E_t|max]`` expert ids aligned with the mask.
        keep_mask: ``[B, |E_t|max]`` boolean mask for valid edges.
        route_stats: dictionary with routing diagnostics.
    """

    def __init__(self, config: RelationRouterMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.base_experts = max(1, config.num_relations)
        self.residual_idx: Optional[int] = None
        self.prune_idx: Optional[int] = None
        total_experts = self.base_experts
        if config.residual_enabled:
            self.residual_idx = total_experts
            total_experts += 1
        if config.prune_enabled:
            self.prune_idx = total_experts
            total_experts += 1
        self.num_experts = total_experts

        self.edge_proj = torch.nn.LazyLinear(config.d_model)
        self.node_proj = torch.nn.Linear(config.hidden_dim, config.d_model)
        self.community_proj = torch.nn.LazyLinear(config.d_model)
        if config.use_domain_prompt:
            self.domain_prompt = DomainPromptEncoder(config.d_model, config.domain_vocab)
        else:
            self.domain_prompt = None
        self.token_proj = torch.nn.LazyLinear(config.d_model)
        self.dropout = torch.nn.Dropout(config.dropout)

        if config.router_type == "mlp":
            self.router = torch.nn.Sequential(
                torch.nn.Linear(config.d_model, config.d_model),
                torch.nn.GELU(),
                torch.nn.LayerNorm(config.d_model),
                torch.nn.Linear(config.d_model, config.d_model),
                torch.nn.GELU(),
            )
        else:
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                dim_feedforward=config.d_model * 4,
            )
            self.router = torch.nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.router_type = config.router_type
        self.head_prompts = torch.nn.Parameter(torch.randn(self.num_experts, config.d_model))
        self.cluster_adapter = (
            torch.nn.Sequential(
                torch.nn.LayerNorm(config.d_model),
                torch.nn.Linear(config.d_model, config.d_model),
                torch.nn.GELU(),
            )
            if config.use_relation_clusters
            else None
        )

        self.capacity_start = max(1.0, config.capacity_factor)
        self.capacity_final = max(1.0, config.capacity_final)
        self.edge_budget_warm = (
            config.edge_budget_warmup
            if config.edge_budget_warmup is not None
            else config.edge_budget_target * 1.5
        )
        self.current_tau = config.tau_start
        self.current_topk = config.topk
        self.current_capacity = self.capacity_start
        self.current_budget = self.edge_budget_warm
        self.hard_mode = False

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Update routing schedule according to the training epoch."""

        soft_epochs = max(1, self.config.hard_after_epochs)
        phase = min(1.0, epoch / soft_epochs)
        tail_phase = min(1.0, epoch / max(1, total_epochs - 1))
        self.current_tau = self.config.tau_start + (self.config.tau_end - self.config.tau_start) * phase
        self.current_topk = 1 if epoch >= self.config.hard_after_epochs else self.config.topk
        self.hard_mode = epoch >= self.config.hard_after_epochs
        self.current_capacity = self.capacity_start + (self.capacity_final - self.capacity_start) * phase
        budget_target = self.config.edge_budget_target
        warm = self.edge_budget_warm if self.edge_budget_warm > 0 else budget_target
        self.current_budget = warm + (budget_target - warm) * tail_phase

    def forward(
        self,
        node_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_bank: torch.Tensor,
        community_ctx: Optional[torch.Tensor] = None,
        meta_ctx: Optional[Sequence[Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        device = node_x.device
        B, R, _ = node_x.shape
        num_edges = edge_bank.shape[1]
        node_feat = self.node_proj(node_x)
        src_idx = edge_index[:, 0]
        dst_idx = edge_index[:, 1]
        src_feat = self._gather(node_feat, src_idx)
        dst_feat = self._gather(node_feat, dst_idx)
        edge_feat = self.edge_proj(edge_bank)
        if self.cluster_adapter is not None:
            edge_feat = edge_feat + self.cluster_adapter(edge_feat)
        if community_ctx is not None:
            comm_feat = self.community_proj(community_ctx)
            c_src = self._gather(comm_feat, src_idx)
            c_dst = self._gather(comm_feat, dst_idx)
            comm_token = torch.abs(c_src - c_dst)
        else:
            comm_token = torch.zeros_like(src_feat)
        if self.domain_prompt is not None:
            domain_prompt = self.domain_prompt(meta_ctx, num_edges, B, device)
        else:
            domain_prompt = torch.zeros(B, num_edges, src_feat.size(-1), device=device)
        token = torch.cat([edge_feat, src_feat, dst_feat, comm_token, domain_prompt], dim=-1)
        token = self.token_proj(token)
        token = self.dropout(token)
        routed = self.router(token)
        logits = torch.einsum("bed,md->bem", routed, self.head_prompts)
        route_probs, hard_assign = self._route_distribution(logits)
        relation_id = hard_assign
        edge_scores = torch.gather(route_probs, -1, relation_id.unsqueeze(-1)).squeeze(-1)

        keep_mask, overflow_ratio = self._apply_capacity(relation_id, edge_scores)
        keep_mask, fallback_edges = self._connectivity_guard(edge_index, keep_mask, edge_scores, relation_id, R)
        keep_mask = self._apply_edge_budget(keep_mask, edge_scores, R)
        typed_index, typed_rel, typed_mask, typed_scores = self._pack_typed_edges(
            edge_index, relation_id, keep_mask, edge_scores
        )

        stats = self._collect_stats(
            route_probs,
            typed_rel,
            typed_mask,
            typed_scores,
            logits,
            overflow_ratio,
            fallback_edges,
            R,
        )
        return typed_index, typed_rel, typed_mask, stats

    def _route_distribution(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and not self.hard_mode:
            probs = F.gumbel_softmax(logits, tau=self.current_tau, hard=False, dim=-1)
            if self.current_topk < self.num_experts:
                topk = min(self.num_experts, max(1, self.current_topk))
                topk_vals, topk_idx = torch.topk(probs, topk, dim=-1)
                mask = torch.zeros_like(probs)
                mask.scatter_(-1, topk_idx, 1.0)
                probs = probs * mask
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            tau = self.config.tau_end if not self.training else self.current_tau
            probs = torch.softmax(logits / max(tau, 1e-3), dim=-1)
        relation_id = probs.argmax(dim=-1)
        hard_probs = torch.zeros_like(probs).scatter_(-1, relation_id.unsqueeze(-1), 1.0)
        if self.training and not self.hard_mode:
            probs = probs + (hard_probs - probs).detach()
        else:
            probs = hard_probs
        return probs, relation_id

    def _apply_capacity(
        self,
        relation_id: torch.Tensor,
        edge_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, num_edges = relation_id.shape
        keep_mask = torch.ones_like(relation_id, dtype=torch.bool)
        overflow_total = 0.0
        for b in range(B):
            capacity = max(
                1,
                int(num_edges * self.current_capacity / max(1, self.base_experts)),
            )
            for expert in range(self.base_experts):
                expert_mask = relation_id[b] == expert
                count = int(expert_mask.sum())
                if count <= capacity:
                    continue
                overflow_total += (count - capacity) / max(1, num_edges)
                expert_scores = edge_scores[b][expert_mask]
                sorted_idx = torch.argsort(expert_scores, descending=True)
                drop_ids = expert_mask.nonzero(as_tuple=False).squeeze(-1)[sorted_idx[capacity:]]
                if self.residual_idx is not None:
                    relation_id[b, drop_ids] = self.residual_idx
                else:
                    keep_mask[b, drop_ids] = False
        overflow_ratio = relation_id.new_tensor(overflow_total / max(1, B))
        if self.prune_idx is not None:
            prune_mask = relation_id == self.prune_idx
            keep_mask = torch.where(prune_mask, torch.zeros_like(keep_mask), keep_mask)
        return keep_mask, overflow_ratio

    def _apply_edge_budget(
        self,
        keep_mask: torch.Tensor,
        edge_scores: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if self.training:
            target_density = self.current_budget
        else:
            target_density = self.config.edge_budget_target
        target_edges = int(target_density * (num_nodes * (num_nodes - 1)))
        if target_edges <= 0:
            return keep_mask
        for b in range(keep_mask.shape[0]):
            active = keep_mask[b]
            keep_count = int(active.sum())
            if keep_count <= target_edges:
                continue
            scores = edge_scores[b][active]
            sorted_idx = torch.argsort(scores, descending=True)
            keep_selection = active.nonzero(as_tuple=False).squeeze(-1)[sorted_idx[:target_edges]]
            new_mask = torch.zeros_like(active)
            new_mask[keep_selection] = True
            keep_mask[b] = new_mask
        return keep_mask

    def _connectivity_guard(
        self,
        edge_index: torch.Tensor,
        keep_mask: torch.Tensor,
        edge_scores: torch.Tensor,
        relation_id: torch.Tensor,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fallback_total = 0.0
        for b in range(edge_index.shape[0]):
            mask = keep_mask[b]
            if mask.sum() >= num_nodes - 1 and mask.any():
                continue
            src = edge_index[b, 0]
            dst = edge_index[b, 1]
            incidence = torch.zeros(num_nodes, device=edge_index.device)
            active_edges = mask.nonzero(as_tuple=False).squeeze(-1)
            for idx in active_edges:
                incidence[src[idx]] += 1
                incidence[dst[idx]] += 1
            lonely = (incidence == 0).nonzero(as_tuple=False).squeeze(-1)
            for node in lonely.tolist():
                candidates = (src == node) | (dst == node)
                if not bool(candidates.any()):
                    continue
                cand_scores = edge_scores[b][candidates]
                best_idx = torch.argmax(cand_scores)
                global_idx = candidates.nonzero(as_tuple=False).squeeze(-1)[best_idx]
                keep_mask[b, global_idx] = True
                fallback_total += 1.0
        return keep_mask, relation_id.new_tensor(fallback_total / max(1, edge_index.shape[0]))

    def _pack_typed_edges(
        self,
        edge_index: torch.Tensor,
        relation_id: torch.Tensor,
        keep_mask: torch.Tensor,
        edge_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_kept = keep_mask.sum(dim=1).tolist()
        max_edges = max(1, max(batch_kept))
        device = edge_index.device
        typed_index = edge_index.new_full((edge_index.shape[0], 2, max_edges), fill_value=-1)
        typed_rel = relation_id.new_full((relation_id.shape[0], max_edges), fill_value=-1)
        typed_scores = edge_scores.new_zeros((edge_scores.shape[0], max_edges))
        typed_mask = torch.zeros(relation_id.shape[0], max_edges, dtype=torch.bool, device=device)
        for b in range(edge_index.shape[0]):
            mask = keep_mask[b]
            if not bool(mask.any()):
                continue
            idx = edge_index[b][:, mask]
            rel = relation_id[b][mask]
            scores = edge_scores[b][mask]
            length = idx.shape[1]
            typed_index[b, :, :length] = idx
            typed_rel[b, :length] = rel
            typed_scores[b, :length] = scores
            typed_mask[b, :length] = True
        return typed_index, typed_rel, typed_mask, typed_scores

    def _collect_stats(
        self,
        route_probs: torch.Tensor,
        relation_id: torch.Tensor,
        typed_mask: torch.Tensor,
        edge_scores: torch.Tensor,
        logits: torch.Tensor,
        overflow_ratio: torch.Tensor,
        fallback_edges: torch.Tensor,
        num_nodes: int,
    ) -> Dict[str, torch.Tensor]:
        kept = typed_mask.float()
        usage = (
            torch.zeros(self.num_experts, device=route_probs.device)
            if kept.sum() == 0
            else torch.stack(
                [
                    (kept * (relation_id == expert).float()).sum()
                    for expert in range(self.num_experts)
                ]
            )
            / (kept.sum() + 1e-6)
        )
        special_budget = 0.0
        if self.residual_idx is not None:
            special_budget += 0.05
        if self.prune_idx is not None:
            special_budget += 0.05
        base_budget = max(1e-3, 1.0 - special_budget)
        target = torch.full_like(usage, base_budget / max(1, self.base_experts))
        if self.residual_idx is not None:
            target[self.residual_idx] = special_budget / 2 if special_budget else 0.05
        if self.prune_idx is not None:
            target[self.prune_idx] = special_budget / 2 if special_budget else 0.05
        target = target / target.sum()
        entropy = -(route_probs.clamp_min(1e-6).log() * route_probs).sum(dim=-1).mean()
        total_edges = kept.sum(dim=1)
        density = total_edges / max(1.0, num_nodes * (num_nodes - 1))
        keep_ratio = kept.mean()
        score_mean = (
            edge_scores[typed_mask].mean()
            if bool(typed_mask.any())
            else torch.tensor(0.0, device=route_probs.device)
        )
        stats: Dict[str, torch.Tensor] = {
            "expert_usage": usage,
            "target_usage": target,
            "keep_ratio": keep_ratio,
            "drop_ratio": 1.0 - keep_ratio,
            "overflow_ratio": overflow_ratio,
            "edge_density": density.mean(),
            "edge_budget_target": torch.tensor(
                self.config.edge_budget_target, device=route_probs.device
            ),
            "gating_entropy": entropy,
            "temperature": torch.tensor(
                self.current_tau if self.training else self.config.tau_end,
                device=route_probs.device,
            ),
            "topk": torch.tensor(self.current_topk if self.training else 1, device=route_probs.device),
            "capacity_factor": torch.tensor(self.current_capacity, device=route_probs.device),
            "time_smooth_penalty": self._time_smooth_penalty(relation_id, typed_mask, num_nodes),
            "fallback_edges": fallback_edges,
            "edge_scores_mean": score_mean,
            "comm_cross_penalty": torch.tensor(0.0, device=route_probs.device),
        }
        return stats

    @staticmethod
    def _gather(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, features.size(-1))
        return torch.gather(features, 1, gather_idx)

    @staticmethod
    def _time_smooth_penalty(
        relation_id: torch.Tensor, mask: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        penalties = []
        for b in range(relation_id.shape[0]):
            valid = mask[b]
            rel = relation_id[b][valid]
            if rel.numel() <= 1:
                penalties.append(relation_id.new_tensor(0.0))
                continue
            # Approximate smoothness by variance of neighbouring expert ids.
            norm = (rel.float() - rel.float().mean()).pow(2).mean()
            penalties.append(norm / max(1.0, num_nodes))
        stacked = torch.stack(penalties)
        return stacked.mean()
