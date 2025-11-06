"""Edge feature bank generation."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_edge_feature_bank(
    roi_ts: torch.Tensor,
    num_edges: int,
    num_metrics: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a simple synthetic edge feature bank.

    Returns both the feature matrix ``[|E|, M]`` and the ``edge_index`` tensor of
    shape ``[2, |E|]`` describing the ROI pairs sampled from the time-series.
    """

    R = roi_ts.shape[0]
    device = roi_ts.device
    generator = generator or torch.Generator(device=device)
    edges = torch.randint(0, R, size=(num_edges, 2), generator=generator)
    ts = roi_ts.reshape(R, -1)
    features = []
    for idx in range(num_edges):
        i, j = edges[idx]
        a, b = ts[i], ts[j]
        pearson = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6)
        covariance = torch.mean((a - a.mean()) * (b - b.mean()))
        phase = torch.cos(a - b).mean()
        metrics = torch.tensor([pearson, covariance, phase], dtype=torch.float32)
        if num_metrics > 3:
            noise = torch.randn(num_metrics - 3, generator=generator) * 0.01
            metrics = torch.cat([metrics, noise])
        elif num_metrics < 3:
            metrics = metrics[:num_metrics]
        features.append(metrics)
    feature_bank = torch.stack(features)
    edge_index = edges.t().contiguous()
    return feature_bank, edge_index
