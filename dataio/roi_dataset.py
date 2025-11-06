"""Dataset utilities for fMRI ROI dynamic connectivity experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .edge_feature_bank import build_edge_feature_bank
from .windowing import WindowConfig, build_time_windows


@dataclass
class SyntheticConfig:
    """Configuration used to generate synthetic ROI time-series."""

    num_subjects: int
    num_roi: int
    num_timepoints: int
    num_frequency: int
    num_edges: int
    num_edge_metrics: int
    num_classes: int
    noise_level: float = 0.1


class RoiDataset(Dataset):
    """Dataset returning ROI time-series and derived connectivity features."""

    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        synthetic_config: Optional[SyntheticConfig] = None,
        window_config: Optional[WindowConfig] = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.synthetic_config = synthetic_config
        self.window_config = window_config
        self.generator = torch.Generator().manual_seed(seed)
        if synthetic_config is None and self.root is None:
            raise ValueError("Either root or synthetic_config must be provided.")
        if self.root is not None and not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root} does not exist.")
        self._cache: List[Dict[str, torch.Tensor]] = []
        if synthetic_config is not None:
            self._generate_synthetic()
        else:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        raise NotImplementedError(
            "Real dataset loading is not implemented. Provide ``synthetic_config`` "
            "or override ``_load_from_disk`` in a subclass."
        )

    def _generate_synthetic(self) -> None:
        cfg = self.synthetic_config
        assert cfg is not None
        for idx in range(cfg.num_subjects):
            roi_ts = self._generate_timeseries(cfg)
            edge_bank, edge_index = build_edge_feature_bank(
                roi_ts,
                num_edges=cfg.num_edges,
                num_metrics=cfg.num_edge_metrics,
                generator=self.generator,
            )
            label = torch.tensor(idx % cfg.num_classes, dtype=torch.long)
            sample = {
                "roi_ts": roi_ts,
                "edge_bank": edge_bank,
                "edge_index": edge_index,
                "label": label,
                "metadata": {
                    "subject_id": f"synthetic_{idx}",
                },
            }
            self._cache.append(sample)

    def _generate_timeseries(self, cfg: SyntheticConfig) -> torch.Tensor:
        base_signal = torch.sin(
            torch.linspace(0, 3.14, cfg.num_timepoints, generator=self.generator)
        )
        roi_patterns = torch.stack([
            torch.roll(base_signal, shifts=i) for i in range(cfg.num_roi)
        ])
        roi_patterns = roi_patterns + cfg.noise_level * torch.randn_like(roi_patterns)
        if cfg.num_frequency > 1:
            freq_bands = []
            for f in range(cfg.num_frequency):
                shift = f * 2
                band = torch.roll(roi_patterns, shifts=shift, dims=1)
                freq_bands.append(band)
            roi_ts = torch.stack(freq_bands, dim=-1)
        else:
            roi_ts = roi_patterns.unsqueeze(-1)
        return roi_ts.float()

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._cache[idx]
        roi_ts = item["roi_ts"]
        if self.transform is not None:
            roi_ts = self.transform(roi_ts)
        sample = {
            "roi_ts": roi_ts,
            "edge_bank": item["edge_bank"],
            "edge_index": item["edge_index"],
            "label": item["label"],
            "metadata": item["metadata"],
        }
        if self.window_config is not None:
            windows = build_time_windows(roi_ts, self.window_config)
            sample["windows"] = windows
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    roi_ts = torch.stack([b["roi_ts"] for b in batch])
    edge_bank = torch.stack([b["edge_bank"] for b in batch])
    edge_index = torch.stack([b["edge_index"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    metadata = [b["metadata"] for b in batch]
    sample = {
        "roi_ts": roi_ts,
        "edge_bank": edge_bank,
        "edge_index": edge_index,
        "label": labels,
        "metadata": metadata,
    }
    if "windows" in batch[0]:
        windows = torch.stack([b["windows"] for b in batch])
        sample["windows"] = windows
    return sample
