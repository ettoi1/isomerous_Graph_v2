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
    """Dataset returning ROI time-series and derived connectivity features.

    When ``root`` is provided, the dataset will attempt to read real ROI data
    from disk. ``.mat`` files are supported out-of-the-box via ``scipy`` or
    ``h5py``. By default it searches recursively for ``*.mat`` under ``root``.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        synthetic_config: Optional[SyntheticConfig] = None,
        window_config: Optional[WindowConfig] = None,
        seed: int = 0,
        # Real data options
        num_edges: Optional[int] = None,
        num_edge_metrics: Optional[int] = None,
        mat_var: Optional[str] = None,
        label_from: str = "filename",  # one of {filename, parent, matvar}
        label_var: Optional[str] = "label",
        labels_csv: Optional[Path] = None,
        id_column: str = "subject_id",
        label_column: str = "label",
        subject_id_from: str = "stem",  # one of {stem, parent, basename}
        recursive: bool = True,
        file_pattern: str = "*.mat",
        allow_transpose: bool = True,
        precomputed_edge_bank: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.synthetic_config = synthetic_config
        self.window_config = window_config
        self.generator = torch.Generator().manual_seed(seed)
        # Real data options
        self.real_num_edges = num_edges
        self.real_num_edge_metrics = num_edge_metrics
        self.mat_var = mat_var
        self.label_from = label_from
        self.label_var = label_var
        self.labels_csv = Path(labels_csv) if labels_csv is not None else None
        self.id_column = id_column
        self.label_column = label_column
        self.subject_id_from = subject_id_from
        self.recursive = recursive
        self.file_pattern = file_pattern
        self.allow_transpose = allow_transpose
        self.precomputed_edge_bank = precomputed_edge_bank

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
        # Discover files
        assert self.root is not None
        files = (
            list(self.root.rglob(self.file_pattern)) if self.recursive else list(self.root.glob(self.file_pattern))
        )
        if not files:
            raise FileNotFoundError(f"No files matching {self.file_pattern} found under {self.root}")

        # Optional imports for .mat
        try:
            from scipy.io import loadmat as _scipy_loadmat  # type: ignore
        except Exception:
            _scipy_loadmat = None  # type: ignore
        try:
            import h5py  # type: ignore
        except Exception:
            h5py = None  # type: ignore

        # Optional CSV labels
        csv_id_to_int: Optional[Dict[str, int]] = None
        int_to_raw: Optional[Dict[int, str]] = None
        if self.labels_csv is not None:
            import csv  # type: ignore
            raw_map: Dict[str, str] = {}
            with open(self.labels_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if self.id_column not in reader.fieldnames or self.label_column not in reader.fieldnames:
                    raise KeyError(
                        f"CSV {self.labels_csv} must contain columns '{self.id_column}' and '{self.label_column}'."
                    )
                for row in reader:
                    sid = str(row[self.id_column]).strip()
                    raw_label = str(row[self.label_column]).strip()
                    if sid:
                        raw_map[sid] = raw_label
            # Convert to int labels; if raw labels already ints, keep; else factorize
            csv_id_to_int = {}
            int_to_raw = {}
            uniques = sorted(set(raw_map.values()))
            all_int = True
            tmp_vals: Dict[str, int] = {}
            for r in uniques:
                try:
                    tmp_vals[r] = int(r)
                except Exception:
                    all_int = False
                    break
            if all_int:
                for sid, r in raw_map.items():
                    val = int(r)
                    csv_id_to_int[sid] = val
                    int_to_raw[val] = r
            else:
                label2idx = {r: i for i, r in enumerate(uniques)}
                for sid, r in raw_map.items():
                    idx = label2idx[r]
                    csv_id_to_int[sid] = idx
                    int_to_raw[idx] = r

        def _read_mat(path: Path) -> Dict:
            if _scipy_loadmat is not None:
                try:
                    data = _scipy_loadmat(path.as_posix())
                    # Filter out MATLAB metadata keys
                    return {k: v for k, v in data.items() if not k.startswith("__")}
                except Exception:
                    pass
            if h5py is not None:
                with h5py.File(path.as_posix(), "r") as f:  # type: ignore
                    def _to_numpy(obj):
                        return obj[()] if hasattr(obj, "__getitem__") and not isinstance(obj, (int, float)) else obj
                    return {k: _to_numpy(v) for k, v in f.items()}
            raise ImportError(
                "Reading .mat requires scipy or h5py. Please install one of them: pip install scipy or pip install h5py"
            )

        num_edges = int(self.real_num_edges) if self.real_num_edges is not None else 96
        num_metrics = int(self.real_num_edge_metrics) if self.real_num_edge_metrics is not None else 6

        for fpath in sorted(files):
            mat = _read_mat(fpath)
            # Pick ROI time-series variable
            roi_array = None
            if self.mat_var and self.mat_var in mat:
                roi_array = mat[self.mat_var]
            else:
                # Heuristic: pick the first 2D/3D numeric array
                for k, v in mat.items():
                    try:
                        import numpy as np  # local import
                        if isinstance(v, np.ndarray) and v.ndim in (2, 3) and np.issubdtype(v.dtype, np.number):
                            roi_array = v
                            break
                    except Exception:
                        pass
            if roi_array is None:
                raise KeyError(f"No suitable ROI array found in {fpath}. Set mat_var to the variable name.")

            import numpy as np  # type: ignore
            ts_np = np.array(roi_array)
            # Ensure shape [R, T] or [R, T, F]
            if ts_np.ndim == 2:
                R, T = ts_np.shape
                # If obvious transpose (more ROIs than timepoints is unusual), fix when allowed
                if self.allow_transpose and R > T:
                    ts_np = ts_np.T
            elif ts_np.ndim == 3:
                R, T, F = ts_np.shape
                if self.allow_transpose and R > T and T >= F:
                    ts_np = np.transpose(ts_np, (1, 0, 2))
            else:
                raise ValueError(f"Unsupported ROI array shape {ts_np.shape} in {fpath}")

            roi_ts = torch.from_numpy(ts_np).float()

            # Determine subject_id for CSV mapping or metadata
            if self.subject_id_from == "parent":
                subject_id = fpath.parent.name
            elif self.subject_id_from == "basename":
                subject_id = fpath.name
            else:
                subject_id = fpath.stem

            # Determine label
            label_tensor = torch.tensor(0, dtype=torch.long)
            raw_label: Optional[str] = None
            if csv_id_to_int is not None:
                if subject_id not in csv_id_to_int:
                    raise KeyError(
                        f"No label found for subject_id '{subject_id}' in CSV {self.labels_csv}."
                    )
                label_val = int(csv_id_to_int[subject_id])
                label_tensor = torch.tensor(label_val, dtype=torch.long)
                if int_to_raw is not None and label_val in int_to_raw:
                    raw_label = int_to_raw[label_val]
            elif self.label_from == "matvar" and self.label_var and self.label_var in mat:
                try:
                    label_val = int(np.array(mat[self.label_var]).squeeze().tolist())
                    label_tensor = torch.tensor(label_val, dtype=torch.long)
                except Exception:
                    pass
            elif self.label_from == "parent":
                try:
                    parent = fpath.parent.name
                    label_val = int(parent) if parent.isdigit() else 0
                    label_tensor = torch.tensor(label_val, dtype=torch.long)
                except Exception:
                    pass
            else:  # filename
                stem = fpath.stem
                digits = "".join(ch for ch in stem if ch.isdigit())
                if digits:
                    label_tensor = torch.tensor(int(digits) % 1000, dtype=torch.long)

            # Edge feature bank
            if self.precomputed_edge_bank and "edge_bank" in mat and "edge_index" in mat:
                edge_bank = torch.from_numpy(np.array(mat["edge_bank"]))
                edge_index = torch.from_numpy(np.array(mat["edge_index"]))
                if edge_index.dim() == 2 and edge_index.shape[0] == 2:
                    edge_index = edge_index  # [2, E]
                elif edge_index.dim() == 2 and edge_index.shape[1] == 2:
                    edge_index = edge_index.T
                else:
                    raise ValueError(f"edge_index shape must be [2, E] or [E, 2], got {edge_index.shape}")
            else:
                edge_bank, edge_index = build_edge_feature_bank(
                    roi_ts,
                    num_edges=num_edges,
                    num_metrics=num_metrics,
                    generator=self.generator,
                )

            sample = {
                "roi_ts": roi_ts,
                "edge_bank": edge_bank,
                "edge_index": edge_index,
                "label": label_tensor,
                "metadata": {
                    "subject_id": subject_id,
                    "source": fpath.as_posix(),
                    **({"label_raw": raw_label} if raw_label is not None else {}),
                },
            }
            self._cache.append(sample)

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
        base_signal = torch.sin(torch.linspace(0, 3.14, cfg.num_timepoints))
        roi_patterns = torch.stack(
            [torch.roll(base_signal, shifts=i) for i in range(cfg.num_roi)]
        )
        noise = torch.randn(roi_patterns.shape, generator=self.generator)
        roi_patterns = roi_patterns + cfg.noise_level * noise
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
