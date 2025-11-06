"""Evaluation script."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from dataio.roi_dataset import RoiDataset, SyntheticConfig, collate_fn
from dataio.windowing import WindowConfig
from engines.builder import build_model
from engines.metrics import classification_metrics
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 0))

    synthetic_cfg = None
    if args.synthetic:
        synth = config.get("synthetic", {})
        synthetic_cfg = SyntheticConfig(
            num_subjects=synth.get("num_subjects", 8),
            num_roi=synth.get("num_roi", 32),
            num_timepoints=synth.get("num_timepoints", 120),
            num_frequency=synth.get("num_frequency", 3),
            num_edges=synth.get("num_edges", 96),
            num_edge_metrics=synth.get("num_edge_metrics", 6),
            num_classes=synth.get("num_classes", 2),
            noise_level=synth.get("noise_level", 0.1),
        )
    window_cfg = None
    ts_cfg = config["model"]["timeseries"]
    if ts_cfg.get("window_size"):
        window_cfg = WindowConfig(window_size=ts_cfg["window_size"], stride=ts_cfg["window_size"])
    dataset = RoiDataset(
        synthetic_config=synthetic_cfg,
        window_config=window_cfg,
        seed=config.get("seed", 0),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["trainer"].get("batch_size", 2),
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(config)
    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    metrics_accum: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch)
            metrics = classification_metrics(outputs["logits"], batch["label"])
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + float(v)
            count += 1
    metrics_mean = {k: v / max(count, 1) for k, v in metrics_accum.items()}
    print(json.dumps(metrics_mean, indent=2))


if __name__ == "__main__":
    main()
