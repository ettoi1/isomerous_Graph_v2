"""Export the trained model to ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from dataio.roi_dataset import RoiDataset, SyntheticConfig, collate_fn
from dataio.windowing import WindowConfig
from engines.builder import build_model
from utils.checkpoint import load_checkpoint
from utils.seed import set_seed


class _OnnxWrapper(torch.nn.Module):
    """Wrap the main model to provide tuple outputs for ONNX export."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, roi_ts: torch.Tensor, edge_bank: torch.Tensor, edge_index: torch.Tensor):
        batch = {
            "roi_ts": roi_ts,
            "edge_bank": edge_bank,
            "edge_index": edge_index,
        }
        outputs = self.model(batch)
        ordered_keys = sorted(outputs.keys())
        return tuple(outputs[k] for k in ordered_keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ONNX graph")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
            num_subjects=synth.get("num_subjects", 2),
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
    sample = collate_fn([dataset[0]])

    model = build_model(config)
    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state["model"])
    model.eval()

    roi_ts = sample["roi_ts"].float()
    edge_bank = sample["edge_bank"].float()
    edge_index = sample["edge_index"].float()
    wrapper = _OnnxWrapper(model)
    torch.onnx.export(
        wrapper,
        (roi_ts, edge_bank, edge_index),
        args.output,
        input_names=["roi_ts", "edge_bank", "edge_index"],
        output_names=sorted(model(sample).keys()),
        opset_version=17,
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
