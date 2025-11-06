"""Shape tests for the main modules."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

torch = pytest.importorskip("torch")

from dataio.roi_dataset import RoiDataset, SyntheticConfig, collate_fn
from engines.builder import build_model


def load_demo_config() -> dict:
    path = Path("configs/demo_synthetic.yaml")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_model_output_shapes() -> None:
    config = load_demo_config()
    synth = config["synthetic"]
    dataset = RoiDataset(
        synthetic_config=SyntheticConfig(
            num_subjects=2,
            num_roi=synth["num_roi"],
            num_timepoints=synth["num_timepoints"],
            num_frequency=synth["num_frequency"],
            num_edges=synth["num_edges"],
            num_edge_metrics=synth["num_edge_metrics"],
            num_classes=synth["num_classes"],
        )
    )
    batch = collate_fn([dataset[0], dataset[1]])
    model = build_model(config)
    outputs = model(batch)
    assert outputs["logits"].shape[0] == 2
    assert outputs["community_assign"].shape[0] == 2
    assert outputs["node_embeddings_final"].shape[1] == synth["num_roi"]
