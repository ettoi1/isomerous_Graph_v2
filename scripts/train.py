"""Training entry point for dynamic fMRI connectivity models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Ensure the project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.roi_dataset import RoiDataset, SyntheticConfig, collate_fn
from dataio.windowing import WindowConfig
from engines.builder import build_model
from engines.trainer import Trainer
from utils.profiling import model_parameter_count
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the dynamic connectivity model")
    parser.add_argument("--config", type=Path, required=True, help="Configuration YAML file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data generator")
    parser.add_argument("--disable", nargs="*", default=[], help="Disable modules by name")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    for module in args.disable:
        key = f"disable_{module}"
        if key in config.get("ablation", {}):
            config["ablation"][key] = True

    set_seed(config.get("seed", 0))

    synthetic_cfg = None
    data_cfg = config.get("data", {})
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
        root=(Path(data_cfg["root"]) if (not args.synthetic and data_cfg.get("root")) else None),
        synthetic_config=synthetic_cfg,
        window_config=window_cfg,
        seed=config.get("seed", 0),
        num_edges=data_cfg.get("num_edges"),
        num_edge_metrics=data_cfg.get("num_edge_metrics"),
        mat_var=data_cfg.get("mat_var"),
        label_from=data_cfg.get("label_from", "filename"),
        label_var=data_cfg.get("label_var", "label"),
        labels_csv=(Path(data_cfg["labels_csv"]) if data_cfg.get("labels_csv") else None),
        id_column=data_cfg.get("id_column", "subject_id"),
        label_column=data_cfg.get("label_column", "label"),
        subject_id_from=data_cfg.get("subject_id_from", "stem"),
        recursive=data_cfg.get("recursive", True),
        file_pattern=data_cfg.get("file_pattern", "*.mat"),
        allow_transpose=data_cfg.get("allow_transpose", True),
        precomputed_edge_bank=data_cfg.get("precomputed_edge_bank", False),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["trainer"].get("batch_size", 2),
        shuffle=True,
        num_workers=config["trainer"].get("num_workers", 0),
        collate_fn=collate_fn,
    )

    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model parameter count: {model_parameter_count(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["trainer"].get("lr", 1e-3),
        weight_decay=config["trainer"].get("weight_decay", 0.0),
    )
    scheduler = None
    scheduler_cfg = config["trainer"].get("scheduler", {})
    if scheduler_cfg.get("name") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["trainer"].get("epochs", 1),
            eta_min=scheduler_cfg.get("min_lr", 1e-5),
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        log_dir=output_dir,
    )
    trainer.train(train_loader)
    save_visualizations(model, dataset, output_dir, device)


def save_visualizations(
    model: torch.nn.Module,
    dataset: RoiDataset,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Produce a minimal visualization showcasing attention distributions."""

    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        print("matplotlib not available, skipping visualization.")
        return
    model.eval()
    sample = collate_fn([dataset[0]])
    sample = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample.items()}
    with torch.no_grad():
        outputs = model(sample)
    community = outputs["community_assign"].squeeze(0).cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(community, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Community assignment heatmap")
    plt.tight_layout()
    fig_path = output_dir / "community_heatmap.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved visualization to {fig_path}")


if __name__ == "__main__":
    main()
