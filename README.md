# Dynamic fMRI ROI Connectivity Pipeline

This repository implements an end-to-end pipeline for dynamic functional
connectivity modelling on region-of-interest (ROI) fMRI signals. The design
follows a modular stack combining Transformer-based components, mixture of
experts routing and hierarchical readout. All code is implemented in Python
with PyTorch and PyTorch Geometric friendly abstractions (graph processing is
implemented with standard tensors to keep the demo lightweight).

## Pipeline Overview

The project assembles six major components:

1. **ROI Time-Series Transformer** (`models/timeseries_transformer.py`)
   converts raw ROI windows into node embeddings. It supports rotary
   positional encoding and optional frequency-aware representations.
2. **Relation Router Transformer** (`models/relation_router_transformer.py`)
   assigns relation types to candidate edges before message passing using a
   Switch-Transformer style gating mechanism.
3. **Graph Transformer** (`models/graph_transformer.py`) performs relation
   aware message passing with sparse attention masks informed by the router.
4. **Community Attention** (`models/community_attention.py`) implements a
   Slot-Attention block that discovers soft communities and provides
   regularisation statistics.
5. **Hypergraph Transformer** (`models/hypergraph_transformer.py`) models
   higher-order interactions via node ↔ hyperedge attention.
6. **Hierarchical Readout** (`models/readout_hierarchical.py`) aggregates node
   and community representations into final predictions with temperature
   controlled attention.

Each component can be individually disabled at runtime via the configuration or
`--disable` command line flag to support ablation studies.

## Repository Structure

```
project_root/
├── configs/                 # YAML experiment configurations
├── data/                    # Placeholder for real datasets
├── dataio/                  # Dataset, windowing and hypergraph utilities
├── engines/                 # Builder, trainer, losses and metrics
├── layers/                  # Attention helpers and positional encodings
├── models/                  # Transformer, community, hypergraph and readout
├── scripts/                 # CLI entry points (train/evaluate/export)
├── tests/                   # PyTest smoke tests
└── utils/                   # Logging, seeding, checkpointing, profiling
```

## Synthetic Demo

A minimal synthetic configuration is provided in `configs/demo_synthetic.yaml`.
It simulates a dataset with:

- Batch size 2, ROI count 32, sequence length 120 and three frequency bands.
- Ninety-six candidate edges with six correlation metrics.
- Two target classes and a three slot community module.

Run the demo training loop for three epochs using:

```bash
python scripts/train.py --config configs/demo_synthetic.yaml --synthetic
```

During execution the script prints training metrics and saves a community
assignment heatmap to `outputs/demo/community_heatmap.png`.

## Evaluation and ONNX Export

After training, evaluate a checkpoint on the synthetic dataset:

```bash
python scripts/evaluate.py --config configs/demo_synthetic.yaml \
    --checkpoint outputs/demo/best.pt --synthetic
```

Export the model to ONNX (requires a saved checkpoint):

```bash
python scripts/export_onnx.py --config configs/demo_synthetic.yaml \
    --checkpoint outputs/demo/best.pt --output outputs/demo/model.onnx --synthetic
```

## Real Data Integration

To use real fMRI ROI datasets, subclass or extend
`dataio/roi_dataset.RoiDataset`:

1. Place time-series arrays in `data/` grouped by subject or session.
2. Override `_load_from_disk` to read ROI matrices and associated labels.
3. Optionally pre-compute edge feature banks and store them alongside the time
   series. The expected shapes are documented in each module docstring.

The remainder of the pipeline remains unchanged. Ensure the YAML configuration
reflects the number of ROIs, candidate edges and metrics produced by your
pre-processing pipeline.

### Using `.mat` ROI files (built-in)

`RoiDataset` now supports reading `.mat` files directly (via `scipy` or `h5py`).
Add a `data` section to your YAML config and run without `--synthetic`:

```
data:
  root: data/my_subjects           # folder containing .mat files (searched recursively)
  num_edges: 96                    # number of candidate edges per subject
  num_edge_metrics: 6              # edge metrics dimension
  mat_var: roi_ts                  # variable name inside .mat; omit to auto-detect
  label_from: filename             # one of: filename | parent | matvar
  label_var: label                 # when label_from=matvar, variable name
  recursive: true                  # search subfolders
  file_pattern: "*.mat"            # pattern to match
  allow_transpose: true            # transpose if shape looks like [T, R]
  precomputed_edge_bank: false     # set true if .mat contains edge_bank & edge_index
```

Requirements: install either `scipy` or `h5py` (e.g., `pip install scipy`).

Naming assumptions: if `label_from=filename`, an integer in the filename will be
used as label when present; otherwise 0.

#### Labels from CSV

If labels are stored in a CSV file, add these fields under `data`:

```
data:
  labels_csv: data/labels.csv   # CSV path
  id_column: subject_id         # column with subject IDs matching file stems (or parents)
  label_column: label           # column with class labels (int or string)
  subject_id_from: stem         # how to derive the id from files: stem | parent | basename
```

- If `label_column` contains integers, they are used directly.
- If strings, they are factorized deterministically to integers (sorted unique order).
- When `labels_csv` is set, it overrides `label_from`/`label_var` heuristics.

See `configs/demo_mat_csv.yaml` for a complete example.

## Testing

Run the provided smoke tests with:

```bash
pytest tests/
```

The tests validate tensor shapes and confirm that forward passes and losses are
finite for the synthetic demo configuration.

## Reproducibility and Logging

- `utils/seed.py` provides a single entry point for deterministic runs.
- TensorBoard logging is enabled when the dependency is installed; disable it by
  setting `logging.use_tensorboard` to `false` in the YAML config.
- The training script saves the active configuration alongside outputs for easy
  experiment replay.

## License

This repository is intended for research and educational use. Adapt and extend
it for domain-specific studies or more realistic datasets.
