"""Logging helpers wrapping TensorBoard when available."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - TensorBoard is optional
    SummaryWriter = None


class MetricLogger:
    """Simple logger used across training and evaluation."""

    def __init__(self, log_dir: Path, use_tensorboard: bool = True) -> None:
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard and SummaryWriter is not None
        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(log_dir))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
