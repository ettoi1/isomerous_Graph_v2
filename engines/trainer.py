"""Training and evaluation loops."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from engines.losses import compute_losses
from engines.metrics import classification_metrics
from utils.checkpoint import save_checkpoint
from utils.logger import MetricLogger


class Trainer:
    """Manage the training, evaluation and logging lifecycle."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: Dict,
        log_dir: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = MetricLogger(log_dir, use_tensorboard=config["logging"].get("use_tensorboard", False))
        self.best_metric = 0.0
        self.log_dir = log_dir
        self.grad_clip = config["trainer"].get("grad_clip", 0.0)
        self.amp = config["trainer"].get("amp", False) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        epochs = self.config["trainer"]["epochs"]
        patience = self.config["trainer"].get("patience", epochs)
        patience_counter = 0
        for epoch in range(epochs):
            if hasattr(self.model, "on_epoch_start"):
                self.model.on_epoch_start(epoch, epochs)
            train_metrics = self._run_epoch(train_loader, epoch, training=True)
            print(f"Epoch {epoch} train metrics: {train_metrics}")
            if self.scheduler is not None:
                self.scheduler.step()
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, epoch, training=False)
                print(f"Epoch {epoch} validation metrics: {val_metrics}")
                main_metric = val_metrics.get("accuracy", 0.0)
                if main_metric > self.best_metric:
                    self.best_metric = main_metric
                    patience_counter = 0
                    save_checkpoint(
                        self.log_dir / "best.pt",
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "metric": main_metric,
                        },
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

    def _run_epoch(self, loader: DataLoader, epoch: int, training: bool) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        device = self.device
        losses_accum: Dict[str, float] = {}
        metrics_accum: Dict[str, float] = {}
        count = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.set_grad_enabled(training):
                if self.amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                        loss, loss_dict = compute_losses(
                            outputs,
                            batch,
                            self.config.get("losses", {}),
                            self.config.get("loss_weights", {}),
                        )
                else:
                    outputs = self.model(batch)
                    loss, loss_dict = compute_losses(
                        outputs,
                        batch,
                        self.config.get("losses", {}),
                        self.config.get("loss_weights", {}),
                    )
            if training:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            losses_accum = _accumulate(losses_accum, loss_dict)
            metric_dict = classification_metrics(outputs["logits"], batch["label"])
            metrics_accum = _accumulate(metrics_accum, metric_dict)
            count += 1
            if training and step % self.config["trainer"].get("log_interval", 10) == 0:
                for name, value in loss_dict.items():
                    self.logger.log_scalar(f"train/{name}", value.item(), epoch * len(loader) + step)
                self._log_router_stats(outputs.get("router_stats"), epoch * len(loader) + step)
        losses_mean = {k: float(v / max(count, 1)) for k, v in losses_accum.items()}
        metrics_mean = {k: float(v / max(count, 1)) for k, v in metrics_accum.items()}
        return {**losses_mean, **metrics_mean}

    def _log_router_stats(self, stats: Optional[Dict[str, torch.Tensor]], global_step: int) -> None:
        if not stats:
            return
        for key, value in stats.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.detach()
            if tensor.ndim == 0:
                self.logger.log_scalar(f"router/{key}", float(tensor), global_step)
            elif tensor.ndim == 1:
                for idx, item in enumerate(tensor):
                    self.logger.log_scalar(f"router/{key}_{idx}", float(item), global_step)

def _accumulate(acc: Dict[str, float], values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    for key, value in values.items():
        acc[key] = acc.get(key, 0.0) + float(value.detach().cpu())
    return acc
