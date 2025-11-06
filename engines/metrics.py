"""Evaluation metrics for the project."""
from __future__ import annotations

from typing import Dict

import torch


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute accuracy-oriented metrics for binary/multi-class tasks."""

    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    accuracy = (preds == labels).float().mean()
    num_classes = probs.shape[-1]
    confusion = torch.zeros(num_classes, num_classes, device=logits.device)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1
    recall = confusion.diag() / (confusion.sum(dim=1) + 1e-6)
    precision = confusion.diag() / (confusion.sum(dim=0) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    balanced_acc = recall.mean()
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1": f1.mean(),
    }
