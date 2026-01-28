"""Evaluation loop: test accuracy and loss."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import load_checkpoint


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
) -> tuple[float, float | None]:
    """Run model over loader in eval mode. Returns (accuracy %, average loss or None)."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    use_loss = criterion is not None

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            logits = model(inputs)
            if use_loss and criterion is not None:
                total_loss += criterion(logits, labels).item() * batch_size
            _, pred = logits.max(dim=1)
            total_correct += (pred == labels).sum().item()
            total_samples += batch_size

    n = max(total_samples, 1)
    acc = 100.0 * total_correct / n
    avg_loss = total_loss / n if use_loss else None
    return acc, avg_loss


def load_and_evaluate(
    model: nn.Module,
    checkpoint_path: str,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[float, float | None]:
    """Load checkpoint into model, then run evaluate. Returns (accuracy, loss)."""
    state = load_checkpoint(checkpoint_path, device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    return evaluate(model, loader, criterion, device)
