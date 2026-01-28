"""Training loop with validation, checkpointing, and metrics."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils import format_metrics, save_checkpoint


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """Run one epoch. If train, perform gradient updates. Returns (avg loss, accuracy %)."""
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        if train and optimizer is not None:
            optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        if train and optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * batch_size
        _, pred = logits.max(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += batch_size

    n = max(total_samples, 1)
    return total_loss / n, 100.0 * total_correct / n


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    save_path: str | Path,
    resume: str | Path | None = None,
) -> dict[str, list[float]]:
    """Full training loop. Saves best (by val loss) and last checkpoints."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    best_path = save_path / "best_model.pt"
    last_path = save_path / "last_model.pt"

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.to(device)
    info = defaultdict(list)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        info["train_losses"].append(train_loss)
        info["train_accuracies"].append(train_acc)
        info["val_losses"].append(val_loss)
        info["val_accuracies"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({"state_dict": model.state_dict(), "epoch": epoch}, best_path)
        save_checkpoint({"state_dict": model.state_dict(), "epoch": epoch}, last_path)

        print(format_metrics(train_loss, train_acc, val_loss, val_acc, epoch, num_epochs))

    return dict(info)
