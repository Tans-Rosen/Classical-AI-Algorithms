"""Utilities: seed, device, checkpoint save/load, metric helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for PyTorch, NumPy, and Python RNG."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    """Return CUDA device if available, else CPU."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Save checkpoint dict to path. Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device) -> Any:
    """Load checkpoint from path. Map to device if needed."""
    return torch.load(path, map_location=device, weights_only=True)


def format_metrics(
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    epoch: int,
    num_epochs: int,
) -> str:
    """Format a one-line epoch summary."""
    return (
        f"Epoch {epoch}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )
