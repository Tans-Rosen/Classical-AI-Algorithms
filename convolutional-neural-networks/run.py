#!/usr/bin/env python3
"""
CLI entrypoint: train or evaluate the Imagenette CNN.

Examples:
  python run.py --data-dir ./data
  python run.py --data-dir ./data --epochs 20 --batch-size 64
  python run.py --data-dir ./data --eval-only --checkpoint ./checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args():
    p = argparse.ArgumentParser(
        description="Train or evaluate Imagenette CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=str, default="./data",
                   help="Root directory for Imagenette data.")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", type=str, choices=("cpu", "cuda"), default="cuda",
                   help="Device to use.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers.")
    p.add_argument("--save-path", type=str, default="./checkpoints",
                   help="Directory for best/last checkpoints.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from.")
    p.add_argument("--eval-only", action="store_true", help="Skip training; run evaluation only.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path for --eval-only (required if --eval-only).")
    return p.parse_args()


def main():
    args = _parse_args()

    import torch
    from data import build_dataloaders
    from eval import load_and_evaluate
    from model import ImagenetteCNN
    from train import train_loop
    from utils import get_device, set_seed

    set_seed(args.seed)
    device = get_device(args.device)
    if args.device == "cuda" and str(device) == "cpu":
        print("Warning: CUDA requested but not available; using CPU.")
    print(f"Device: {device}")

    try:
        train_loader, val_loader, test_loader = build_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    model = ImagenetteCNN(in_channels=3, num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()

    if args.eval_only:
        if not args.checkpoint:
            print("Error: --eval-only requires --checkpoint.")
            sys.exit(1)
        acc, loss = load_and_evaluate(
            model, args.checkpoint, test_loader, device, criterion
        )
        print(f"Test Accuracy: {acc:.2f}%")
        if loss is not None:
            print(f"Test Loss: {loss:.4f}")
        return

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loop(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=args.epochs, save_path=args.save_path, resume=args.resume,
    )

    best_path = Path(args.save_path) / "best_model.pt"
    last_path = Path(args.save_path) / "last_model.pt"
    for name, path in [("best", best_path), ("last", last_path)]:
        if not path.exists():
            continue
        acc, loss = load_and_evaluate(model, str(path), test_loader, device, criterion)
        extra = f" | Loss {loss:.4f}" if loss is not None else ""
        print(f"Test [{name}]: Accuracy {acc:.2f}%{extra}")


if __name__ == "__main__":
    main()
