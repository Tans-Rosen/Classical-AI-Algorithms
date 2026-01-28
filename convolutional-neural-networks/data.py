"""Dataset setup: Imagenette download, transforms, and DataLoaders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

_HAS_IMAGENETTE = hasattr(datasets, "Imagenette")

IMAGENETTE_SIZE = "320px"
INPUT_SIZE = 224
NUM_CLASSES = 10
VALID_FRAC = 0.2


def _default_transform(train: bool) -> transforms.Compose:
    t = [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(t)


def _get_imagenette(
    root: str | Path, train: bool, transform: Optional[transforms.Compose] = None
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return datasets.Imagenette(
        root=str(root),
        split="train" if train else "val",
        size=IMAGENETTE_SIZE,
        download=True,
        transform=transform,
    )


def _use_certifi_for_ssl() -> None:
    """Use certifi's CA bundle for SSL. Helps avoid CERTIFICATE_VERIFY_FAILED on macOS."""
    try:
        import ssl
        import certifi
        path = certifi.where()
        os.environ["SSL_CERT_FILE"] = path
        os.environ["REQUESTS_CA_BUNDLE"] = path
        # Patch default HTTPS context so urllib uses certifi's bundle
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=path)
    except ImportError:
        pass


def build_datasets(data_dir: str | Path, seed: int = 42):
    """Build train, validation, and test datasets from Imagenette."""
    if not _HAS_IMAGENETTE:
        raise RuntimeError(
            "Imagenette requires torchvision with Imagenette support (e.g. torchvision >= 0.14). "
            "Install with: pip install torch torchvision"
        )

    _use_certifi_for_ssl()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_transform = _default_transform(train=True)
    eval_transform = _default_transform(train=False)

    try:
        full_train = _get_imagenette(data_dir, train=True, transform=train_transform)
        test_ds = _get_imagenette(data_dir, train=False, transform=eval_transform)
    except Exception as e:
        _print_data_instructions(data_dir)
        raise RuntimeError(f"Could not load Imagenette: {e}") from e

    n = len(full_train)
    n_val = int(n * VALID_FRAC)
    n_train = n - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    return train_ds, val_ds, test_ds


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
):
    """Build train, validation, and test DataLoaders."""
    train_ds, val_ds, test_ds = build_datasets(data_dir, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def _print_data_instructions(data_dir: Path) -> None:
    print("\n--- Data not found ---")
    print(f"Data directory: {data_dir.resolve()}")
    print("Options:")
    print("  1. Run from a machine with internet; Imagenette will download via torchvision.")
    print("  2. Ensure torchvision supports Imagenette: pip install --upgrade torch torchvision")
    print("  3. Use --data-dir /path/to/parent so 'imagenette2-320' (or similar) can be created there.")
    print("  4. SSL error? Install certifi and use bundled certs: pip install certifi")
    print("     On macOS with python.org Python: run 'Install Certificates.command' from the Python app folder.")
    print()
