# Imagenette Image Classifier

## Project Overview

This project implements a convolutional neural network (CNN) for 10-class image classification on [Imagenette](https://github.com/fastai/imagenette). It uses PyTorch to define the model, run training and validation, evaluate on a held-out test set, and save best/last checkpoints. The code is organized into separate modules (`data`, `model`, `train`, `eval`, `utils`) and run via a CLI (`run.py`).

## What This Demonstrates

- **CNNs**: A small CNN with two conv blocks (Conv2d → ReLU → MaxPool2d) and an MLP head for 224×224 RGB images
- **Training loop**: Mini-batch gradient updates, Adam optimizer, cross-entropy loss
- **Validation**: 80/20 train–validation split from Imagenette train set; epoch-wise validation metrics
- **Evaluation**: Test-set accuracy and loss; comparison of best (by validation loss) vs last checkpoint
- **Data pipeline**: Imagenette via torchvision (auto-download or local `--data-dir`), resize to 224×224, DataLoaders
- **Checkpointing**: Save best and last model; `--resume` and `--eval-only` modes

## What Was Provided

The original setup provided:

- High-level structure (data loading pattern, model skeleton, training/eval signatures)
- Specifications for the CNN layout (conv layers, MLP, 10 classes), 80/20 train–val split, and training setup (Adam, cross-entropy, batch size)

## What I Contributed

I implemented and refactored:

### 1. **Model** (`model.py`)

- `ImagenetteCNN`: two conv blocks (16→32 channels) plus MLP head (hidden dim 256)
- Forward pass; computation of conv output size for correct FC input dimension

### 2. **Data** (`data.py`)

- Imagenette load via torchvision (train/val splits), resize and `ToTensor` transforms
- 80/20 random split of train data for validation; test set from Imagenette val split
- `build_datasets` and `build_dataloaders`; SSL/certifi workaround for macOS download issues

### 3. **Training** (`train.py`)

- `run_epoch`: one epoch over a DataLoader with optional gradient updates
- `train_loop`: full training with validation, best/last checkpointing, optional `--resume`
- Per-sample loss averaging (sum of batch-mean loss × batch size / total samples)

### 4. **Evaluation** (`eval.py`)

- `evaluate`: accuracy and optional loss over a DataLoader in eval mode
- `load_and_evaluate`: load checkpoint into model, then run evaluation

### 5. **Utils** (`utils.py`)

- `set_seed`, `get_device`, `save_checkpoint`, `load_checkpoint`, `format_metrics`

### 6. **CLI** (`run.py`)

- Argparse CLI: `--data-dir`, `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--seed`, `--device`, `--num-workers`, `--save-path`, `--resume`, `--eval-only`, `--checkpoint`
- Train vs eval-only modes; prints train/val metrics and final test results for best/last

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, torchvision, numpy, and certifi (for SSL on macOS). Use a PyTorch build that matches your platform (e.g. [pytorch.org](https://pytorch.org)).

### Basic Usage

```bash
python run.py [options]
```

### Arguments and Options

- **`--data-dir`** (default: `./data`): Root directory for Imagenette. Data downloads automatically on first run if missing.
- **`--epochs`** (default: `50`): Number of training epochs.
- **`--batch-size`** (default: `32`): Batch size.
- **`--lr`** (default: `1e-4`): Learning rate for Adam.
- **`--weight-decay`** (default: `0`): Adam weight decay.
- **`--seed`** (default: `42`): Random seed.
- **`--device`** (default: `cuda`): `cuda` or `cpu`.
- **`--num-workers`** (default: `0`): DataLoader workers.
- **`--save-path`** (default: `./checkpoints`): Directory for `best_model.pt` and `last_model.pt`.
- **`--resume`**: Path to checkpoint to resume training from.
- **`--eval-only`**: Skip training; run evaluation only.
- **`--checkpoint`**: Checkpoint path (required if `--eval-only`).

### Example Commands

```bash
# Train (downloads Imagenette into ./data if needed)
python run.py --data-dir ./data

# Shorter run
python run.py --data-dir ./data --epochs 10 --batch-size 64

# Eval only
python run.py --data-dir ./data --eval-only --checkpoint ./checkpoints/best_model.pt

# Resume training
python run.py --data-dir ./data --resume ./checkpoints/last_model.pt
```

## Output

The program prints:

- Device in use (or a warning if CUDA requested but unavailable)
- Train / validation / test dataset sizes
- Each epoch: train loss, train accuracy, validation loss, validation accuracy
- After training: test accuracy (and loss) for **best** and **last** checkpoints

Example:

```
Device: cpu
Train: 7575 | Val: 1894 | Test: 3925
Epoch 1/50 | Train Loss: 2.3021 | Train Acc: 10.12% | Val Loss: 2.2894 | Val Acc: 11.35%
...
Test [best]: Accuracy 56.74% | Loss 0.0412
Test [last]: Accuracy 55.82% | Loss 0.0421
```

## Data

- **Automatic**: Use `--data-dir ./data`. Imagenette (320px variant) is downloaded via torchvision on first run. Requires internet. If you see an SSL error, ensure `certifi` is installed; on macOS with python.org Python, running `Install Certificates.command` can also fix it.
- **Manual**: Download [Imagenette](https://github.com/fastai/imagenette) (e.g. 320px tarball), extract under `--data-dir`, and run as above.

## Project Structure

```
convolutional-neural-networks/
├── README.md           # This file
├── requirements.txt    # torch, torchvision, numpy, certifi
├── run.py              # CLI entrypoint
├── data.py             # Imagenette load, transforms, DataLoaders
├── model.py            # ImagenetteCNN
├── train.py            # Train loop, validation, checkpointing
├── eval.py             # Test evaluation, load-and-eval
├── utils.py            # Seed, device, checkpoints, metrics
├── checkpoints/        # best_model.pt, last_model.pt (created at runtime)
├── data/               # Imagenette data (optional; use --data-dir)
└── ai_hw_6.ipynb       # Original notebook
```

## Notes

- The CNN expects 224×224 RGB input. Images are resized via transforms; no extra normalization.
- Best checkpoint is chosen by lowest validation loss.
- Both best and last are evaluated on the test set after training.

## Academic Integrity Notice

**This code is shared for portfolio purposes only.**

This project originated as a programming assignment and has been refactored into a standalone codebase. Do not submit this or any derivative for academic credit. Use it as inspiration, not as a solution. If you are an instructor and believe this should not be public, please contact me and I will remove it.

---

**Language**: Python 3  
**Libraries**: PyTorch, torchvision, NumPy, certifi
