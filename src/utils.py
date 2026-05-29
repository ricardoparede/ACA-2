"""
utils.py
Shared helper utilities: checkpoints, visualization, tensor helpers.
"""

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: torch.nn.Module, path: str, **extra) -> None:
    """Save model state dict and any extra metadata (e.g. epoch, loss)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    payload.update(extra)
    torch.save(payload, path)
    print(f"[checkpoint] saved → {path}")


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device = None):
    """Load state dict into model inplace. Returns the full payload dict."""
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    print(f"[checkpoint] loaded ← {path}")
    return payload


# ---------------------------------------------------------------------------
# Tensor normalization helpers
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Undo Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) applied during data loading.
    Maps [-1, 1] → [0, 1].
    Works on batched (N,C,H,W) or single (C,H,W) tensors.
    """
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (C,H,W) float tensor in [0,1] to a PIL Image.
    Call denormalize() first if tensor is in [-1,1].
    """
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_grid(
    images: torch.Tensor,
    labels: list | None = None,
    idx_to_label: dict | None = None,
    n: int = 16,
    nrow: int = 8,
    title: str = "",
    denorm: bool = True,
) -> None:
    """
    Display a grid of images.

    Args:
        images      : (N, C, H, W) tensor, optionally normalized to [-1,1]
        labels      : integer class indices for each image
        idx_to_label: mapping from index to human-readable string
        n           : how many images to show
        nrow        : images per row
        title       : plot title
        denorm      : if True, apply denormalize() before display
    """
    imgs = images[:n].cpu()
    if denorm:
        imgs = denormalize(imgs)
    grid = make_grid(imgs, nrow=nrow, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(min(nrow * 2, 16), math.ceil(n / nrow) * 2))
    plt.imshow(np_grid)
    plt.axis("off")
    if title:
        plt.title(title)
    if labels is not None and idx_to_label is not None:
        label_str = " | ".join(
            idx_to_label.get(int(l), str(int(l))) for l in labels[:n]
        )
        plt.xlabel(label_str[:120], fontsize=7)
    plt.tight_layout()
    plt.show()


def plot_losses(
    train_losses: list,
    val_losses: list,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    title: str = "Training curve",
) -> None:
    """Plot train vs. validation loss curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a validation metric and signals when training should stop.

    Usage:
        es = EarlyStopping(patience=10, min_delta=1e-4)
        for epoch in ...:
            val_loss = ...
            if es(val_loss):
                break
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        improved = (
            metric < self.best - self.min_delta
            if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
