"""
dataset.py
Butterfly dataset utilities shared across all notebooks.

CSV format (train.csv):
    filename   label
    Image_0.jpg  CLOUDED SULPHUR
    ...
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def build_label_map(csv_path: str) -> tuple[dict, dict]:
    """Return (label_to_idx, idx_to_label) built from a CSV file."""
    df = pd.read_csv(csv_path)
    classes = sorted(df["label"].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def get_splits(
    csv_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by label.  Returns (train_df, val_df, test_df).
    test_ratio is inferred as 1 - train_ratio - val_ratio.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)

    train_rows, val_rows, test_rows = [], [], []
    for _, group in df.groupby("label"):
        idx = rng.permutation(len(group))
        n_train = int(len(group) * train_ratio)
        n_val = int(len(group) * val_ratio)
        rows = group.iloc[idx]
        train_rows.append(rows.iloc[:n_train])
        val_rows.append(rows.iloc[n_train : n_train + n_val])
        test_rows.append(rows.iloc[n_train + n_val :])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# ButterflyDataset  (mirrors the one in TP2-students.ipynb)
# ---------------------------------------------------------------------------

class ButterflyDataset(Dataset):
    """
    Loads butterfly images from a flat directory using a DataFrame slice.

    Args:
        df          : DataFrame with columns ['filename', 'label']
        img_dir     : Directory that contains the image files
        label_to_idx: Mapping from species string to integer class index
        transform   : torchvision transform to apply to each image
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        label_to_idx: dict,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# AugmentedButterflyDataset
# ---------------------------------------------------------------------------

class AugmentedButterflyDataset(Dataset):
    """
    Identical interface to ButterflyDataset but accepts a combined DataFrame
    that may reference images from multiple directories.

    The DataFrame must have columns ['filename', 'label', 'img_dir'] where
    'img_dir' is the absolute directory for that row's image.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(row["img_dir"], row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Standard transforms (reusable across notebooks)
# ---------------------------------------------------------------------------

def get_train_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_eval_transform(image_size: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_baseline_transform(image_size: int = 64) -> transforms.Compose:
    """Matches the exact transform in TP2-students.ipynb (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
