"""
dataset.py
Core dataset abstractions and transformation pipelines for butterfly image processing.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def build_label_map(csv_path: str) -> tuple[dict, dict]:
    """Construct bidirectional mapping between label strings and integer indices from metadata CSV."""
    df = pd.read_csv(csv_path)
    classes = sorted(df["label"].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

def get_splits(
    csv_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified dataset partitioning based on class labels.
    
    Args:
        csv_path: Path to the training metadata CSV.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        (train_df, val_df, test_df) stratified by label.
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

class ButterflyDataset(Dataset):
    """
    Map-style dataset for loading butterfly images from a single directory using a DataFrame slice.

    Attributes:
        df: DataFrame containing ['filename', 'label'] columns.
        img_dir: Root directory for image assets.
        label_to_idx: Label-to-index mapping.
        transform: Image transformation pipeline.
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

class AugmentedButterflyDataset(Dataset):
    """
    Dataset variant for handling mixed-source data with per-row directory paths.

    Expects DataFrame columns: ['filename', 'label', 'img_dir'].
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

def get_train_transform(image_size: int = 64) -> transforms.Compose:
    """Define standard augmentation pipeline for training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def get_eval_transform(image_size: int = 64) -> transforms.Compose:
    """Standard normalization pipeline for evaluation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def get_baseline_transform(image_size: int = 64) -> transforms.Compose:
    """Minimal transformation pipeline matching the original project baseline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Initialize DataLoader with optimal memory pinning for CUDA execution."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
