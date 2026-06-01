"""
metrics.py
Quantitative evaluation framework for generative models.

Supported Metrics:
    - Fréchet Inception Distance (FID): Distributional distance between feature representations.
    - Inception Score (IS): Assessment of generative diversity and image quality.
    - Structural Similarity Index (SSIM): Pairwise structural comparison (primarily for cVAE).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim_fn

_inception_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _get_inception(device: torch.device) -> torch.nn.Module:
    """Initialize InceptionV3 with a truncated FC layer for feature extraction."""
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)

def _preprocess_images(images) -> torch.Tensor:
    """Normalize input (Tensor or PIL list) to [0, 1] range and (N, 3, H, W) format."""
    if isinstance(images, (list, tuple)):
        to_tensor = transforms.ToTensor()
        images = torch.stack([to_tensor(img.convert("RGB")) for img in images])
    images = images.float()
    if images.min() < -0.01:
        images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)

@torch.no_grad()
def _extract_features(
    images: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 2048-dimensional InceptionV3 latent features."""
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    for (batch,) in loader:
        batch = batch.to(device)
        batch = _inception_transform(batch)
        feat = model(batch)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

def compute_fid(
    real_images,
    generated_images,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> float:
    """
    Compute Fréchet Inception Distance between real and synthetic image distributions.
    
    Evaluates global distribution matching rather than per-class metrics to ensure
    statistical robustness given limited samples per class.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_images = _preprocess_images(real_images)
    gen_images = _preprocess_images(generated_images)

    inception = _get_inception(device)

    real_feats = _extract_features(real_images, inception, device, batch_size)
    gen_feats = _extract_features(gen_images, inception, device, batch_size)

    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))
    return fid

@torch.no_grad()
def compute_inception_score(
    generated_images,
    device: torch.device | None = None,
    batch_size: int = 64,
    splits: int = 10,
) -> tuple[float, float]:
    """
    Calculate Inception Score based on KL-divergence of marginal and conditional distributions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_images = _preprocess_images(generated_images)

    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.eval()
    inception = inception.to(device)

    preds = []
    dataset = TensorDataset(gen_images)
    loader = DataLoader(dataset, batch_size=batch_size)

    _is_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            batch = _is_transform(batch)
            logits = inception(batch)
            if hasattr(logits, "logits"):
                logits = logits.logits
            p = F.softmax(logits, dim=1).cpu().numpy()
            preds.append(p)

    preds = np.concatenate(preds, axis=0)
    n = preds.shape[0]
    split_size = n // splits

    scores = []
    for k in range(splits):
        part = preds[k * split_size : (k + 1) * split_size]
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))

    return float(np.mean(scores)), float(np.std(scores))

def compute_ssim(
    real_images,
    generated_images,
    channel_axis: int = 0,
) -> float:
    """
    Calculate Mean Structural Similarity Index between paired images.
    
    Useful for reconstruction tasks (e.g., cVAE) where generated images have 
    a direct ground-truth counterpart.
    """
    real_images = _preprocess_images(real_images)
    gen_images = _preprocess_images(generated_images)

    n = min(len(real_images), len(gen_images))
    scores = []
    for i in range(n):
        r = real_images[i].numpy()
        g = gen_images[i].numpy()
        score = ssim_fn(
            r, g,
            data_range=1.0,
            channel_axis=channel_axis,
        )
        scores.append(score)

    return float(np.mean(scores))

def classification_report_dict(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    idx_to_label: dict | None = None,
) -> dict:
    """
    Generate comprehensive classification metrics including macro-F1 and per-class accuracy.
    """
    from collections import defaultdict

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float((y_true == y_pred).mean())

    classes = np.unique(y_true)
    f1s = []
    per_class_acc = {}
    for c in classes:
        mask = y_true == c
        tp = ((y_pred == c) & mask).sum()
        fp = ((y_pred == c) & ~mask).sum()
        fn = (mask & (y_pred != c)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
        name = idx_to_label.get(int(c), str(c)) if idx_to_label else str(c)
        per_class_acc[name] = float(tp / mask.sum()) if mask.sum() > 0 else 0.0

    macro_f1 = float(np.mean(f1s))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_acc": per_class_acc,
    }
