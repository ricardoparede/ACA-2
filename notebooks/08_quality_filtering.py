# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic Data Quality Filtering (Oracle)
# ### Automated refinement of generated datasets using a pretrained ResNet18 Oracle.
#
# Implementation details:
# - Phase A: Fine-tune a ResNet18 Oracle on the real training distribution.
# - Phase B: Score all generated samples across models (cVAE, GAN, etc.).
# - Filtering: Preserving samples where (Oracle Prediction == Intended Class) AND (Softmax Confidence >= Threshold).
#

# %%
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import pandas as pd
from tqdm import tqdm

# %%
# ── Project Path Setup ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(".."))

from src.dataset import (
    ButterflyDataset, AugmentedButterflyDataset, get_splits,
    build_label_map, get_train_transform, get_eval_transform, make_dataloader
)
from src.utils import save_checkpoint, load_checkpoint

# %%
# ── Device Configuration ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# ── Hyperparameters and Paths ──────────────────────────────────────────────
BASE_DIR       = os.path.abspath("..")
CSV_PATH       = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
IMG_DIR        = os.path.join(BASE_DIR, "aca-butterflies", "train")
ORACLE_PATH    = os.path.join(BASE_DIR, "saved_models", "oracle_resnet18.pth")

CONF_THRESHOLD = 0.6
BATCH_SIZE     = 64
IMAGE_SIZE     = 64
NUM_EPOCHS     = 15
LR             = 1e-4

GENERATIVE_MODELS = ["cvae", "gan", "wgan_gp", "biggan", "diffusion"]

# %%
def get_oracle_model(num_classes):
    """Initialize ResNet18 with modified classification head for target classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

# %%
def train_oracle():
    """Train the Oracle model on the real dataset to establish a reference distribution classifier."""
    label_to_idx, idx_to_label = build_label_map(CSV_PATH)
    num_classes = len(label_to_idx)
    
    if os.path.exists(ORACLE_PATH):
        print(f"Oracle already exists at {ORACLE_PATH}. Loading...")
        model = get_oracle_model(num_classes)
        load_checkpoint(ORACLE_PATH, model, device)
        return model

    train_df, val_df, _ = get_splits(CSV_PATH, train_ratio=0.8, val_ratio=0.2, seed=42)
    
    train_set = ButterflyDataset(train_df, IMG_DIR, label_to_idx, transform=get_train_transform(IMAGE_SIZE))
    val_set   = ButterflyDataset(val_df,   IMG_DIR, label_to_idx, transform=get_eval_transform(IMAGE_SIZE))
    
    train_loader = make_dataloader(train_set, BATCH_SIZE, shuffle=True)
    val_loader   = make_dataloader(val_set,   BATCH_SIZE, shuffle=False)
    
    model = get_oracle_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, ORACLE_PATH, epoch=epoch+1, val_acc=acc)
            
    load_checkpoint(ORACLE_PATH, model, device)
    return model

# %%
def filter_images(model):
    """Prune synthetic datasets by removing samples that fail Oracle class verification."""
    label_to_idx, _ = build_label_map(CSV_PATH)
    model.eval()

    for gen_model in GENERATIVE_MODELS:
        gen_dir = os.path.join(BASE_DIR, "data", "generated", gen_model)
        csv_in  = os.path.join(gen_dir, "generated.csv")
        csv_out = os.path.join(gen_dir, "filtered_generated.csv")

        if not os.path.exists(csv_in):
            print(f"Skipping {gen_model}: {csv_in} not found.")
            continue

        df = pd.read_csv(csv_in)
        dataset = AugmentedButterflyDataset(df, label_to_idx, transform=get_eval_transform(IMAGE_SIZE))
        loader = make_dataloader(dataset, BATCH_SIZE, shuffle=False)

        results = []
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc=f"Scanning {gen_model}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, dim=1)
                
                for j in range(len(imgs)):
                    predicted_idx = preds[j].item()
                    intended_idx = labels[j].item()
                    confidence = conf[j].item()
                    
                    # Accept if prediction aligns with intent and meets confidence threshold
                    is_kept = (predicted_idx == intended_idx) and (confidence >= CONF_THRESHOLD)
                    results.append(is_kept)

        df_filtered = df[results].copy()
        df_filtered.to_csv(csv_out, index=False)
        
        print(f"  {gen_model.upper()}: Retained {len(df_filtered)}/{len(df)} ({len(df_filtered)/len(df)*100:.1f}%) samples.")

# %%
if __name__ == "__main__":
    oracle = train_oracle()
    filter_images(oracle)
