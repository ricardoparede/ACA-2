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
# # Conditional BigGAN (Simplified for 64x64)
# ### Butterfly Dataset — 75-class conditional image generation
#
# Upgrades the generative pipeline by implementing BigGAN features:
# - Residual Blocks in both Generator and Discriminator for massive capacity.
# - Spectral Normalization applied to all Convolutional and Linear layers.
# - Projection Discriminator for elegant class conditioning.
# - Hinge Loss (Standard for BigGAN architectures).

# %%
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# ── Device setup ─────────────────────────────────────────────────────────
device = torch.device(
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# %%
sys.path.insert(0, os.path.abspath(".."))

from src.dataset import (
    ButterflyDataset, get_splits, build_label_map,
    get_train_transform, get_eval_transform, make_dataloader,
)
from src.utils import (
    save_checkpoint, load_checkpoint,
    visualize_grid, tensor_to_pil
)
from src.metrics import compute_fid, compute_inception_score

# %%
# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath("..")
IMG_DIR     = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH    = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "data", "generated", "biggan")
NETG_PATH   = os.path.join(BASE_DIR, "saved_models", "biggan_generator.pth")
NETD_PATH   = os.path.join(BASE_DIR, "saved_models", "biggan_critic.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE  = 64
BATCH_SIZE  = 64
LATENT_DIM  = 120
NUM_CLASSES = 75
EMB_DIM     = 120

NUM_EPOCHS  = 200
SAVE_EVERY  = 20

# %%
label_to_idx, idx_to_label = build_label_map(CSV_PATH)
num_classes = len(label_to_idx)

train_df, val_df, test_df = get_splits(CSV_PATH, train_ratio=0.70, val_ratio=0.15, seed=42)
train_set    = ButterflyDataset(train_df, IMG_DIR, label_to_idx, transform=get_train_transform(IMAGE_SIZE))
train_loader = make_dataloader(train_set, BATCH_SIZE, shuffle=True)
print(f"Batches: {len(train_loader)}")

# %% [markdown]
# ## Model Architecture: BigGAN Residual Blocks

# %%
class GBlock(nn.Module):
    """Generator Residual Block with Upsampling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.up    = nn.Upsample(scale_factor=2)
        # Skip connection needs to adjust channels and resolution
        self.skip  = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

    def forward(self, x):
        h = F.relu(self.bn1(x))
        h = self.up(h)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        
        s = self.up(x)
        s = self.skip(s)
        return h + s

class DBlock(nn.Module):
    """Discriminator Residual Block with Downsampling"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        if downsample:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.pool(h)

        s = self.skip(x)
        if self.downsample:
            s = self.pool(s)
        return h + s

# %%
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, emb_dim=EMB_DIM):
        super().__init__()
        self.class_emb = spectral_norm(nn.Embedding(num_classes, emb_dim))
        
        in_ch = latent_dim + emb_dim
        self.init_linear = spectral_norm(nn.Linear(in_ch, 4 * 4 * 512))
        
        self.blocks = nn.Sequential(
            GBlock(512, 256), # 4x4 -> 8x8
            GBlock(256, 128), # 8x8 -> 16x16
            GBlock(128, 64),  # 16x16 -> 32x32
            GBlock(64, 32),   # 32x32 -> 64x64
        )
        self.out_bn = nn.BatchNorm2d(32)
        self.out_conv = spectral_norm(nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, noise, label):
        c = self.class_emb(label)
        z = torch.cat([noise, c], dim=1)
        h = self.init_linear(z)
        h = h.view(h.size(0), 512, 4, 4)
        
        h = self.blocks(h)
        h = F.relu(self.out_bn(h))
        h = self.out_conv(h)
        return torch.tanh(h)

class Discriminator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.blocks = nn.Sequential(
            DBlock(3, 64, downsample=True),    # 64x64 -> 32x32
            DBlock(64, 128, downsample=True),  # 32x32 -> 16x16
            DBlock(128, 256, downsample=True), # 16x16 -> 8x8
            DBlock(256, 512, downsample=True), # 8x8 -> 4x4
            DBlock(512, 512, downsample=False) # 4x4 -> 4x4
        )
        self.out_linear = spectral_norm(nn.Linear(512, 1))
        # Projection Discriminator Embedding
        self.class_emb = spectral_norm(nn.Embedding(num_classes, 512))

    def forward(self, img, label):
        h = self.blocks(img)
        h = F.relu(h)
        h = torch.sum(h, dim=[2, 3]) # Global Average Pooling
        
        out = self.out_linear(h)
        
        # Projection Conditioning: Add the dot product of features and class embedding
        c = self.class_emb(label)
        out += torch.sum(c * h, dim=1, keepdim=True)
        return out.view(-1)

# %%
netG = Generator().to(device)
netD = Discriminator().to(device)

print(f"Generator parameters    : {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

# BigGAN uses TTUR natively with specific Betas
LR_D = 0.0004
LR_G = 0.0001
BETAS = (0.0, 0.999)

optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=BETAS)
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=BETAS)

# %%
# ── Training Loop (Hinge Loss) ───────────────────────────────────────────
G_losses, D_losses = [], []

for epoch in range(NUM_EPOCHS):
    epoch_G, epoch_D = 0.0, 0.0
    
    for i, (data, labels) in enumerate(train_loader):
        real_imgs = data.to(device)
        lbls = labels.to(device)
        batch_size = real_imgs.size(0)

        # ─── 1. Update Discriminator ─────────────────────────────────────
        netD.zero_grad()
        
        # Real Pass
        d_real = netD(real_imgs, lbls)
        # Hinge Loss for Real
        loss_d_real = F.relu(1.0 - d_real).mean() 
        
        # Fake Pass
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_imgs = netG(noise, lbls)
        d_fake = netD(fake_imgs.detach(), lbls)
        # Hinge Loss for Fake
        loss_d_fake = F.relu(1.0 + d_fake).mean() 
        
        errD = loss_d_real + loss_d_fake
        errD.backward()
        optimizerD.step()
        
        epoch_D += errD.item()

        # ─── 2. Update Generator ─────────────────────────────────────────
        netG.zero_grad()
        
        # Generator wants Discriminator to think fake is real
        d_fake_g = netD(fake_imgs, lbls)
        errG = -d_fake_g.mean()
        
        errG.backward()
        optimizerG.step()
        
        epoch_G += errG.item()

        if i % 50 == 0:
            print(f"[{epoch+1}/{NUM_EPOCHS}][{i}/{len(train_loader)}] "
                  f"Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}")

    G_losses.append(epoch_G / len(train_loader))
    D_losses.append(epoch_D / len(train_loader))

    # ── Save generated samples every SAVE_EVERY epochs ───────────────────
    if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        netG.eval()
        with torch.no_grad():
            sample_lbl = torch.arange(min(16, NUM_CLASSES), device=device)
            noise_fixed = torch.randn(len(sample_lbl), LATENT_DIM, device=device)
            fake_sample = (netG(noise_fixed, sample_lbl).detach().cpu() + 1) * 0.5
            
        grid = make_grid(fake_sample, nrow=8, padding=2)
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"BigGAN Epoch {epoch+1}")
        plt.show()
        netG.train()

        save_checkpoint(netG, NETG_PATH, epoch=epoch + 1)
        save_checkpoint(netD, NETD_PATH, epoch=epoch + 1)

print("BigGAN Training complete.")

# %%
# ── Plot Loss Curves ──────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator (Critic)")
plt.title("BigGAN Training Curve (Hinge Loss)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# ── Generate Data and Evaluate ────────────────────────────────────────────
load_checkpoint(NETG_PATH, netG, device)
netG = netG.to(device)
netG.eval()

class_counts = Counter(train_df["label"].tolist())
TARGET_PER_CLASS = max(class_counts.values())

generated_records = []
with torch.no_grad():
    for class_name, class_idx in label_to_idx.items():
        existing = class_counts.get(class_name, 0)
        n_gen = max(0, TARGET_PER_CLASS - existing)
        if n_gen == 0: continue

        all_gen = []
        remaining = n_gen
        while remaining > 0:
            batch_n = min(remaining, BATCH_SIZE)
            noise = torch.randn(batch_n, LATENT_DIM, device=device)
            lbl   = torch.full((batch_n,), class_idx, dtype=torch.long, device=device)
            gen   = netG(noise, lbl).cpu()
            all_gen.append(gen)
            remaining -= batch_n

        gen_imgs = torch.cat(all_gen, dim=0)
        gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)

        for j, img_t in enumerate(gen_imgs):
            fname = f"biggan_{class_idx:03d}_{j:04d}.jpg"
            tensor_to_pil(img_t).save(os.path.join(SAVE_DIR, fname))
            generated_records.append({"filename": fname, "label": class_name, "img_dir": SAVE_DIR})

        print(f"  [{class_idx:02d}] {class_name}: generated {n_gen}")

gen_df = pd.DataFrame(generated_records)
gen_df.to_csv(os.path.join(SAVE_DIR, "generated.csv"), index=False)
print(f"Total BigGAN generated: {len(gen_df):,} | CSV saved.")

def collect_tensor(loader):
    return torch.cat([imgs for imgs, _ in loader], dim=0)

real_imgs = collect_tensor(train_loader)
gen_set  = ButterflyDataset(gen_df, SAVE_DIR, label_to_idx, transform=get_eval_transform(IMAGE_SIZE))
gen_dl   = make_dataloader(gen_set, BATCH_SIZE, shuffle=False)
gen_imgs = collect_tensor(gen_dl)

fid_score = compute_fid(real_imgs, gen_imgs, device=device)
is_mean, is_std = compute_inception_score(gen_imgs, device=device)

print("\n=== BigGAN Generative Quality Summary ===")
print(f"  FID : {fid_score:.2f}")
print(f"  IS  : {is_mean:.3f} ± {is_std:.3f}")
