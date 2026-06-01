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
# # Conditional Diffusion Model (DDPM)
# ### Butterfly Dataset — 75-class conditional image generation
#
# Implementation details:
# - Linear noise schedule for forward diffusion (1000 steps)
# - UNet architecture with sinusoidal time embeddings and global self-attention
# - Classifier-Free Guidance (CFG) for steerable class conditioning
# - Training objective: Mean Squared Error on predicted noise ε
#

# %%
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# ── Device Configuration ──────────────────────────────────────────────────
device = torch.device(
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# %%
# ── Project Path Setup ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(".."))

from src.dataset import (
    ButterflyDataset, get_splits, build_label_map,
    get_train_transform, get_eval_transform, make_dataloader,
)
from src.utils import (
    save_checkpoint, load_checkpoint,
    visualize_grid, plot_losses,
    EarlyStopping, tensor_to_pil, denormalize,
)
from src.metrics import compute_fid, compute_inception_score

# %%
# ── Hyperparameters and Paths ──────────────────────────────────────────────
BASE_DIR     = os.path.abspath("..")
IMG_DIR      = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH     = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
SAVE_DIR     = os.path.join(BASE_DIR, "data", "generated", "diffusion")
MODEL_PATH   = os.path.join(BASE_DIR, "saved_models", "diffusion_unet.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE   = 64
BATCH_SIZE   = 64
NUM_CLASSES  = 75
TIME_EMB_DIM = 256

T_STEPS      = 1000
BETA_START   = 1e-4
BETA_END     = 2e-2

NUM_EPOCHS          = 300
SAVE_EVERY          = 20
LR                  = 2e-4

NULL_CLASS_IDX      = NUM_CLASSES
CFG_DROPOUT_RATE    = 0.1
CFG_GUIDANCE_SCALE  = 3.0

# %%
# ── Dataset Initialization ────────────────────────────────────────────────
label_to_idx, idx_to_label = build_label_map(CSV_PATH)
num_classes = len(label_to_idx)

train_df, val_df, test_df = get_splits(CSV_PATH, train_ratio=0.70, val_ratio=0.15, seed=42)
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

train_set    = ButterflyDataset(train_df, IMG_DIR, label_to_idx,
                                transform=get_train_transform(IMAGE_SIZE))
train_loader = make_dataloader(train_set, BATCH_SIZE, shuffle=True)


# %%
# ── Batch Visualization ───────────────────────────────────────────────────
imgs, labs = next(iter(train_loader))
visualize_grid(imgs, labs, idx_to_label, n=16, nrow=8,
               title="Training samples", denorm=True)


# %% [markdown]
# ## Model Architecture
#
# ### UNet (Noise Predictor)
# Estimates the noise ε added to image `x_0` at timestep `t`.
# Features sinusoidal time embeddings and affine conditioning in Residual Blocks.
#
# ### Diffusion Engine
# Implements the forward variance schedule and reverse sampling logic.
# Supports Classifier-Free Guidance (CFG) for enhanced class adherence.
#

# %%
class SinusoidalPositionEmbeddings(nn.Module):
    """Scalar-to-vector transformation for timestep t using trigonometric frequencies."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device   = time.device
        half_dim = self.dim // 2
        scale    = math.log(10000) / (half_dim - 1)
        freqs    = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb      = time[:, None].float() * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with GroupNorm and affine conditioning injection."""
    def __init__(self, in_ch, out_ch, cond_dim, num_groups=8):
        super().__init__()
        ng1 = min(num_groups, in_ch)
        ng2 = min(num_groups, out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.block1    = nn.Sequential(
            nn.GroupNorm(ng1, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2    = nn.Sequential(
            nn.GroupNorm(ng2, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.block1(x)
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """Spatial self-attention for global structural consistency."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha      = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ln       = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat       = x.view(B, C, H * W).transpose(1, 2)
        x_norm       = self.ln(x_flat)
        attn_out, _  = self.mha(x_norm, x_norm, x_norm)
        x_flat       = x_flat + attn_out
        return x_flat.transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    """Conditional UNet for iterative denoising of 64x64 images."""
    def __init__(self, num_classes=NUM_CLASSES, time_emb_dim=TIME_EMB_DIM):
        super().__init__()
        cond_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # ── Encoder Path ──────────────────────────────────────────────────
        self.down1   = ResBlock(3,   64,  cond_dim)
        self.pool1   = nn.AvgPool2d(2)
        self.down2   = ResBlock(64,  128, cond_dim)
        self.pool2   = nn.AvgPool2d(2)
        self.down3   = ResBlock(128, 256, cond_dim)
        self.pool3   = nn.AvgPool2d(2)

        # ── Bottleneck Path ───────────────────────────────────────────────
        self.mid1    = ResBlock(256, 512, cond_dim)
        self.attn    = SelfAttention(512)
        self.mid2    = ResBlock(512, 256, cond_dim)

        # ── Decoder Path ──────────────────────────────────────────────────
        self.up3     = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up3_res = ResBlock(256 + 256, 256, cond_dim)
        self.up2     = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up2_res = ResBlock(128 + 128, 128, cond_dim)
        self.up1     = nn.ConvTranspose2d(128,  64, 2, 2)
        self.up1_res = ResBlock( 64 +  64,  64, cond_dim)

        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, t, label):
        cond = self.time_mlp(t) + self.class_emb(label)

        d1 = self.down1(x,               cond)
        d2 = self.down2(self.pool1(d1),  cond)
        d3 = self.down3(self.pool2(d2),  cond)

        m  = self.mid1(self.pool3(d3),   cond)
        m  = self.attn(m)
        m  = self.mid2(m,                cond)

        u3 = self.up3_res(torch.cat([self.up3(m),  d3], dim=1), cond)
        u2 = self.up2_res(torch.cat([self.up2(u3), d2], dim=1), cond)
        u1 = self.up1_res(torch.cat([self.up1(u2), d1], dim=1), cond)

        return self.out_conv(F.silu(self.out_norm(u1)))


# %%
class DiffusionModel(nn.Module):
    """DDPM state manager: handles variance schedules and denoising steps."""
    def __init__(self, model: nn.Module, n_steps=T_STEPS, device='cpu'):
        super().__init__()
        self.model   = model
        self.n_steps = n_steps

        self.beta      = torch.linspace(BETA_START, BETA_END, n_steps).to(device)
        self.alpha     = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2    = self.beta

    def forward_process(self, x0, t):
        """Compute q(x_t | x_0) by injecting Gaussian noise scaled by alpha_bar."""
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        noise       = torch.randn_like(x0)
        xt          = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return xt, noise

    def _ddpm_step(self, xt, t: int, pred_noise):
        """Execute one reverse step p_theta(x_{t-1} | x_t) using predicted noise."""
        beta_t      = self.beta[t]
        alpha_t     = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        mu = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

        if t == 0:
            return mu

        sigma = torch.sqrt(beta_t)
        return mu + sigma * torch.randn_like(xt)

    def reverse(self, xt, t: int, label):
        """Estimate x_{t-1} from x_t via UNet noise prediction."""
        t_tensor   = torch.full((xt.size(0),), t, dtype=torch.long, device=xt.device)
        pred_noise = self.model(xt, t_tensor, label)
        return self._ddpm_step(xt, t, pred_noise)

    @torch.no_grad()
    def sample(self, n: int, label, device):
        """Generate samples by traversing the full reverse diffusion chain."""
        x = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        for t in reversed(range(self.n_steps)):
            x = self.reverse(x, t, label)
        return x


@torch.no_grad()
def sample_cfg(unet, diffusion, labels, image_size=IMAGE_SIZE,
               guidance_scale=CFG_GUIDANCE_SCALE, device=device):
    """Sample with Classifier-Free Guidance for enhanced class adherence."""
    unet.eval()
    B             = labels.size(0)
    uncond_labels = torch.full_like(labels, NULL_CLASS_IDX)

    x = torch.randn(B, 3, image_size, image_size, device=device)

    for t_step in reversed(range(diffusion.n_steps)):
        t = torch.full((B,), t_step, dtype=torch.long, device=device)

        cond_noise   = unet(x, t, labels)
        uncond_noise = unet(x, t, uncond_labels)

        # Guidance extrapolation: ε = ε_uncond + w * (ε_cond - ε_uncond)
        pred_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)
        x = diffusion._ddpm_step(x, t_step, pred_noise)

    return x


# %%
# ── Model Initialization and Training Setup ───────────────────────────────
unet      = UNet(NUM_CLASSES, TIME_EMB_DIM).to(device)
diffusion = DiffusionModel(unet, T_STEPS, device).to(device)

print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

optimizer = optim.Adam(unet.parameters(), lr=LR)


# %%
# ── Training Loop ─────────────────────────────────────────────────────────
losses = []

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    for i, (data, labels) in enumerate(train_loader):
        x0  = data.to(device)
        lbl = labels.to(device)

        # Label dropout for CFG training
        mask = torch.rand(lbl.size(0), device=device) < CFG_DROPOUT_RATE
        lbl  = lbl.masked_fill(mask, NULL_CLASS_IDX)

        t = torch.randint(0, T_STEPS, (x0.size(0),), device=device)
        xt, noise = diffusion.forward_process(x0, t)
        pred_noise = unet(xt, t, lbl)

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 50 == 0:
            print(f"[{epoch+1}/{NUM_EPOCHS}][{i}/{len(train_loader)}]  Loss: {loss.item():.4f}")

    losses.append(epoch_loss / len(train_loader))

    # ── Checkpointing and Periodic Sampling ──────────────────────────────
    if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        unet.eval()
        with torch.no_grad():
            sample_lbl  = torch.arange(min(16, NUM_CLASSES), device=device)
            fake_sample = sample_cfg(unet, diffusion, sample_lbl,
                                     guidance_scale=CFG_GUIDANCE_SCALE, device=device)
            fake_sample = ((fake_sample.cpu() + 1) * 0.5).clamp(0, 1)

        grid = make_grid(fake_sample, nrow=8, padding=2)
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.show()
        unet.train()

        save_checkpoint(unet, MODEL_PATH, epoch=epoch + 1)
        print(f"Checkpoint saved at epoch {epoch+1}")

print("Training complete.")


# %%
# ── Evaluation: Loss Curve ───────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("DDPM Training Curve")
plt.tight_layout()
plt.show()


# %%
# ── Sample Generation and Dataset Balancing ───────────────────────────────
load_checkpoint(MODEL_PATH, unet, device)
unet = unet.to(device)
unet.eval()

class_counts     = Counter(train_df["label"].tolist())
TARGET_PER_CLASS = max(class_counts.values())

generated_records = []
with torch.no_grad():
    for class_name, class_idx in label_to_idx.items():
        existing  = class_counts.get(class_name, 0)
        n_gen     = max(0, TARGET_PER_CLASS - existing)
        if n_gen == 0:
            continue

        all_gen   = []
        remaining = n_gen
        while remaining > 0:
            batch_n = min(remaining, BATCH_SIZE)
            lbl     = torch.full((batch_n,), class_idx, dtype=torch.long, device=device)
            gen     = sample_cfg(unet, diffusion, lbl,
                                 guidance_scale=CFG_GUIDANCE_SCALE, device=device).cpu()
            all_gen.append(gen)
            remaining -= batch_n

        gen_imgs = torch.cat(all_gen, dim=0)
        gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)

        for j, img_t in enumerate(gen_imgs):
            fname = f"diffusion_{class_idx:03d}_{j:04d}.jpg"
            tensor_to_pil(img_t).save(os.path.join(SAVE_DIR, fname))
            generated_records.append({
                "filename": fname,
                "label":    class_name,
                "img_dir":  SAVE_DIR,
            })

        print(f"  [{class_idx:02d}] {class_name}: generated {n_gen}")

gen_df = pd.DataFrame(generated_records)
gen_df.to_csv(os.path.join(SAVE_DIR, "generated.csv"), index=False)
print(f"\nTotal generated: {len(gen_df):,}  |  CSV saved.")
