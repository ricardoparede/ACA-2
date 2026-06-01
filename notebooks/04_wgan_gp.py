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
# # Conditional WGAN-GP
# ### Butterfly Dataset — 75-class conditional image generation
#
# Implementation details:
# - Wasserstein objective with Gradient Penalty (GP) for Lipschitz constraint
# - Critic-heavy training (5 iterations per Generator update)
# - No batch normalization in Critic to avoid inter-sample dependency in GP
# - TTUR (Two Time-Scale Update Rule) for stable convergence
#

# %%
import os, sys
import numpy as np
import torch
import torch.nn as nn
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
BASE_DIR    = os.path.abspath("..")
IMG_DIR     = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH    = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "data", "generated", "wgan_gp")
NETG_PATH   = os.path.join(BASE_DIR, "saved_models", "wgan_gp_generator.pth")
NETD_PATH   = os.path.join(BASE_DIR, "saved_models", "wgan_gp_critic.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE  = 64
BATCH_SIZE  = 64
LATENT_DIM  = 100
NUM_CLASSES = 75
EMB_DIM     = 100

NUM_EPOCHS  = 200
SAVE_EVERY  = 20


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
# ### Generator (netG)
# Residual-based or ConvTranspose upsampler mapping `noise` + `class embedding` to RGB.
#
# ### Critic (netD)
# Estimates the Wasserstein distance. Uses InstanceNorm or no normalization
# to maintain Gradient Penalty validity.
#

# %%
def weights_init(m):
    """Initialize weights using standard DCGAN/WGAN normal distributions."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# %%
class Generator(nn.Module):
    """Conditional Generator: maps latent vector + class index to image space."""

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, emb_dim=EMB_DIM):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        in_ch = latent_dim + emb_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, label):
        c = self.class_emb(label).unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([noise, c], dim=1)
        return self.main(z)



# %%
class Discriminator(nn.Module):
    """Conditional Critic: estimates Wasserstein score for (image, label) pairs."""

    def __init__(self, num_classes=NUM_CLASSES, emb_dim=EMB_DIM, image_size=IMAGE_SIZE):
        super().__init__()
        self.image_size = image_size
        
        self.class_proj = nn.Sequential(
            nn.Embedding(num_classes, emb_dim),
            nn.Linear(emb_dim, image_size * image_size),
        )
        
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, label):
        B = img.size(0)
        c = self.class_proj(label).view(B, 1, self.image_size, self.image_size)
        x = torch.cat([img, c], dim=1)
        return self.model(x).view(B)


# %%
def compute_gradient_penalty(netD, real_samples, fake_samples, labels, device):
    """Enforce 1-Lipschitz continuity via L2 norm of gradients on interpolated samples."""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = netD(interpolates, labels)
    fake = torch.ones(real_samples.size(0), device=device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# %%
# ── Model Initialization and Optimizers ──────────────────────────────────
netG = Generator(LATENT_DIM, NUM_CLASSES, EMB_DIM).to(device)
netD = Discriminator(NUM_CLASSES, EMB_DIM, IMAGE_SIZE).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

LR_D = 0.0004
LR_G = 0.0001
BETAS_WGAN = (0.0, 0.9)

optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=BETAS_WGAN)
optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=BETAS_WGAN)

CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0


# %%
# ── Training Loop ─────────────────────────────────────────────────────────
G_losses, D_losses = [], []

for epoch in range(NUM_EPOCHS):
    epoch_G, epoch_D = 0.0, 0.0
    
    for i, (data, labels) in enumerate(train_loader):
        real_cpu = data.to(device)
        lbl = labels.to(device)
        batch_size = real_cpu.size(0)

        # ─── Update D (Critic): minimize -D(x) + D(G(z)) + λ*GP ──────────
        netD.zero_grad()
        
        d_real = netD(real_cpu, lbl).mean()
        
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake = netG(noise, lbl)
        d_fake = netD(fake.detach(), lbl).mean()
        
        gp = compute_gradient_penalty(netD, real_cpu, fake.detach(), lbl, device)
        
        errD = -d_real + d_fake + LAMBDA_GP * gp
        errD.backward()
        optimizerD.step()
        
        epoch_D += errD.item()

        # ─── Update G: minimize -D(G(z)) ─────────────────────────────────
        if i % CRITIC_ITERATIONS == 0:
            netG.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise, lbl)
            errG = -netD(fake, lbl).mean()
            errG.backward()
            optimizerG.step()
            epoch_G += errG.item()

        if i % 50 == 0:
            print(f"[{epoch+1}/{NUM_EPOCHS}][{i}/{len(train_loader)}] "
                  f"Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}  "
                  f"D(x): {d_real.item():.4f}  D(G(z)): {d_fake.item():.4f}")

    num_G_updates = max(1, len(train_loader) // CRITIC_ITERATIONS)
    G_losses.append(epoch_G / num_G_updates)
    D_losses.append(epoch_D / len(train_loader))

    # ── Checkpointing and Periodic Visualization ─────────────────────────
    if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
        netG.eval()
        with torch.no_grad():
            sample_lbl  = torch.arange(min(16, NUM_CLASSES), device=device)
            noise_fixed = torch.randn(len(sample_lbl), LATENT_DIM, 1, 1, device=device)
            fake_sample = (netG(noise_fixed, sample_lbl).detach().cpu() + 1) * 0.5
            
        grid = make_grid(fake_sample, nrow=8, padding=2)
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"Epoch {epoch+1}")
        plt.show()
        netG.train()

        save_checkpoint(netG, NETG_PATH, epoch=epoch + 1)
        save_checkpoint(netD, NETD_PATH, epoch=epoch + 1)
        print(f"Checkpoints saved at epoch {epoch+1}")

print("Training complete.")


# %%
# ── Evaluation: Loss Curves ──────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator (Critic)")
plt.title("WGAN-GP Training Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %%
# ── Sample Generation and Dataset Balancing ───────────────────────────────
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
        if n_gen == 0:
            continue

        all_gen = []
        remaining = n_gen
        while remaining > 0:
            batch_n = min(remaining, BATCH_SIZE)
            noise = torch.randn(batch_n, LATENT_DIM, 1, 1, device=device)
            lbl   = torch.full((batch_n,), class_idx, dtype=torch.long, device=device)
            gen   = netG(noise, lbl).cpu()
            all_gen.append(gen)
            remaining -= batch_n

        gen_imgs = torch.cat(all_gen, dim=0)
        gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)

        for j, img_t in enumerate(gen_imgs):
            fname = f"wgan_gp_{class_idx:03d}_{j:04d}.jpg"
            tensor_to_pil(img_t).save(os.path.join(SAVE_DIR, fname))
            generated_records.append({
                "filename": fname,
                "label": class_name,
                "img_dir": SAVE_DIR,
            })

        print(f"  [{class_idx:02d}] {class_name}: generated {n_gen}")

gen_df = pd.DataFrame(generated_records)
gen_df.to_csv(os.path.join(SAVE_DIR, "generated.csv"), index=False)
print(f"\nTotal generated: {len(gen_df):,}  |  CSV saved.")


# %%
# ── Generate images per class to balance the dataset ──────────────────────
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
        if n_gen == 0:
            continue

        all_gen = []
        remaining = n_gen
        while remaining > 0:
            batch_n = min(remaining, BATCH_SIZE)
            noise = torch.randn(batch_n, LATENT_DIM, 1, 1, device=device)
            lbl   = torch.full((batch_n,), class_idx, dtype=torch.long, device=device)
            gen   = netG(noise, lbl).cpu()        # (batch_n, 3, 64, 64) in [-1,1]
            all_gen.append(gen)
            remaining -= batch_n

        gen_imgs = torch.cat(all_gen, dim=0)      # (n_gen, 3, 64, 64)
        gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)   # → [0,1]

        for j, img_t in enumerate(gen_imgs):
            fname = f"wgan_gp_{class_idx:03d}_{j:04d}.jpg"
            tensor_to_pil(img_t).save(os.path.join(SAVE_DIR, fname))
            generated_records.append({
                "filename": fname,
                "label": class_name,
                "img_dir": SAVE_DIR,
            })

        print(f"  [{class_idx:02d}] {class_name}: generated {n_gen}")

gen_df = pd.DataFrame(generated_records)
gen_df.to_csv(os.path.join(SAVE_DIR, "generated.csv"), index=False)
print(f"\nTotal generated: {len(gen_df):,}  |  CSV saved.")


# %% [markdown]
# ## Evaluation of Generative Quality
#
# FID and IS computed globally (all classes) — same protocol as the cVAE notebook.
# SSIM is omitted for GANs (outputs are not reconstruction-paired with real images).
#

# %%
def collect_tensor(loader):
    batches = [imgs for imgs, _ in loader]
    return torch.cat(batches, dim=0)

print("Collecting real images…")
real_imgs = collect_tensor(train_loader)

print("Collecting WGAN-GP-generated images…")
gen_set  = ButterflyDataset(gen_df, SAVE_DIR, label_to_idx,
                             transform=get_eval_transform(IMAGE_SIZE))
gen_dl   = make_dataloader(gen_set, BATCH_SIZE, shuffle=False)
gen_imgs = collect_tensor(gen_dl)

print(f"Real: {real_imgs.shape}  |  Generated: {gen_imgs.shape}")


# %%
print("Computing FID…")
fid_score = compute_fid(real_imgs, gen_imgs, device=device)
print(f"FID: {fid_score:.2f}  (lower is better)")

print("Computing IS…")
is_mean, is_std = compute_inception_score(gen_imgs, device=device)
print(f"IS:  {is_mean:.3f} ± {is_std:.3f}  (higher is better)")

print("\n=== WGAN-GP Generative Quality Summary ===")
print(f"  FID : {fid_score:.2f}")
print(f"  IS  : {is_mean:.3f} ± {is_std:.3f}")

