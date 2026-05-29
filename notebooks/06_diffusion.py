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
# Implements a Denoising Diffusion Probabilistic Model (DDPM) for butterfly image synthesis,
# adapted from the `DiffusionModel` class structure in `exemplos_modelos/DIFFUSION.py`:
# - Linear beta noise schedule (as suggested in DIFFUSION.py: *"some papers argue it can be linearly spaced!"*)
# - UNet noise predictor with sinusoidal time embeddings (mirrors the per-timestep `network_tail` in `MLP`)
# - Same `forward_process` / `reverse` / `sample` method names as in DIFFUSION.py
# - DDPM training objective: MSE on predicted noise ε (replaces the KL loss used for 2-D data)
#
# Class conditioning via `nn.Embedding` summed into the time embedding at every ResBlock,
# following the same conditioning pattern used in 03_gan and 04_WGAN_GP.

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
    visualize_grid, plot_losses,
    EarlyStopping, tensor_to_pil, denormalize,
)
from src.metrics import compute_fid, compute_inception_score

# %%
# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath("..")
IMG_DIR      = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH     = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
SAVE_DIR     = os.path.join(BASE_DIR, "data", "generated", "diffusion")
MODEL_PATH   = os.path.join(BASE_DIR, "saved_models", "diffusion_unet.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

IMAGE_SIZE   = 64       # matches TP2-students.ipynb
BATCH_SIZE   = 64       # RTX 5060 Laptop 8.5 GB VRAM
NUM_CLASSES  = 75
TIME_EMB_DIM = 256      # sinusoidal embedding dimension

# Diffusion hyperparameters — following DIFFUSION.py structure
T_STEPS      = 1000     # total diffusion timesteps (DIFFUSION.py uses N=40 for 2-D; 1000 is standard for images)
BETA_START   = 1e-4     # linear schedule start  (DIFFUSION.py: "linearly spaced!")
BETA_END     = 2e-2     # linear schedule end

NUM_EPOCHS          = 300
SAVE_EVERY          = 20
LR                  = 2e-4     # standard Adam lr for DDPM

# ── Classifier-Free Guidance (CFG) hyperparameters ────────────────────────
NULL_CLASS_IDX      = NUM_CLASSES   # index 75 = "null" class for unconditional pass
CFG_DROPOUT_RATE    = 0.1           # probability of dropping class label during training
CFG_GUIDANCE_SCALE  = 3.0           # guidance scale w (try 3.0, 4.0, 5.0 at generation time)

# %%
label_to_idx, idx_to_label = build_label_map(CSV_PATH)
num_classes = len(label_to_idx)

train_df, val_df, test_df = get_splits(CSV_PATH, train_ratio=0.70, val_ratio=0.15, seed=42)
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

train_set    = ButterflyDataset(train_df, IMG_DIR, label_to_idx,
                                transform=get_train_transform(IMAGE_SIZE))
train_loader = make_dataloader(train_set, BATCH_SIZE, shuffle=True)
print(f"Batches: {len(train_loader)}")

# %%
imgs, labs = next(iter(train_loader))
visualize_grid(imgs, labs, idx_to_label, n=16, nrow=8,
               title="Training samples", denorm=True)

# %% [markdown]
# ## Model Architecture
#
# ### UNet (noise predictor)
# Adapts the `MLP` noise-predictor concept from DIFFUSION.py to RGB images via a UNet:
# - **Encoder** : 3→64→128→256 with `AvgPool2d` downsampling
# - **Bottleneck**: 256→512→256
# - **Decoder**  : skip connections + `ConvTranspose2d` upsampling → 3 output channels
#
# **Conditioning**: sinusoidal time embedding + class embedding (both `TIME_EMB_DIM`)
# are summed and injected into every `ResBlock` as a channel-wise shift,
# mirroring how DIFFUSION.py injects timestep `t` into `MLP.network_tail[t]`.
#
# ### DiffusionModel
# Wraps the UNet following the exact DIFFUSION.py class pattern:
# - `forward_process(x0, t)` → q(x_t|x_0): adds noise at timestep t, returns (x_t, ε)
# - `reverse(x_t, t, label)`  → p_θ(x_{t-1}|x_t): one denoising step
# - `sample(n, label, device)` → full reverse chain from Gaussian noise → clean image

# %%
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal timestep embeddings — converts scalar t into a fixed-frequency vector.
    Mirrors the per-timestep indexing in DIFFUSION.py's MLP (network_tail[t]).
    """
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
    """
    Residual block with GroupNorm + SiLU activations and conditioning injection.
    The condition vector (time + class) is projected to `out_ch` and added channel-wise,
    implementing the affine shift described in the DDPM architecture literature.
    """
    def __init__(self, in_ch, out_ch, cond_dim, num_groups=8):
        super().__init__()
        # num_groups must divide the channel count; clamp for small in_ch (e.g. 3)
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
    """
    Multi-head self-attention for global structure (e.g. wing symmetry).
    Applied at the 8×8 bottleneck where full spatial attention is cheap.
    Mirrors the global-context idea from DIFFUSION.py's latent diffusion context.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha      = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ln       = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Flatten to (B, H*W, C) — every pixel attends to every other pixel
        x_flat       = x.view(B, C, H * W).transpose(1, 2)
        x_norm       = self.ln(x_flat)
        attn_out, _  = self.mha(x_norm, x_norm, x_norm)
        x_flat       = x_flat + attn_out            # residual connection
        return x_flat.transpose(1, 2).view(B, C, H, W)


class UNet(nn.Module):
    """
    Conditional UNet noise predictor for 64×64 RGB images.
    Input : noisy image (B, 3, 64, 64), timestep t (B,), class label (B,)
    Output: predicted noise ε (B, 3, 64, 64)
    """
    def __init__(self, num_classes=NUM_CLASSES, time_emb_dim=TIME_EMB_DIM):
        super().__init__()
        cond_dim = time_emb_dim

        # Time embedding: sinusoidal → 2-layer MLP (mirrors DIFFUSION.py's MLP structure)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Class embedding: +1 slot for the null class used in CFG
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # ── Encoder ──────────────────────────────────────────────────────
        self.down1   = ResBlock(3,   64,  cond_dim)    # (B, 64,  64, 64)
        self.pool1   = nn.AvgPool2d(2)                 # → 32×32
        self.down2   = ResBlock(64,  128, cond_dim)    # (B, 128, 32, 32)
        self.pool2   = nn.AvgPool2d(2)                 # → 16×16
        self.down3   = ResBlock(128, 256, cond_dim)    # (B, 256, 16, 16)
        self.pool3   = nn.AvgPool2d(2)                 # → 8×8

        # ── Bottleneck (Self-Attention for global structure) ───────────────
        self.mid1    = ResBlock(256, 512, cond_dim)    # (B, 512, 8, 8)
        self.attn    = SelfAttention(512)              # global attention at 8×8
        self.mid2    = ResBlock(512, 256, cond_dim)    # (B, 256, 8, 8)

        # ── Decoder ───────────────────────────────────────────────────────
        self.up3     = nn.ConvTranspose2d(256, 256, 2, 2)               # → 16×16
        self.up3_res = ResBlock(256 + 256, 256, cond_dim)
        self.up2     = nn.ConvTranspose2d(256, 128, 2, 2)               # → 32×32
        self.up2_res = ResBlock(128 + 128, 128, cond_dim)
        self.up1     = nn.ConvTranspose2d(128,  64, 2, 2)               # → 64×64
        self.up1_res = ResBlock( 64 +  64,  64, cond_dim)

        # Output head
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, t, label):
        # Combined condition: time embedding + class embedding (both TIME_EMB_DIM)
        cond = self.time_mlp(t) + self.class_emb(label)     # (B, cond_dim)

        # Encoder
        d1 = self.down1(x,               cond)   # (B, 64,  64, 64)
        d2 = self.down2(self.pool1(d1),  cond)   # (B, 128, 32, 32)
        d3 = self.down3(self.pool2(d2),  cond)   # (B, 256, 16, 16)

        # Bottleneck
        m  = self.mid1(self.pool3(d3),   cond)   # (B, 512,  8,  8)
        m  = self.attn(m)                         # global self-attention at 8×8
        m  = self.mid2(m,                cond)   # (B, 256,  8,  8)

        # Decoder with skip connections
        u3 = self.up3_res(torch.cat([self.up3(m),  d3], dim=1), cond)  # (B, 256, 16, 16)
        u2 = self.up2_res(torch.cat([self.up2(u3), d2], dim=1), cond)  # (B, 128, 32, 32)
        u1 = self.up1_res(torch.cat([self.up1(u2), d1], dim=1), cond)  # (B,  64, 64, 64)

        return self.out_conv(F.silu(self.out_norm(u1)))


# %%
class DiffusionModel(nn.Module):
    """
    DDPM wrapper — mirrors the DiffusionModel class in DIFFUSION.py.

    Forward process  : q(x_t | x_0) — adds noise progressively using alpha_bar
    Reverse process  : p_θ(x_{t-1}|x_t) — UNet predicts ε at each step
    Training objective: E[ ||ε - ε_θ(x_t, t, c)||² ]  (DDPM Eq. 14)

    Key variables match DIFFUSION.py naming:
        beta, alpha, alpha_bar, sigma2, n_steps
    """
    def __init__(self, model: nn.Module, n_steps=T_STEPS, device='cpu'):
        super().__init__()
        self.model   = model
        self.n_steps = n_steps

        # Linear beta schedule — DIFFUSION.py uses sigmoid-spaced betas but notes
        # "some papers argue that it can be linearly spaced!" — using linear here.
        self.beta      = torch.linspace(BETA_START, BETA_END, n_steps).to(device)
        self.alpha     = 1.0 - self.beta
        # alpha_bar: cumulative product — same as DIFFUSION.py alpha_bar
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # sigma2 matches DIFFUSION.py: same value as beta
        self.sigma2    = self.beta

    # ── Forward process ───────────────────────────────────────────────────
    def forward_process(self, x0, t):
        """
        q(x_t | x_0) = N( sqrt(ᾱ_t)·x0,  (1-ᾱ_t)·I )

        Mirrors DIFFUSION.py forward_process: computes noisy x_t and true noise ε.
        t is a batch tensor of timestep indices (one per sample).
        Returns (x_t, ε).
        """
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        noise       = torch.randn_like(x0)
        xt          = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return xt, noise

    # ── DDPM denoising step (shared by reverse and sample_cfg) ────────────
    def _ddpm_step(self, xt, t: int, pred_noise):
        """
        One DDPM denoising step given a pre-computed noise prediction.
        Extracted so that both reverse() and sample_cfg() can reuse it
        without duplicating the posterior mean / variance formula.
        """
        beta_t      = self.beta[t]
        alpha_t     = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        # DDPM posterior mean (Eq. 11)
        mu = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

        if t == 0:
            return mu

        # sigma2 = beta_t  (matches DIFFUSION.py: self.sigma2 = self.beta)
        sigma = torch.sqrt(beta_t)
        return mu + sigma * torch.randn_like(xt)

    # ── Reverse step ──────────────────────────────────────────────────────
    def reverse(self, xt, t: int, label):
        """
        One p_θ step: reconstruct x_{t-1} from x_t.
        Mirrors DIFFUSION.py reverse: predicts mu & sigma, then samples from them.
        t is a scalar integer.
        """
        t_tensor   = torch.full((xt.size(0),), t, dtype=torch.long, device=xt.device)
        pred_noise = self.model(xt, t_tensor, label)
        return self._ddpm_step(xt, t, pred_noise)

    # ── Standard sampling ─────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, n: int, label, device):
        """
        Full reverse chain: pure Gaussian noise → clean image.
        Mirrors DIFFUSION.py sample(): iterates reverse() from T-1 down to 0.
        """
        x = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        for t in reversed(range(self.n_steps)):
            x = self.reverse(x, t, label)
        return x


@torch.no_grad()
def sample_cfg(unet, diffusion, labels, image_size=IMAGE_SIZE,
               guidance_scale=CFG_GUIDANCE_SCALE, device=device):
    """
    Classifier-Free Guidance sampling.
    Runs UNet twice per step — once with the real class label (conditional)
    and once with the null class (unconditional) — then combines:
        ε = ε_uncond + guidance_scale × (ε_cond − ε_uncond)
    Higher guidance_scale → more class-faithful images (try 3.0, 4.0, 5.0).
    """
    unet.eval()
    B             = labels.size(0)
    uncond_labels = torch.full_like(labels, NULL_CLASS_IDX)

    x = torch.randn(B, 3, image_size, image_size, device=device)

    for t_step in reversed(range(diffusion.n_steps)):
        t = torch.full((B,), t_step, dtype=torch.long, device=device)

        cond_noise   = unet(x, t, labels)
        uncond_noise = unet(x, t, uncond_labels)

        # CFG formula: steer prediction away from unconditional toward conditional
        pred_noise = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

        x = diffusion._ddpm_step(x, t_step, pred_noise)

    return x


# %%
# ── Instantiate ───────────────────────────────────────────────────────────
unet      = UNet(NUM_CLASSES, TIME_EMB_DIM).to(device)
diffusion = DiffusionModel(unet, T_STEPS, device).to(device)

print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

optimizer = optim.Adam(unet.parameters(), lr=LR)


# %%
losses = []

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    for i, (data, labels) in enumerate(train_loader):
        x0  = data.to(device)
        lbl = labels.to(device)

        # ── CFG label dropout: 10% of labels → null class ─────────────────
        mask = torch.rand(lbl.size(0), device=device) < CFG_DROPOUT_RATE
        lbl  = lbl.masked_fill(mask, NULL_CLASS_IDX)

        # 1. Sample a random timestep t for each image in the batch
        t = torch.randint(0, T_STEPS, (x0.size(0),), device=device)

        # 2. Forward process: add noise  (mirrors DIFFUSION.py train() step 1)
        xt, noise = diffusion.forward_process(x0, t)

        # 3. Predict noise via UNet  (mirrors DIFFUSION.py train() step 2)
        pred_noise = unet(xt, t, lbl)

        # 4. DDPM loss: MSE on noise (replaces KL used in DIFFUSION.py for 2-D data)
        loss = F.mse_loss(pred_noise, noise)

        # 5. Backprop and optimiser step  (mirrors DIFFUSION.py train() step 3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 50 == 0:
            print(f"[{epoch+1}/{NUM_EPOCHS}][{i}/{len(train_loader)}]  Loss: {loss.item():.4f}")

    losses.append(epoch_loss / len(train_loader))

    # ── Visualise generated samples every SAVE_EVERY epochs ──────────────
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
# ── Plot loss curve  (mirrors DIFFUSION.py train() loss tracking) ─────────
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("DDPM Training Curve")
plt.tight_layout()
plt.show()

# %%
# ── Generate images per class to balance the dataset ──────────────────────
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
                                 guidance_scale=CFG_GUIDANCE_SCALE, device=device).cpu()  # (B,3,64,64) in [-1,1]
            all_gen.append(gen)
            remaining -= batch_n

        gen_imgs = torch.cat(all_gen, dim=0)
        gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)   # → [0,1]

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

# %% [markdown]
# ## Evaluation of Generative Quality
#
# FID and IS computed globally (all classes) — same protocol as cVAE, GAN, and WGAN-GP notebooks.
# SSIM is omitted (outputs are not reconstruction-paired with real images).

# %%
def collect_tensor(loader):
    batches = [imgs for imgs, _ in loader]
    return torch.cat(batches, dim=0)


print("Collecting real images…")
real_imgs = collect_tensor(train_loader)

print("Collecting diffusion-generated images…")
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

print("\n=== DDPM Generative Quality Summary ===")
print(f"  FID : {fid_score:.2f}")
print(f"  IS  : {is_mean:.3f} ± {is_std:.3f}")
