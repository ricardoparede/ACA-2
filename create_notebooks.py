"""
create_notebooks.py
Generates the three project notebooks:
  notebooks/02_autoencoder.ipynb
  notebooks/03_gan.ipynb
  notebooks/04_augmented_training.ipynb

Run from the project root:
    python create_notebooks.py
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def lines(code: str) -> list:
    """Split a multiline string into the Jupyter source-line list format."""
    return code.splitlines(keepends=True)


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(code),
    }


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def save_notebook(nb: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 02_autoencoder.ipynb  —  Conditional VAE
# ═════════════════════════════════════════════════════════════════════════════

CVAE_CELLS = [

md_cell("""\
# Conditional Variational AutoEncoder (cVAE)
### Butterfly Dataset — 75-class conditional image generation

Follows the AUTOENCODER.py example pattern from `exemplos_modelos/`:
- Class-based model definition (Encoder / Decoder / cVAE)
- `train()` function that returns model + loss history
- Visualisation helpers mirroring `plot_results()`

Extensions over the base example:
- **Convolutional** encoder/decoder (instead of linear)
- **Class conditioning** via `nn.Embedding` (required for 75-class generation)
- **ELBO loss** = MSE + λ·Perceptual + β·KL (β annealed to avoid posterior collapse)
- **GPU training** (cuda → mps → cpu fallback, same as all example files)
"""),

code_cell("""\
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# ── Device setup (following exemplos_modelos pattern) ──────────────────────
device = torch.device(
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")
"""),

code_cell("""\
# ── Add project root to path so src/ is importable ────────────────────────
sys.path.insert(0, os.path.abspath(".."))

from src.dataset import (
    ButterflyDataset, get_splits, build_label_map,
    get_train_transform, get_eval_transform, make_dataloader,
)
from src.utils import (
    save_checkpoint, load_checkpoint,
    visualize_grid, plot_losses,
    EarlyStopping, get_device, tensor_to_pil, denormalize,
)
from src.metrics import compute_fid, compute_inception_score, compute_ssim
"""),

code_cell("""\
# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath("..")
IMG_DIR    = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH   = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
SAVE_DIR   = os.path.join(BASE_DIR, "data", "generated", "cvae")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "cvae_checkpoint.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
IMAGE_SIZE    = 64       # matches TP2-students.ipynb IMAGE_SIZE
BATCH_SIZE    = 32       # matches TP2-students.ipynb BATCH_SIZE
LATENT_DIM    = 128
NUM_CLASSES   = 75
EMB_DIM       = 64

# Training
LR            = 1e-3     # from AUTOENCODER.py
WEIGHT_DECAY  = 1e-5     # from AUTOENCODER.py
NUM_EPOCHS    = 100
ES_PATIENCE   = 10

# Loss weights
BETA_MAX      = 1.0      # max KL weight
ANNEAL_EPOCHS = 10       # ramp beta from 0 → BETA_MAX over this many epochs
LAMBDA_PERC   = 0.1      # perceptual loss weight
"""),

code_cell("""\
# ── Data loading ───────────────────────────────────────────────────────────
label_to_idx, idx_to_label = build_label_map(CSV_PATH)
num_classes = len(label_to_idx)
print(f"Classes: {num_classes}")

train_df, val_df, test_df = get_splits(CSV_PATH, train_ratio=0.70, val_ratio=0.15, seed=42)
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

train_set = ButterflyDataset(train_df, IMG_DIR, label_to_idx, transform=get_train_transform(IMAGE_SIZE))
val_set   = ButterflyDataset(val_df,   IMG_DIR, label_to_idx, transform=get_eval_transform(IMAGE_SIZE))
test_set  = ButterflyDataset(test_df,  IMG_DIR, label_to_idx, transform=get_eval_transform(IMAGE_SIZE))

train_loader = make_dataloader(train_set, BATCH_SIZE, shuffle=True)
val_loader   = make_dataloader(val_set,   BATCH_SIZE, shuffle=False)
test_loader  = make_dataloader(test_set,  BATCH_SIZE, shuffle=False)

print(f"Batches — Train: {len(train_loader)}  Val: {len(val_loader)}")
"""),

code_cell("""\
# ── Visualise a training batch ─────────────────────────────────────────────
imgs, labs = next(iter(train_loader))
visualize_grid(imgs, labs, idx_to_label, n=16, nrow=8,
               title="Training samples (random batch)", denorm=True)
"""),

md_cell("""\
## Model Architecture

### Encoder
`Image (3×64×64)` + `class embedding (64-dim)`
→ Conv blocks (3→64→128→256, stride-2) → Flatten → FC(16448, 512) → **μ, log σ²** (128-dim each)

### Decoder
`z (128-dim)` + `class embedding (64-dim)`
→ FC(192, 16384) → Reshape (256×8×8) → ConvTranspose blocks (256→128→64→3) → **Tanh output**

Conditioning is applied via `nn.Embedding(75, 64)` injected at both the encoder (after flattening)
and the decoder (before the FC projection), following the approach from the course examples.
"""),

code_cell("""\
class Encoder(nn.Module):
    \"\"\"Convolutional encoder with class conditioning.\"\"\"

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, emb_dim=EMB_DIM):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_dim)

        # Conv feature extractor — mirrors Discriminator pattern in GAN.py
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),           # 64→32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # 32→16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),         # 16→8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = 256 * 8 * 8                              # 16384

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim + emb_dim, 512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x, label):
        h = self.conv(x).view(x.size(0), -1)                    # (B, 16384)
        c = self.class_emb(label)                                # (B, emb_dim)
        h = self.fc(torch.cat([h, c], dim=1))                   # (B, 512)
        return self.fc_mu(h), self.fc_logvar(h)
"""),

code_cell("""\
class Decoder(nn.Module):
    \"\"\"Convolutional decoder with class conditioning.\"\"\"

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, emb_dim=EMB_DIM):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.fc = nn.Linear(latent_dim + emb_dim, 256 * 8 * 8)

        # ConvTranspose upsampler — mirrors Generator pattern in GAN.py
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8→16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16→32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),     # 32→64
            nn.Tanh(),
        )

    def forward(self, z, label):
        c = self.class_emb(label)                                # (B, emb_dim)
        h = self.fc(torch.cat([z, c], dim=1))                   # (B, 256*8*8)
        h = h.view(-1, 256, 8, 8)                               # (B, 256, 8, 8)
        return self.deconv(h)                                    # (B, 3, 64, 64)
"""),

code_cell("""\
class ConditionalVAE(nn.Module):
    \"\"\"cVAE: combines Encoder + reparameterisation trick + Decoder.\"\"\"

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, emb_dim=EMB_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes, emb_dim)
        self.decoder = Decoder(latent_dim, num_classes, emb_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        \"\"\"z = mu + eps * sigma,  eps ~ N(0, I)\"\"\"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label):
        mu, logvar = self.encoder(x, label)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, label)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        \"\"\"Sample from the prior N(0, I) and decode for given class labels.\"\"\"
        z = torch.randn(len(labels), self.latent_dim, device=labels.device)
        return self.decoder(z, labels)
"""),

code_cell("""\
class PerceptualLoss(nn.Module):
    \"\"\"
    Feature-level loss using VGG16 up to relu2_2.
    Input tensors must be in [-1, 1] (will be rescaled internally).
    Mitigates the blurriness that plain MSE reconstruction causes in VAEs.
    \"\"\"

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # Layers 0-8 correspond to relu2_2
        self.slice = nn.Sequential(*list(vgg.children())[:9])
        for p in self.slice.parameters():
            p.requires_grad = False
        # ImageNet normalisation buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, target):
        # [-1,1] → [0,1] → ImageNet normalised
        x      = ((x      + 1) / 2 - self.mean) / self.std
        target = ((target + 1) / 2 - self.mean) / self.std
        return F.mse_loss(self.slice(x), self.slice(target))
"""),

code_cell("""\
def vae_loss(recon, x, mu, logvar, perceptual_fn, beta=1.0, lambda_perc=LAMBDA_PERC):
    \"\"\"
    ELBO loss for the cVAE.
    Extends the MSELoss approach from AUTOENCODER.py with:
      - Perceptual loss term  (reduces blur)
      - KL divergence term    (regularises latent space)
    \"\"\"
    mse  = F.mse_loss(recon, x, reduction="mean")
    perc = perceptual_fn(recon, x)
    kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = mse + lambda_perc * perc + beta * kl
    return total, mse.item(), perc.item(), kl.item()
"""),

code_cell("""\
def train_cvae(model, train_loader, val_loader, perceptual_fn,
               num_epochs=NUM_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
               es_patience=ES_PATIENCE, device=device):
    \"\"\"
    Training loop for the cVAE.
    Follows the train() function structure from exemplos_modelos/AUTOENCODER.py:
      - Adam optimiser (same lr and weight_decay)
      - Per-epoch loss printing
      - Returns (model, train_losses, val_losses)
    Extensions: validation loop, early stopping, beta annealing, GPU support.
    \"\"\"
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    es = EarlyStopping(patience=es_patience, mode="min")

    # Move everything to device (GPU if available)
    model = model.to(device)
    perceptual_fn = perceptual_fn.to(device)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(num_epochs):
        # Beta annealing: ramp KL weight from 0 → BETA_MAX
        beta = min(BETA_MAX, BETA_MAX * epoch / max(ANNEAL_EPOCHS, 1))

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs   = imgs.to(device)          # move data to GPU
            labels = labels.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(imgs, labels)
            loss, mse, perc, kl = vae_loss(recon, imgs, mu, logvar,
                                            perceptual_fn, beta=beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 50 == 0:
                print(f"[{epoch+1}/{num_epochs}][{i}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}  MSE: {mse:.4f}  "
                      f"Perc: {perc:.4f}  KL: {kl:.4f}  beta: {beta:.3f}")

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                recon, mu, logvar = model(imgs, labels)
                loss, *_ = vae_loss(recon, imgs, mu, logvar,
                                     perceptual_fn, beta=beta)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch: {epoch+1}  Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        # Save best checkpoint
        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(model, MODEL_PATH, epoch=epoch + 1, val_loss=avg_val)

        # Early stopping
        if es(avg_val):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return model.to("cpu"), train_losses, val_losses
"""),

code_cell("""\
# ── Instantiate and train (mirrors AUTOENCODER.py usage pattern) ───────────
cvae = ConditionalVAE(LATENT_DIM, NUM_CLASSES, EMB_DIM)
perceptual_fn = PerceptualLoss()

print(f"cVAE parameters: {sum(p.numel() for p in cvae.parameters()):,}")

cvae, train_losses, val_losses = train_cvae(
    cvae, train_loader, val_loader, perceptual_fn,
    num_epochs=NUM_EPOCHS, device=device,
)
"""),

code_cell("""\
# ── Plot training curves (mirrors plot_results() in AUTOENCODER.py) ────────
plot_losses(train_losses, val_losses, title="cVAE Training Curve — ELBO Loss")
"""),

code_cell("""\
# ── Visualise reconstructions (side-by-side: original vs reconstruction) ───
# Follows the two-row grid style of AUTOENCODER.py plot_results()
load_checkpoint(MODEL_PATH, cvae, device)
cvae = cvae.to(device)
cvae.eval()

imgs, labels = next(iter(val_loader))
imgs, labels = imgs.to(device), labels.to(device)
with torch.no_grad():
    recon, _, _ = cvae(imgs, labels)

n = 8
fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
for i in range(n):
    for row, t in enumerate([imgs, recon]):
        img = (t[i].cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[row, i].imshow(img)
        axes[row, i].axis("off")
axes[0, 0].set_title("Original", fontsize=9)
axes[1, 0].set_title("Reconstruction", fontsize=9)
plt.tight_layout()
plt.show()
"""),

code_cell("""\
# ── Generate images per class (balance dataset to largest class count) ─────
class_counts = Counter(train_df["label"].tolist())
TARGET_PER_CLASS = max(class_counts.values())
print(f"Max images in one class: {TARGET_PER_CLASS}")
print(f"Will generate up to {TARGET_PER_CLASS} images per under-represented class")

cvae.eval()
generated_records = []

with torch.no_grad():
    for class_name, class_idx in label_to_idx.items():
        existing = class_counts.get(class_name, 0)
        n_gen = max(0, TARGET_PER_CLASS - existing)
        if n_gen == 0:
            continue

        # Generate in mini-batches to avoid OOM on large n_gen
        all_gen = []
        remaining = n_gen
        while remaining > 0:
            batch_n = min(remaining, BATCH_SIZE)
            lbl = torch.full((batch_n,), class_idx, dtype=torch.long, device=device)
            gen = cvae.generate(lbl)                # (batch_n, 3, 64, 64)
            all_gen.append(gen.cpu())
            remaining -= batch_n
        gen_imgs = torch.cat(all_gen, dim=0)        # (n_gen, 3, 64, 64)
        gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]

        for j, img_t in enumerate(gen_imgs):
            fname = f"cvae_{class_idx:03d}_{j:04d}.jpg"
            tensor_to_pil(img_t).save(os.path.join(SAVE_DIR, fname))
            generated_records.append({
                "filename": fname,
                "label": class_name,
                "img_dir": SAVE_DIR,
            })

        print(f"  [{class_idx:02d}] {class_name}: generated {n_gen}")

gen_df = pd.DataFrame(generated_records)
gen_df.to_csv(os.path.join(SAVE_DIR, "generated.csv"), index=False)
print(f"\\nTotal generated: {len(gen_df):,}  |  Saved CSV → {SAVE_DIR}/generated.csv")
"""),

code_cell("""\
# ── Visualise a grid of generated samples (one per class, first 16) ────────
sample_labels = torch.arange(min(16, NUM_CLASSES), dtype=torch.long, device=device)
with torch.no_grad():
    samples = cvae.generate(sample_labels)
visualize_grid(samples.cpu(), sample_labels.cpu(), idx_to_label,
               n=16, nrow=8, title="cVAE — generated samples (first 16 classes)", denorm=True)
"""),

md_cell("""\
## Evaluation of Generative Quality

| Metric | Description | Applies to |
|--------|-------------|------------|
| **FID** (lower is better) | Fréchet distance between Inception features of real vs. generated sets | cVAE |
| **IS** (higher is better) | Inception Score — measures sharpness and diversity | cVAE |
| **SSIM** (higher is better) | Structural similarity vs. nearest real image of the same class | cVAE only |

FID and IS are computed **globally** (all classes combined) because per-class sample counts (~69)
are too small for reliable per-class FID estimation (requires ≥2000 samples).
"""),

code_cell("""\
from torch.utils.data import TensorDataset

def collect_tensor(loader, max_n=None):
    \"\"\"Collect all images from a DataLoader into one (N, C, H, W) tensor.\"\"\"
    batches = []
    n = 0
    for imgs, _ in loader:
        batches.append(imgs)
        n += len(imgs)
        if max_n and n >= max_n:
            break
    t = torch.cat(batches, dim=0)
    return t[:max_n] if max_n else t

print("Collecting real training images…")
real_imgs = collect_tensor(train_loader)          # (N, 3, 64, 64) in [-1,1]

print("Collecting generated images…")
gen_transform = get_eval_transform(IMAGE_SIZE)
gen_set  = ButterflyDataset(gen_df, SAVE_DIR, label_to_idx, transform=gen_transform)
gen_dl   = make_dataloader(gen_set, BATCH_SIZE, shuffle=False)
gen_imgs = collect_tensor(gen_dl)

print(f"Real: {real_imgs.shape}  |  Generated: {gen_imgs.shape}")
"""),

code_cell("""\
# ── FID ───────────────────────────────────────────────────────────────────
print("Computing FID…")
fid_score = compute_fid(real_imgs, gen_imgs, device=device)
print(f"FID: {fid_score:.2f}  (lower is better)")
"""),

code_cell("""\
# ── Inception Score ───────────────────────────────────────────────────────
print("Computing IS…")
is_mean, is_std = compute_inception_score(gen_imgs, device=device)
print(f"IS:  {is_mean:.3f} ± {is_std:.3f}  (higher is better)")
"""),

code_cell("""\
# ── SSIM — pair generated with real images of the same class ──────────────
# For SSIM we need matching pairs: for each generated image, its class' real images
from torch.utils.data import DataLoader as DL

# Build per-class lists of real image tensors
per_class_real = {lbl: [] for lbl in label_to_idx.values()}
for imgs, labs in train_loader:
    imgs_01 = (imgs * 0.5 + 0.5).clamp(0, 1)   # to [0,1]
    for img, lbl in zip(imgs_01, labs):
        per_class_real[lbl.item()].append(img)

# For each generated image, pick the first real image of same class as pair
paired_real, paired_gen = [], []
for imgs, labs in gen_dl:
    imgs_01 = (imgs * 0.5 + 0.5).clamp(0, 1)
    for img, lbl in zip(imgs_01, labs):
        c = lbl.item()
        if per_class_real[c]:
            paired_real.append(per_class_real[c][0])
            paired_gen.append(img)

real_t = torch.stack(paired_real)
gen_t  = torch.stack(paired_gen)
print("Computing SSIM…")
ssim_score = compute_ssim(real_t, gen_t)
print(f"SSIM: {ssim_score:.4f}  (higher is better, range [-1, 1])")

print("\\n=== cVAE Generative Quality Summary ===")
print(f"  FID  : {fid_score:.2f}")
print(f"  IS   : {is_mean:.3f} ± {is_std:.3f}")
print(f"  SSIM : {ssim_score:.4f}")
"""),

]


# ═════════════════════════════════════════════════════════════════════════════
# 03_gan.ipynb  —  Conditional DCGAN
# ═════════════════════════════════════════════════════════════════════════════

GAN_CELLS = [

md_cell("""\
# Conditional DCGAN (cGAN)
### Butterfly Dataset — 75-class conditional image generation

Follows the GAN.py example pattern from `exemplos_modelos/` very closely:
- Same variable names: `netG`, `netD`, `optimizerG`, `optimizerD`, `criterion`
- Same Adam settings: `lr=0.0002, betas=(0.5, 0.999)`
- Same training loop structure and per-batch print format
- Same `real_label / fake_label` convention (with label smoothing: real=0.9)

Extension over the example: **class conditioning** via `nn.Embedding` injected into
both Generator (noise concatenation) and Discriminator (spatial projection).
"""),

code_cell("""\
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# ── Device setup (following exemplos_modelos pattern) ──────────────────────
device = torch.device(
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")
"""),

code_cell("""\
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
"""),

code_cell("""\
# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath("..")
IMG_DIR     = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH    = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "data", "generated", "gan")
NETG_PATH   = os.path.join(BASE_DIR, "saved_models", "gan_generator.pth")
NETD_PATH   = os.path.join(BASE_DIR, "saved_models", "gan_discriminator.pth")
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters — mirroring GAN.py where applicable
IMAGE_SIZE  = 64        # matches TP2-students.ipynb
BATCH_SIZE  = 32        # matches TP2-students.ipynb
LATENT_DIM  = 100       # same as GAN.py noise dim
NUM_CLASSES = 75
EMB_DIM     = 100       # class embedding fed into G; projected to spatial in D

LR          = 0.0002    # from GAN.py
BETAS       = (0.5, 0.999)  # from GAN.py
NUM_EPOCHS  = 200
SAVE_EVERY  = 20        # save checkpoint every N epochs
real_label  = 0.9       # label smoothing (GAN.py uses 1; smoothing helps stability)
fake_label  = 0.0
"""),

code_cell("""\
label_to_idx, idx_to_label = build_label_map(CSV_PATH)
num_classes = len(label_to_idx)

train_df, val_df, test_df = get_splits(CSV_PATH, train_ratio=0.70, val_ratio=0.15, seed=42)
print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

train_set    = ButterflyDataset(train_df, IMG_DIR, label_to_idx,
                                transform=get_train_transform(IMAGE_SIZE))
train_loader = make_dataloader(train_set, BATCH_SIZE, shuffle=True)
print(f"Batches: {len(train_loader)}")
"""),

code_cell("""\
imgs, labs = next(iter(train_loader))
visualize_grid(imgs, labs, idx_to_label, n=16, nrow=8,
               title="Training samples", denorm=True)
"""),

md_cell("""\
## Model Architecture

### Generator (netG)
`noise (100-dim)` + `class embedding (100-dim)` → concat (200-dim) → reshape (200×1×1)
→ ConvTranspose blocks: 200→256→128→64→32→3 → **Tanh** → `64×64×3` image

### Discriminator (netD)
`image (3×64×64)` + class embedding projected to `(1×64×64)` spatial map
→ concat on channel dim → Conv blocks: 4→64→128→256→512→1 → **Sigmoid**

This design closely follows the `Generator` and `Discriminator` in `exemplos_modelos/GAN.py`,
extended with class conditioning as described in the conditional GAN literature.
"""),

code_cell("""\
def weights_init(m):
    \"\"\"
    Custom weight initialisation — standard for DCGAN.
    Matches the initialisation strategy implied by GAN.py (default Conv/BN init).
    \"\"\"
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
"""),

code_cell("""\
class Generator(nn.Module):
    \"\"\"
    Conditional Generator — extends GAN.py Generator with class conditioning.
    Input: noise (LATENT_DIM,) + class_idx → output: (3, 64, 64) image in [-1,1].
    \"\"\"

    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, emb_dim=EMB_DIM):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        # Input to first ConvTranspose: latent_dim + emb_dim = 200
        in_ch = latent_dim + emb_dim
        self.main = nn.Sequential(
            # Same ConvTranspose backbone as GAN.py Generator, adjusted for 64×64 output
            nn.ConvTranspose2d(in_ch, 256, 4, 1, 0, bias=False),  # → 4×4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # → 8×8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),     # → 16×16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),      # → 32×32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),       # → 64×64
            nn.Tanh(),
        )

    def forward(self, noise, label):
        c = self.class_emb(label).unsqueeze(-1).unsqueeze(-1)      # (B, emb, 1, 1)
        z = torch.cat([noise, c], dim=1)                           # (B, in_ch, 1, 1)
        return self.main(z)
"""),

code_cell("""\
class Discriminator(nn.Module):
    \"\"\"
    Conditional Discriminator — extends GAN.py Discriminator with class conditioning.
    Projects class embedding to a spatial (1×H×W) map, concatenated to image channels.
    \"\"\"

    def __init__(self, num_classes=NUM_CLASSES, emb_dim=EMB_DIM, image_size=IMAGE_SIZE):
        super().__init__()
        self.image_size = image_size
        # Project class label to a single spatial channel
        self.class_proj = nn.Sequential(
            nn.Embedding(num_classes, emb_dim),
            nn.Linear(emb_dim, image_size * image_size),
        )
        # Conv backbone mirrors GAN.py Discriminator, input = 3+1 = 4 channels
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),                # 64→32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),              # 32→16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # 16→8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # 8→4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),               # 4→1
            nn.Sigmoid(),
        )

    def forward(self, img, label):
        B = img.size(0)
        # Spatial class map: (B, 1, H, W)
        c = self.class_proj(label).view(B, 1, self.image_size, self.image_size)
        x = torch.cat([img, c], dim=1)                            # (B, 4, 64, 64)
        return self.model(x).view(B)
"""),

code_cell("""\
# ── Instantiate exactly as in GAN.py ──────────────────────────────────────
netG = Generator(LATENT_DIM, NUM_CLASSES, EMB_DIM).to(device)
netD = Discriminator(NUM_CLASSES, EMB_DIM, IMAGE_SIZE).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

print(f"Generator parameters    : {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

# Optimisers — identical to GAN.py
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=BETAS)
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=BETAS)

# Loss — identical to GAN.py
criterion = nn.BCELoss()
"""),

code_cell("""\
# ── Training loop — mirrors GAN.py structure closely ──────────────────────
# Variable names and comments intentionally match GAN.py for clarity.

G_losses, D_losses = [], []

for epoch in range(NUM_EPOCHS):
    epoch_G, epoch_D = 0.0, 0.0
    for i, (data, labels) in enumerate(train_loader):

        # ─── Update D: maximise log(D(x)) + log(1 - D(G(z))) ─────────────
        # (mirrors GAN.py comment style verbatim)
        netD.zero_grad()
        real_cpu  = data.to(device)           # real images moved to GPU
        lbl       = labels.to(device)
        batch_size = real_cpu.size(0)

        # Real image pass
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output    = netD(real_cpu, lbl)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Fake image pass
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake  = netG(noise, lbl)
        label.fill_(fake_label)
        output    = netD(fake.detach(), lbl)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1    = output.mean().item()
        errD      = errD_real + errD_fake
        optimizerD.step()

        # ─── Update G: maximise log(D(G(z))) ─────────────────────────────
        netG.zero_grad()
        label.fill_(real_label)               # fake labels are real for G cost
        output = netD(fake, lbl)
        errG   = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        epoch_G += errG.item()
        epoch_D += errD.item()

        if i % 50 == 0:
            print(f"[{epoch+1}/{NUM_EPOCHS}][{i}/{len(train_loader)}] "
                  f"Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}  "
                  f"D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

    G_losses.append(epoch_G / len(train_loader))
    D_losses.append(epoch_D / len(train_loader))

    # ── Save generated samples every SAVE_EVERY epochs (mirrors GAN.py) ──
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
"""),

code_cell("""\
# ── Plot Generator and Discriminator loss curves ──────────────────────────
plot_losses(G_losses, D_losses, ylabel="Loss",
            title="cGAN Training Curve")
plt.legend(["Generator", "Discriminator"])
plt.show()
"""),

code_cell("""\
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
            fname = f"gan_{class_idx:03d}_{j:04d}.jpg"
            tensor_to_pil(img_t).save(os.path.join(SAVE_DIR, fname))
            generated_records.append({
                "filename": fname,
                "label": class_name,
                "img_dir": SAVE_DIR,
            })

        print(f"  [{class_idx:02d}] {class_name}: generated {n_gen}")

gen_df = pd.DataFrame(generated_records)
gen_df.to_csv(os.path.join(SAVE_DIR, "generated.csv"), index=False)
print(f"\\nTotal generated: {len(gen_df):,}  |  CSV saved.")
"""),

md_cell("""\
## Evaluation of Generative Quality

FID and IS computed globally (all classes) — same protocol as the cVAE notebook.
SSIM is omitted for GANs (outputs are not reconstruction-paired with real images).
"""),

code_cell("""\
def collect_tensor(loader):
    batches = [imgs for imgs, _ in loader]
    return torch.cat(batches, dim=0)

print("Collecting real images…")
real_imgs = collect_tensor(train_loader)

print("Collecting GAN-generated images…")
gen_set  = ButterflyDataset(gen_df, SAVE_DIR, label_to_idx,
                             transform=get_eval_transform(IMAGE_SIZE))
gen_dl   = make_dataloader(gen_set, BATCH_SIZE, shuffle=False)
gen_imgs = collect_tensor(gen_dl)

print(f"Real: {real_imgs.shape}  |  Generated: {gen_imgs.shape}")
"""),

code_cell("""\
print("Computing FID…")
fid_score = compute_fid(real_imgs, gen_imgs, device=device)
print(f"FID: {fid_score:.2f}  (lower is better)")

print("Computing IS…")
is_mean, is_std = compute_inception_score(gen_imgs, device=device)
print(f"IS:  {is_mean:.3f} ± {is_std:.3f}  (higher is better)")

print("\\n=== cGAN Generative Quality Summary ===")
print(f"  FID : {fid_score:.2f}")
print(f"  IS  : {is_mean:.3f} ± {is_std:.3f}")
"""),

]


# ═════════════════════════════════════════════════════════════════════════════
# 04_augmented_training.ipynb  —  Retrain baseline with augmented data
# ═════════════════════════════════════════════════════════════════════════════

AUG_CELLS = [

md_cell("""\
# Augmented Training — Baseline CNN with Generated Data
### Compare: Baseline vs. +cVAE vs. +GAN

**Constraint from the enunciado (strictly enforced here):**
> *"You should only change the dataset and train the baseline model as in the script
> provided with the augmented dataset."*

- The `BaselineCNN` architecture is copied **exactly** from `TP2-students.ipynb` without any change.
- The training strategy (optimiser, loss, epochs, early stopping) is defined **once** and
  applied identically to all three runs.
- All three runs use the **same held-out test split** (produced by `get_splits(seed=42)`).
"""),

code_cell("""\
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# ── Device setup (following exemplos_modelos pattern) ──────────────────────
device = torch.device(
    "cuda"  if torch.cuda.is_available()  else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")
"""),

code_cell("""\
sys.path.insert(0, os.path.abspath(".."))

from src.dataset import (
    ButterflyDataset, AugmentedButterflyDataset,
    get_splits, build_label_map,
    get_baseline_transform, get_eval_transform, make_dataloader,
)
from src.utils import (
    save_checkpoint, load_checkpoint,
    plot_losses, EarlyStopping,
)
from src.metrics import classification_report_dict
"""),

code_cell("""\
# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath("..")
IMG_DIR   = os.path.join(BASE_DIR, "aca-butterflies", "train")
CSV_PATH  = os.path.join(BASE_DIR, "aca-butterflies", "train.csv")
CVAE_CSV  = os.path.join(BASE_DIR, "data", "generated", "cvae", "generated.csv")
GAN_CSV   = os.path.join(BASE_DIR, "data", "generated", "gan",  "generated.csv")
AUG_DIR   = os.path.join(BASE_DIR, "data", "augmented")
os.makedirs(AUG_DIR, exist_ok=True)

# ── Training hyperparameters (IDENTICAL for all three runs) ────────────────
IMAGE_SIZE   = 64      # from TP2-students.ipynb
BATCH_SIZE   = 32      # from TP2-students.ipynb
LR           = 0.001   # from TP2-students.ipynb: optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS   = 50      # applied equally to baseline and augmented runs
ES_PATIENCE  = 10      # early stopping patience (same for all runs)
"""),

code_cell("""\
label_to_idx, idx_to_label = build_label_map(CSV_PATH)
num_classes = len(label_to_idx)
print(f"Number of classes: {num_classes}")

# Stratified split — same seed as generative notebooks → same test set
train_df, val_df, test_df = get_splits(CSV_PATH, train_ratio=0.70, val_ratio=0.15, seed=42)
print(f"Original — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
"""),

md_cell("""\
## BaselineCNN
Copied **verbatim** from `TP2-students.ipynb` — no modifications allowed.
"""),

code_cell("""\
class BaselineCNN(nn.Module):
    \"\"\"
    Baseline CNN — copied exactly from TP2-students.ipynb.
    DO NOT modify architecture, optimizer, or loss function.
    \"\"\"

    def __init__(self, num_classes=75):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # 64×64 → 32×32
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # 32×32 → 16×16
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                        # 16×16 → 8×8
            nn.AdaptiveAvgPool2d((1, 1)),              # → 1×1
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
"""),

code_cell("""\
def train_classifier(model, train_loader, val_loader,
                     num_epochs=NUM_EPOCHS, lr=LR,
                     es_patience=ES_PATIENCE, device=device,
                     save_path=None, run_name=""):
    \"\"\"
    Identical training strategy for all three runs.
    Optimizer and loss match TP2-students.ipynb exactly:
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      criterion = nn.CrossEntropyLoss()
    Returns (model, train_losses, val_losses, val_accs).
    \"\"\"
    # Exactly as in TP2-students.ipynb
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(patience=es_patience, mode="min")

    model = model.to(device)   # move model to GPU
    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)      # move batch to GPU
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        avg_val  = val_loss / len(val_loader)
        acc      = correct / total
        val_losses.append(avg_val)
        val_accs.append(acc)

        print(f"[{run_name}] Epoch: {epoch+1:3d}  "
              f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}  Val Acc: {acc:.4f}")

        if acc > best_val_acc and save_path:
            best_val_acc = acc
            save_checkpoint(model, save_path, epoch=epoch+1, val_acc=acc)

        if es(avg_val):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model.to("cpu"), train_losses, val_losses, val_accs
"""),

code_cell("""\
def evaluate_classifier(model, test_loader, label_to_idx, idx_to_label, device=device):
    \"\"\"Evaluate on held-out test split. Returns metrics dict.\"\"\"
    model = model.to(device)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(labels.tolist())
    return classification_report_dict(all_true, all_preds, idx_to_label)
"""),

code_cell("""\
# ── Shared test loader (identical across all runs) ────────────────────────
test_set    = ButterflyDataset(test_df, IMG_DIR, label_to_idx,
                                transform=get_eval_transform(IMAGE_SIZE))
test_loader = make_dataloader(test_set, BATCH_SIZE, shuffle=False)
print(f"Test samples: {len(test_set):,}")
"""),

md_cell("## Run 1 — Baseline (original data only)"),

code_cell("""\
baseline_train_set = ButterflyDataset(train_df, IMG_DIR, label_to_idx,
                                       transform=get_baseline_transform(IMAGE_SIZE))
baseline_val_set   = ButterflyDataset(val_df,   IMG_DIR, label_to_idx,
                                       transform=get_eval_transform(IMAGE_SIZE))

baseline_train_loader = make_dataloader(baseline_train_set, BATCH_SIZE, shuffle=True)
baseline_val_loader   = make_dataloader(baseline_val_set,   BATCH_SIZE, shuffle=False)

print(f"Baseline — Train: {len(baseline_train_set):,}  Val: {len(baseline_val_set):,}")
"""),

code_cell("""\
baseline_model = BaselineCNN(num_classes=num_classes)

baseline_model, bl_train_l, bl_val_l, bl_val_acc = train_classifier(
    baseline_model,
    baseline_train_loader,
    baseline_val_loader,
    run_name="BASELINE",
    save_path=os.path.join(BASE_DIR, "saved_models", "baseline.pth"),
)

plot_losses(bl_train_l, bl_val_l, title="Baseline — Training Curve")
"""),

code_cell("""\
load_checkpoint(os.path.join(BASE_DIR, "saved_models", "baseline.pth"), baseline_model)
baseline_metrics = evaluate_classifier(baseline_model, test_loader, label_to_idx, idx_to_label)
print(f"Baseline  |  Accuracy: {baseline_metrics['accuracy']:.4f}"
      f"  Macro F1: {baseline_metrics['macro_f1']:.4f}")
"""),

md_cell("## Run 2 — Baseline + cVAE-Generated Data"),

code_cell("""\
cvae_gen_df = pd.read_csv(CVAE_CSV)
print(f"cVAE generated samples: {len(cvae_gen_df):,}")

# Build augmented train set: original images + cVAE generated images
# Original images use IMG_DIR; generated images use their own img_dir column
orig_train_df = train_df.copy()
orig_train_df["img_dir"] = IMG_DIR
combined_cvae_df = pd.concat([orig_train_df, cvae_gen_df], ignore_index=True)
combined_cvae_df.to_csv(os.path.join(AUG_DIR, "cvae_augmented.csv"), index=False)
print(f"Combined (original + cVAE): {len(combined_cvae_df):,}")

cvae_aug_train = AugmentedButterflyDataset(combined_cvae_df, label_to_idx,
                                            transform=get_baseline_transform(IMAGE_SIZE))
cvae_aug_val   = ButterflyDataset(val_df, IMG_DIR, label_to_idx,
                                   transform=get_eval_transform(IMAGE_SIZE))

cvae_train_loader = make_dataloader(cvae_aug_train, BATCH_SIZE, shuffle=True)
cvae_val_loader   = make_dataloader(cvae_aug_val,   BATCH_SIZE, shuffle=False)
print(f"Augmented train batches: {len(cvae_train_loader)}")
"""),

code_cell("""\
cvae_model = BaselineCNN(num_classes=num_classes)

cvae_model, cv_train_l, cv_val_l, cv_val_acc = train_classifier(
    cvae_model,
    cvae_train_loader,
    cvae_val_loader,
    run_name="+cVAE",
    save_path=os.path.join(BASE_DIR, "saved_models", "baseline_cvae.pth"),
)

plot_losses(cv_train_l, cv_val_l, title="+cVAE — Training Curve")
"""),

code_cell("""\
load_checkpoint(os.path.join(BASE_DIR, "saved_models", "baseline_cvae.pth"), cvae_model)
cvae_metrics = evaluate_classifier(cvae_model, test_loader, label_to_idx, idx_to_label)
print(f"+cVAE     |  Accuracy: {cvae_metrics['accuracy']:.4f}"
      f"  Macro F1: {cvae_metrics['macro_f1']:.4f}")
"""),

md_cell("## Run 3 — Baseline + GAN-Generated Data"),

code_cell("""\
gan_gen_df = pd.read_csv(GAN_CSV)
print(f"GAN generated samples: {len(gan_gen_df):,}")

orig_train_df = train_df.copy()
orig_train_df["img_dir"] = IMG_DIR
combined_gan_df = pd.concat([orig_train_df, gan_gen_df], ignore_index=True)
combined_gan_df.to_csv(os.path.join(AUG_DIR, "gan_augmented.csv"), index=False)
print(f"Combined (original + GAN): {len(combined_gan_df):,}")

gan_aug_train = AugmentedButterflyDataset(combined_gan_df, label_to_idx,
                                           transform=get_baseline_transform(IMAGE_SIZE))
gan_aug_val   = ButterflyDataset(val_df, IMG_DIR, label_to_idx,
                                  transform=get_eval_transform(IMAGE_SIZE))

gan_train_loader = make_dataloader(gan_aug_train, BATCH_SIZE, shuffle=True)
gan_val_loader   = make_dataloader(gan_aug_val,   BATCH_SIZE, shuffle=False)
print(f"Augmented train batches: {len(gan_train_loader)}")
"""),

code_cell("""\
gan_model = BaselineCNN(num_classes=num_classes)

gan_model, gn_train_l, gn_val_l, gn_val_acc = train_classifier(
    gan_model,
    gan_train_loader,
    gan_val_loader,
    run_name="+GAN",
    save_path=os.path.join(BASE_DIR, "saved_models", "baseline_gan.pth"),
)

plot_losses(gn_train_l, gn_val_l, title="+GAN — Training Curve")
"""),

code_cell("""\
load_checkpoint(os.path.join(BASE_DIR, "saved_models", "baseline_gan.pth"), gan_model)
gan_metrics = evaluate_classifier(gan_model, test_loader, label_to_idx, idx_to_label)
print(f"+GAN      |  Accuracy: {gan_metrics['accuracy']:.4f}"
      f"  Macro F1: {gan_metrics['macro_f1']:.4f}")
"""),

md_cell("## Results Summary"),

code_cell("""\
# ── Training curve overlay ─────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(bl_val_acc, label="Baseline")
plt.plot(cv_val_acc, label="+cVAE")
plt.plot(gn_val_acc, label="+GAN")
plt.xlabel("Epoch"); plt.ylabel("Val Accuracy"); plt.title("Validation Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(bl_val_l, label="Baseline")
plt.plot(cv_val_l, label="+cVAE")
plt.plot(gn_val_l, label="+GAN")
plt.xlabel("Epoch"); plt.ylabel("Val Loss"); plt.title("Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()
"""),

code_cell("""\
# ── Summary table ──────────────────────────────────────────────────────────
results = {
    "Baseline":   baseline_metrics,
    "+cVAE":      cvae_metrics,
    "+GAN":       gan_metrics,
}

print(f"{'Model':<12} {'Top-1 Acc':>10} {'Macro F1':>10}")
print("-" * 34)
for name, m in results.items():
    print(f"{name:<12} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")
"""),

code_cell("""\
# ── Per-class accuracy comparison ─────────────────────────────────────────
# Find classes where augmentation helped most
bl_pc  = baseline_metrics["per_class_acc"]
cv_pc  = cvae_metrics["per_class_acc"]
gn_pc  = gan_metrics["per_class_acc"]

deltas = {
    cls: {
        "cvae_delta": cv_pc.get(cls, 0) - bl_pc.get(cls, 0),
        "gan_delta":  gn_pc.get(cls, 0) - bl_pc.get(cls, 0),
    }
    for cls in bl_pc
}

# Top 10 classes most improved by cVAE
top_cvae = sorted(deltas.items(), key=lambda x: x[1]["cvae_delta"], reverse=True)[:10]
print("\\nTop-10 classes most improved by cVAE:")
for cls, d in top_cvae:
    print(f"  {cls:<35} Δ={d['cvae_delta']:+.4f}  (baseline={bl_pc[cls]:.4f})")

# Top 10 classes most improved by GAN
top_gan = sorted(deltas.items(), key=lambda x: x[1]["gan_delta"], reverse=True)[:10]
print("\\nTop-10 classes most improved by GAN:")
for cls, d in top_gan:
    print(f"  {cls:<35} Δ={d['gan_delta']:+.4f}  (baseline={bl_pc[cls]:.4f})")
"""),

]


# ═════════════════════════════════════════════════════════════════════════════
# Write notebooks
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    notebooks = [
        ("02_autoencoder.ipynb", CVAE_CELLS),
        ("03_gan.ipynb",         GAN_CELLS),
        ("04_augmented_training.ipynb", AUG_CELLS),
    ]
    for fname, cells in notebooks:
        path = os.path.join(NB_DIR, fname)
        save_notebook(notebook(cells), path)
    print("\nAll notebooks created successfully.")
