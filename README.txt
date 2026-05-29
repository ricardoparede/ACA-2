# TP2 — "We Need More Butterflies!"
**Course:** Advanced Concepts in AI (ACA) — University of Coimbra, 2025/26  
**Deadline:** June 1 2026  
**Goal:** Use generative models (cVAE + cGAN) to augment a 75-class butterfly image dataset,
then retrain the provided baseline CNN and measure whether augmentation improves accuracy.

---

## Project Structure

```
TP2-Students/
├── aca-butterflies/
│   ├── train.csv               # filename, label columns (5199 images total)
│   └── train/                  # original images
├── data/
│   ├── generated/
│   │   ├── cvae/               # cVAE-generated images + generated.csv
│   │   ├── gan/                # cGAN-generated images + generated.csv (not yet run)
│   │   └── diffusion/          # placeholder (not used)
│   └── augmented/              # combined CSVs written by notebook 04
├── saved_models/
│   ├── cvae_checkpoint.pth     # best cVAE (monitored on val_recon)
│   ├── baseline.pth            # best baseline classifier
│   └── baseline_cvae.pth       # best +cVAE classifier
├── notebooks/
│   ├── 02_autoencoder.ipynb    # cVAE: train + generate + FID/IS/SSIM
│   ├── 03_gan.ipynb            # cGAN: train + generate + evaluate
│   └── 04_augmented_training.ipynb  # classifier: baseline vs +cVAE vs +GAN
├── src/
│   ├── dataset.py
│   ├── utils.py
│   ├── metrics.py
│   └── __init__.py
├── exemplos_modelos/           # course-provided reference implementations
├── TP2-students.ipynb          # original baseline notebook (DO NOT MODIFY)
├── requirements.txt
└── README.md
```

---

## Environment

- **Python 3.12**, PyTorch 2.11.0+cu128, CUDA 12.8, RTX 5060 Laptop 8GB
- venv at `.venv\`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Device pattern in all notebooks:
```python
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")
```

---

## Dataset

- 5,199 images, 75 classes
- Split: 70/15/15 stratified, `seed=42` — **identical across all notebooks**
  - Train ~3,606 | Val ~745 | Test ~848
- `build_label_map(CSV_PATH)` → sorted alphabetically
- Transforms normalize to [-1, 1] (mean=std=0.5)

---

## src/ API

### `src/dataset.py`
| Symbol | Description |
|--------|-------------|
| `ButterflyDataset(df, img_dir, label_to_idx, transform)` | df has `filename`, `label` |
| `AugmentedButterflyDataset(df, label_to_idx, transform)` | df must have `img_dir` column — used when mixing real + generated from different dirs |
| `get_splits(csv_path, train_ratio, val_ratio, seed)` | Stratified; returns `(train_df, val_df, test_df)` |
| `get_train_transform(sz)` | Augmented (flip, jitter, rotate) |
| `get_baseline_transform(sz)` | Lighter augmentation (classifier training) |
| `get_eval_transform(sz)` | Deterministic resize+normalize only |
| `make_dataloader(dataset, batch_size, shuffle)` | Thin DataLoader wrapper |

### `src/utils.py`
| Symbol | Description |
|--------|-------------|
| `save_checkpoint(model, path, **extra)` | state dict + metadata |
| `load_checkpoint(path, model, device)` | loads inplace; returns payload dict |
| `EarlyStopping(patience, mode)` | `mode="min"` or `"max"`; `es(metric)` → True to stop |
| `plot_losses(train_l, val_l, title)` | training curve |
| `visualize_grid(imgs, labels, idx_to_label, n, nrow, title, denorm)` | image grid |
| `tensor_to_pil(tensor)` | (3,H,W) in [0,1] → PIL |
| `denormalize(tensor)` | [-1,1] → [0,1] |

### `src/metrics.py`
| Symbol | Description |
|--------|-------------|
| `compute_fid(real_imgs, gen_imgs, device)` | InceptionV3 FID; inputs in [-1,1] |
| `compute_inception_score(gen_imgs, device, splits)` | returns `(mean, std)`; inputs in [-1,1] |
| `compute_ssim(real_t, gen_t)` | pairwise SSIM on (N,3,H,W) in [0,1] |
| `classification_report_dict(y_true, y_pred, idx_to_label)` | dict: `accuracy`, `macro_f1`, `per_class_acc` |

---

## Notebook 02 — cVAE ✅ COMPLETE

**Architecture:**
- Encoder: Conv(3→64→128→256, stride-2) → FC(16384+32, 512) → μ, logσ² (128-dim)
- Decoder: FC(128+32, 256×8×8) → ConvTranspose(256→128→64→3) + **InstanceNorm2d(affine=True)**
- Conditioning: `nn.Embedding(75, 32)` injected at encoder (after flatten) and decoder (before FC)

**Key hyperparameters:**
```python
IMAGE_SIZE=64, BATCH_SIZE=32, LATENT_DIM=128, EMB_DIM=32
LR=3e-4, WEIGHT_DECAY=1e-5, NUM_EPOCHS=150, ES_PATIENCE=30
LR_PATIENCE=10, LR_FACTOR=0.5, LR_MIN=1e-6
BETA=0.5, LAMBDA_PERC=0.001, GRAD_CLIP=1.0, FREE_BITS=0.2
```

**Loss:** `total = (MSE + 0.001·VGG_perc) + 0.5·KL_free_bits`  
ES + checkpoint use `val_recon` (no KL term) — more stable signal than ELBO.

**Generation:** fills each class up to max class count → `data/generated/cvae/generated.csv`  
`generated.csv` columns: `filename`, `label`, `img_dir` (absolute path)

**Metrics:** FID≈276, IS≈1.68±0.10, SSIM≈0.14  
FID >150 → augmentation expected to hurt classifier (label noise dominates).

---

## Notebook 03 — cGAN ❌ NOT YET RUN

---

## Notebook 04 — Augmented Training ⚠️ IN PROGRESS

**Constraint (enunciado — STRICTLY ENFORCED):** Only the dataset changes. `BaselineCNN` architecture, optimizer, and loss are copied verbatim from `TP2-students.ipynb`.

**BaselineCNN:** 3 conv blocks (3→64→128→256, double conv each) + AdaptiveAvgPool + FC(256→512→1024→2048→75) + Dropout(0.5)

**Training config (all three runs):**
```python
IMAGE_SIZE=64, BATCH_SIZE=32, LR=0.001, NUM_EPOCHS=100, ES_PATIENCE=15
LR_PATIENCE=7, LR_FACTOR=0.5, LR_MIN=1e-5
optimizer = Adam(lr=0.001); criterion = CrossEntropyLoss()
scheduler = ReduceLROnPlateau(mode="max", patience=7, factor=0.5, min_lr=1e-5)
# ES monitors val_acc (mode="max") — consistent with checkpoint
```

**Building augmented dataset:**
```python
orig_train_df["img_dir"] = IMG_DIR
combined = pd.concat([orig_train_df, gen_df])  # gen_df already has img_dir column
AugmentedButterflyDataset(combined, label_to_idx, transform=get_baseline_transform(sz))
```

**Current results:**

| Run | Status | Notes |
|-----|--------|-------|
| Baseline | run (needs re-run) | old ES bug; ~0.20 val acc |
| +cVAE | run (needs re-run) | old ES bug; worse than baseline expected (FID≈276) |
| +GAN | blocked | needs notebook 03 first |

---

## Known Issues & Fixes

| Issue | Fix applied |
|-------|-------------|
| ES fired too early on val_loss | `mode="max"`, `es(acc)` |
| Val loss oscillation | `ReduceLROnPlateau` on val_acc |
| cVAE NaN | `LAMBDA_PERC` 0.1→0.001 + logvar clamping[-4,15] + grad_clip=1.0 |
| KL collapse | `FREE_BITS=0.2`, removed beta annealing (fixed BETA=0.5) |
| Decoder blurriness | `BatchNorm2d` → `InstanceNorm2d(affine=True)` |
| Perceptual loss dominating gradients | `LAMBDA_PERC` 0.05→0.001 |

---

## Potential cVAE Improvements (not implemented)

1. **AdaIN conditioning** — inject class embedding at every decoder upsampling layer (~50–80 FID pts)
2. **LPIPS** (`pip install lpips`) — better perceptual loss than VGG relu2_2
3. **Larger latent** (LATENT_DIM=256) + 4th encoder conv block
4. **Spectral normalization** on encoder convolutions
5. **Self-attention** at the 8×8 spatial bottleneck

---

## Useful Commands

```powershell
# Run notebook headlessly
jupyter nbconvert --to notebook --execute --inplace notebooks/02_autoencoder.ipynb

# Sync jupytext pairs (pip install jupytext first)
jupytext --set-formats ipynb,py:percent notebooks/08_quality_filtering.py
jupytext --sync TP2-students.ipynb notebooks/02_autoencoder.ipynb notebooks/03_gan.ipynb notebooks/04_wgan_gp.ipynb notebooks/05_biggan.ipynb notebooks/06_diffusion.ipynb notebooks/07_augmented_training.ipynb notebooks/08_quality_filtering.py

# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```