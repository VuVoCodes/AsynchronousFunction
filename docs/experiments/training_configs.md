# Training Configuration Reference

**Last updated:** 2026-04-18
**Purpose:** Complete record of all training configs used across 9 datasets and 10+ methods for reproducibility and auditability.

---

## Table of Contents
1. [Datasets and Architectures](#datasets-and-architectures)
2. [Training Hyperparameters per Dataset](#training-hyperparameters-per-dataset)
3. [Method-Specific Hyperparameters](#method-specific-hyperparameters)
4. [Reproduction Commands](#reproduction-commands)

---

## Datasets and Architectures

| Dataset | Modalities | Input Type | Encoder(s) | Pretrained? | Params |
|---------|-----------|-----------|-----------|-------------|--------|
| **CREMA-D** (3 frames @ 3fps) | audio + visual | Raw spectrograms + video frames | 2× ResNet18 | **From scratch** | ~23M |
| **AVE** | audio + visual | Raw spectrograms + video frames | 2× ResNet18 | ImageNet | ~23M |
| **Kinetics-Sounds (KS)** | audio + visual | Raw spectrograms + video frames | 2× ResNet18 | ImageNet | ~23M |
| **CMU-MOSEI** | text + audio + vision | Pre-extracted BERT/COVAREP/FACET | 3× MLP (2-layer) | — | ~1.5M |
| **CMU-MOSI** | text + audio + vision | Pre-extracted GloVe/COVAREP/FACET | 3× MLP (2-layer) | — | ~0.5M |
| **Sarcasm (MMSD v1)** | text + image | Pre-extracted BERT + ResNet18 | 2× MLP (2-layer) | Features frozen | ~1.7M |
| **Twitter15** | text + image | Pre-extracted BERT + ResNet18 | 2× MLP (2-layer) | Features frozen | ~1.7M |
| **UPMC-Food101** | text + image | Pre-extracted BERT + ResNet18 | 2× MLP (2-layer) | Features frozen | ~1.9M |
| **BraTS 2021** | 4× MRI (FLAIR/T1ce/T1/T2) | 3D volumes (slices) | DeepLab v3+ × 4 ResNet101 | ImageNet | ~235M |

**Key distinction:**
- **Representation-learning datasets**: CREMA-D (from scratch), AVE/KS/BraTS (pretrained but fine-tuned) → gradient modulation shapes feature learning
- **Frozen-feature datasets**: MOSEI, MOSI, Sarcasm, Twitter15, Food101 → gradient modulation only affects classifier head

---

## Training Hyperparameters per Dataset

### Audio-Visual Classification (CREMA-D, AVE, KS)

| Parameter | CREMA-D 3f | AVE | KS |
|-----------|-----------|-----|-----|
| Backbone | ResNet18 (scratch) | ResNet18 (pretrained) | ResNet18 (pretrained) |
| Feature dim | 512 | 512 | 512 |
| Fusion | concat (1024→512) | concat | concat |
| Optimizer | SGD | SGD | SGD |
| Initial LR | 1e-3 | 1e-3 | 1e-3 |
| Momentum | 0.9 | 0.9 | 0.9 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Scheduler | StepLR step=70, γ=0.1 | StepLR step=40, γ=0.1 | StepLR step=40, γ=0.1 |
| Batch size | 64 | 64 | 64 |
| Epochs | 100 | 100 | 100 |
| Visual frames | 3 @ 3 fps | 3 | 3 |

### Sentiment Analysis (MOSEI, MOSI)

| Parameter | MOSEI | MOSI |
|-----------|-------|------|
| Backbone | MLP (2-layer, 512 hidden, 0.3 dropout) | Same |
| Text encoder | BERT 768d → MLP | GloVe 300d → MLP |
| Audio encoder | COVAREP 33d → MLP | COVAREP 74d → MLP |
| Visual encoder | FACET 709d → MLP | FACET 35d → MLP |
| Optimizer | Adam | Adam |
| Initial LR | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | StepLR step=40, γ=0.1 | StepLR step=40, γ=0.1 |
| Batch size | 64 | 64 |
| Epochs | 100 | 100 |
| Features source | MMSA pickle | MMSA pickle |
| Label type | 3-class sentiment | 2-class sentiment |

### Text+Image Classification (Sarcasm, Twitter, Food101) — **Frozen Features**

| Parameter | Sarcasm | Twitter15 | Food101 |
|-----------|---------|-----------|---------|
| Text encoder (frozen) | BERT-base-uncased 768d | BERT-base-uncased 768d | BERT-base-uncased 768d |
| Image encoder (frozen) | ResNet18 ImageNet 512d | ResNet18 ImageNet 512d | ResNet18 ImageNet 512d |
| Model | 2× MLP (2-layer, 512 hidden, 0.3 dropout) + concat | Same | Same |
| Optimizer | Adam | Adam | Adam |
| Initial LR | 1e-3 | 1e-3 | 1e-3 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Scheduler | StepLR step=40, γ=0.1 | StepLR step=40, γ=0.1 | StepLR step=40, γ=0.1 |
| Batch size | 64 | 64 | 64 |
| Epochs | 100 | 100 | 100 |
| Classes | 2 | 3 | 101 |
| Train/Val/Test | 19,816 / 2,410 / 2,409 | 3,179 / 1,122 / 1,037 | 67,972 / — / 22,716 |

### Segmentation (BraTS 2021)

| Parameter | Value |
|-----------|-------|
| Architecture | DeepLab v3+ with 4× ResNet101 encoders + shared decoder |
| Params | ~235M |
| Optimizer | SGD |
| Base LR | 0.01 (cosine schedule with 5-epoch warmup) |
| Final LR | 1e-6 |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| Batch size | 12 |
| Epochs | 100 |
| Loss | Dice + CE (class weights [0.2, 0.3, 0.25, 0.25]) |
| Split | 1,000 train / 125 valid / 126 test |
| Data format | NIfTI → h5 with z-score per modality |

---

## Method-Specific Hyperparameters

All methods share the **same backbone, optimizer, and data setup per dataset**. Only the gradient modulation differs.

### Baseline
- No gradient modulation.
- Standard joint training: forward → CE loss → backward → optimizer.step()

### OGM-GE (Peng et al., CVPR 2022)
- `α = 0.8` (paper default)
- Modulation epochs: 0-50 (first half of training)
- Scales down dominant modality's encoder gradients: `coeff = 1 - tanh(α · ReLU(ratio))`
- Adds Gaussian noise to modulated gradients

### CGGM (Guo et al., NeurIPS 2024)
- `ρ = 1.3` (gradient scaling amplifier)
- `λ = 0.2` (direction loss weight)
- `cls_lr = 5e-4` (auxiliary classifier LR)
- Gradient clipping norm: 0.8
- Uses per-modality auxiliary classifiers to modulate both magnitude and direction.

### MMPareto (Wei & Hu, ICML 2024)
- `γ = 1.5` (gradient magnitude scaling)
- Computes cosine similarity between joint and unimodal loss gradients
- If conflict (cos < 0): solves Pareto-optimal weights via MinNormSolver
- If aligned: uniform weights [0.5, 0.5]

### AGM (Li et al., 2023)
- `α = 1.0` (degree of gradient modulation)
- Modulation epochs: 0-50
- Per-modality coefficients: `exp(α · min(optimal_ratio - current_ratio, 10))`
- Uses running average of per-modality CE loss scores

### G-Blend (Wang et al., CVPR 2020)
- No tunable hyperparameters (beyond base config)
- Computes overfitting-to-generalization ratio per modality each epoch
- Weights unimodal losses inversely proportional to OG ratio

### InfoReg (Huang et al., CVPR 2025) — **CREMA-D only**
- `β = 0.9` (regulation strength)
- `K = 0.04` (Fisher trace threshold for PLW detection)
- 100 epochs (paper: 50)

### MILES (Guerra et al., IJCNN 2025) — **CREMA-D only**
- `τ = 0.2` (conditional utilization rate threshold)
- `μ = 0.5` (dominant modality LR reduction factor)
- Per-modality optimizer groups

### Boost only (ours, α=0.5)
- `continuous_alpha = 0.5` (boost strength)
- `continuous_scale_max = 2.0` (max boost cap)
- `continuous_scale_ema = 0.3` (EMA smoothing on scales)
- `continuous_eval_freq = 20` (probe refresh interval K)
- Probe: linear classifier, Adam lr=1e-3, trained on split-half batches
- EMA on probe accuracy: β = 0.1
- Unimodal regularization: γ = 1.0

### Boost + OGM-GE (ours, α=0.75)
- All Boost-only params, but `continuous_alpha = 0.75`
- Plus OGM-GE at `α = 0.8` (standard)
- Applied jointly: OGM-GE throttles dominant, ASGML boosts weak

### ARL (Wei et al., ICCV 2025) — **Excluded from main results**
- Reproducibility issues (non-reproducible 13.7pp gap vs published)
- Not included in Table 1. Only in Related Work.

---

## Random Seeds

All multi-seed experiments use the same 5 seeds for reproducibility:

**1-frame CREMA-D ablation**: `{42, 0, 1, 2, 3}`
**All other datasets**: `{42, 123, 456, 789, 1024}`

---

## Reproduction Commands

### CREMA-D 3f (from-scratch)
```bash
# Baseline
python scripts/train.py --config configs/cremad.yaml --mode baseline \
    --num-frames 3 --fps 3 --seed 42

# OGM-GE
python scripts/train.py --config configs/cremad.yaml --mode baseline \
    --ogm-ge --alpha 0.8 --num-frames 3 --fps 3 --seed 42

# Boost+OGM-GE (ours)
python scripts/train.py --config configs/cremad.yaml --mode adaptive \
    --asgml-mode continuous --continuous-alpha 0.75 \
    --ogm-ge --alpha 0.8 --num-frames 3 --fps 3 --seed 42
```

### Text+Image (Sarcasm, Twitter, Food101)
```bash
# Pre-extract features (one-time, ~10 min)
python scripts/extract_text_image_features.py --dataset food101 --batch-size 128

# Run any method using configs/food101.yaml
python scripts/train.py --config configs/food101.yaml --mode baseline --seed 42
python scripts/train.py --config configs/food101.yaml --mode mmpareto --seed 42
python scripts/train.py --config configs/food101.yaml --mode adaptive \
    --asgml-mode continuous --continuous-alpha 0.5 --seed 42
```

### BraTS (standalone script)
```bash
python scripts/train_brats.py --mode baseline --seed 42
python scripts/train_brats.py --mode ogm_ge --ogm-alpha 0.8 --seed 42
python scripts/train_brats.py --mode boost_ogm_ge --boost-alpha 0.5 \
    --ogm-alpha 0.8 --seed 42
```

---

## Hardware

All experiments on single RTX 4090 (24GB VRAM), 1TB SSD, Ubuntu WSL2.

| Dataset | GPU util | VRAM | Time per run |
|---------|---------|------|--------------|
| CREMA-D 3f | ~90% | ~10GB | ~5-20 min |
| AVE | ~90% | ~10GB | ~20-30 min |
| KS | ~85% | ~10GB | ~60-90 min |
| MOSEI/MOSI | ~20% | ~2GB | ~1-2 min |
| Sarcasm/Twitter | ~20% | ~2GB | ~1-2 min |
| Food101 | ~50% | ~2GB | ~5-10 min |
| BraTS | ~40% | ~15GB | ~1.5-2 hours |

---

## Config Files Reference

| Config | Path |
|--------|------|
| CREMA-D (1f default, 3f via CLI override `--num-frames 3 --fps 3`) | `configs/cremad.yaml` |
| AVE | `configs/ave.yaml` |
| KS | `configs/kinetics_sounds.yaml` |
| MOSEI | `configs/mosei.yaml` |
| MOSI | `configs/mosi.yaml` |
| Sarcasm | `configs/sarcasm.yaml` |
| Twitter15 | `configs/twitter.yaml` |
| Food101 | `configs/food101.yaml` |
| BraTS | hardcoded in `scripts/train_brats.py` |

All YAML configs use identical structure — `dataset`, `model`, `training`, `asgml`, `logging`, `evaluation` sections.
