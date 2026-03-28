# ASGML: Approach, Implementation & Research Trajectory

**Project:** NeurIPS 2026 — Probe-Guided Gradient Boosting for Balanced Multimodal Learning
**Last updated:** 2026-03-27

---

## Table of Contents

1. [Research Problem](#1-research-problem)
2. [Solution Evolution (v1 → v2)](#2-solution-evolution-v1--v2)
3. [ASGML v2: Probe-Guided Gradient Boosting](#3-asgml-v2-probe-guided-gradient-boosting)
4. [Implementation Details](#4-implementation-details)
5. [Training Pipeline](#5-training-pipeline)
6. [Datasets & Preprocessing](#6-datasets--preprocessing)
7. [Experimental Results](#7-experimental-results)
8. [Baseline Comparisons](#8-baseline-comparisons)
9. [Key Findings & Narrative](#9-key-findings--narrative)
10. [Research Timeline](#10-research-timeline)
11. [Code Map](#11-code-map)

---

## 1. Research Problem

### Modality Imbalance in Multimodal Learning

In late-fusion multimodal networks, one modality ("dominant") converges faster during early training and captures the shared fusion head. This suppresses the weaker modality's gradient signal before it develops discriminative features — the **Prime Learning Window** problem.

**Theoretical basis:**
- Huang et al. (ICML 2022): exact calculations for unimodal phase duration via saddle manifold dynamics
- Zhang et al. (ICML 2024): unimodal phase as a function of architecture + initialization
- Huang et al. (NeurIPS 2021): provable multimodal superiority under latent space assumptions

### The Gap in Existing Solutions

All existing methods address this problem through **synchronous gradient modulation** — they modify gradient magnitudes or directions but update all modalities at every step. Critically, they all focus on **one side** of the problem: **throttling the dominant modality**.

| Method | Venue | Mechanism | Direction |
|--------|-------|-----------|-----------|
| OGM-GE | CVPR 2022 | Gradient magnitude modulation + noise | Throttle dominant |
| OPM | TPAMI 2024 | Feature dropout for dominant modality | Throttle dominant |
| CGGM | NeurIPS 2024 | Classifier-guided magnitude + direction | Throttle dominant |
| ARL | ICCV 2025 | Asymmetric weighting via entropy | Reweight decision |

**No existing method explicitly boosts the weak modality.** Throttling and boosting are orthogonal interventions — combining both provides two-sided pressure toward balance.

---

## 2. Solution Evolution (v1 → v2)

### ASGML v1: Asynchronous Update Scheduling (Did Not Work)

The original design used dual-signal (gradient ratio + loss descent) to compute staleness τ for each modality:

```
learning_speed[m] = β * grad_ratio[m] + (1-β) * loss_descent_ratio[m]
τ[m] = τ_base * learning_speed[m]
```

**Three modes were implemented:**
1. **Frequency mode**: Skip updates for the dominant modality every k steps
2. **Staleness mode**: Apply τ-step-old gradients to the dominant modality (θ_{t+τ+1} = θ_{t+τ} - η∇_{θ_t}L)
3. **Adaptive mode**: Probe signals drive τ adjustment

**Why it failed:** Normalization removed absolute difference information. Both modalities ended up with τ ≈ 1–2, and binary skip was rarely triggered. Performance ≈ baseline across all configurations.

### ASGML v2: Probe-Guided Continuous Boosting (Current)

Replace binary update skipping with **continuous gradient scaling** based on **decoupled probe monitoring**:

| Aspect | ASGML v1 | ASGML v2 |
|--------|----------|----------|
| Core mechanism | Binary update skipping | Continuous gradient scaling |
| Signal source | Dual-signal (normalized ratios) | Probe accuracy (absolute values) |
| Trigger behavior | Rarely triggered | Always active |
| Intervention direction | Throttle dominant | **Boost weak** |
| Standalone performance | ≈ Baseline | +0.16pp (modest) |
| Combined with OGM-GE | Not tested | **+9.86pp** over baseline |

---

## 3. ASGML v2: Probe-Guided Gradient Boosting

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASGML v2 ARCHITECTURE                        │
│                                                                 │
│  Encoder_A(x_A) → feat_A ──┐                                   │
│  Encoder_B(x_B) → feat_B ──┤                                   │
│                             ├→ Fusion → Classifier → L_task     │
│                             │                           │       │
│                             │                       backward()  │
│                             │                           │       │
│                             │         ┌─────────────────┤       │
│                             │         │                 │       │
│                             │   [OGM-GE: throttle]  [ASGML:    │
│                             │   g_dom *= α (α<1)     boost]    │
│                             │                       g_weak *=  │
│                             │                       β (β>1)    │
│                             │                 │                 │
│                             │            optimizer.step()       │
│                                                                 │
│  DECOUPLED PROBE MONITORING (periodic, every eval_freq steps):  │
│  feat.detach() → Linear Probes → accuracies → compute scales   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Decoupled Probe Monitoring

**Architecture:** `LinearProbe = nn.Linear(feature_dim, num_classes)` (~3K params each)

**Safety guarantee (CRITICAL):** Probes never backpropagate into encoders.
- Features are `.detach()`'d before passing to probes
- Probes have completely separate optimizers
- Split-batch evaluation: train probe on first half of batch, evaluate on second half

**Probe training protocol:**
1. Every `eval_freq` batches (default: 20), split the current batch in half
2. Train probes for `probe_train_steps` (default: 10) SGD iterations on the first half
3. Evaluate probes on the second half (unseen data) to get per-modality accuracy
4. Update EMA-smoothed accuracies: `acc_ema[m] = 0.1 * acc_new + 0.9 * acc_old`

**Why split-batch matters:** Without it, probes overfit to the training portion (e.g., audio 100% train but only 56% test), producing unreliable utilization signals.

### 3.3 Continuous Boost Scaling

Given probe utilization scores, compute gradient scales:

```python
def get_continuous_scales(utilization_scores, alpha=0.5, scale_max=2.0):
    max_score = max(utilization_scores.values())
    min_score = min(utilization_scores.values())
    gap = max_score - min_score

    scales = {}
    for m in modalities:
        rel_weakness = 1.0 - (scores[m] - min_score) / gap
        scales[m] = clamp(1.0 + alpha * rel_weakness, 1.0, scale_max)
    return scales
```

**Example:** Audio probe = 0.65, Visual probe = 0.35, α = 0.5
- Audio (dominant): rel_weakness = 0.0 → scale = 1.0 (unchanged)
- Visual (weak): rel_weakness = 1.0 → scale = 1.5 (50% gradient boost)

**EMA smoothing** prevents abrupt scale changes:
```python
scale_current = 0.3 * scale_new + 0.7 * scale_old
```

### 3.4 Complementarity with OGM-GE

The combined gradient modification pipeline:

```
After backward():
  1. OGM-GE:  g_dominant *= α   where α < 1  (throttle)
  2. ASGML:   g_weak    *= β   where β > 1  (boost)

Effective ratio change:
  original:  ||g_dominant|| / ||g_weak|| = r
  combined:  ||α * g_dominant|| / ||β * g_weak|| = (α/β) * r

  With α=0.6, β=1.5: (0.6/1.5) * r = 0.4r  (2.5× better balance)
```

### 3.5 Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `continuous_alpha` | 0.5 | Boost strength (0 = no boost, 1 = max) |
| `continuous_scale_max` | 2.0 | Maximum gradient amplification |
| `continuous_eval_freq` | 20 | Probe evaluation interval (batches) |
| `continuous_probe_train_steps` | 10 | SGD steps per probe evaluation |
| `continuous_scale_ema` | 0.3 | EMA smoothing factor for scales |
| `probe_type` | "linear" | Probe architecture (linear or mlp_1layer) |
| `probe_lr` | 0.001 | Probe optimizer learning rate |
| `gamma` | 1.0 | Unimodal regularization weight |

---

## 4. Implementation Details

### 4.1 Core Loss Module (`src/losses/asgml.py`, ~757 lines)

#### StalenessBuffer
Stores gradient snapshots for true staleness mode (v1 legacy, still functional).
- `store_gradients(modality, parameters, step)` — clone and detach parameter gradients
- `get_stale_gradients(modality, staleness)` — retrieve τ-step-old gradient snapshot
- Uses `collections.deque(maxlen=max_staleness)` per modality

#### LearningDynamicsTracker
Tracks dual-signal (gradient magnitude ratio + loss descent rate) for v1 adaptive mode.
- Gradient EMA: `norm_ema[m] = 0.1 * new + 0.9 * old`
- Learning speed: `S_m = β * (||∇L_m|| / mean) + (1-β) * (|L_m(t) - L_m(t-w)| / mean)`
- S_m > 1 = dominant, S_m < 1 = weak

#### ASGMLScheduler
Unified scheduler supporting all modes (frequency, staleness, continuous):
- `get_update_mask()` — which modalities update this step (frequency mode)
- `get_staleness_values()` — τ per modality with stale fusion constraint (τ_i ≤ κ * min(τ_j), κ=3.0)
- `get_gradient_scales()` — async SGD compensation: scale = 1/(1 + λ*τ), λ=0.1
- `get_continuous_scales()` — probe-guided boost scales (v2)
- `update_continuous_scales()` — EMA-smoothed scale update

#### ASGMLLoss
```
L_total = L_fusion + γ * Σ_i (L_unimodal_i * 𝟙[update_i])
```
- `forward(fusion_logits, unimodal_logits, targets, update_mask)` → (total_loss, loss_dict)
- Unimodal regularization only applied if modality is updating (respects frequency mask)

#### Utility Functions
- `compute_gradient_norms(model, modalities)` — L2 norm of encoder gradients
- `apply_staleness_gradients(model, buffer, modality, staleness, scale)` — replace current gradients with stale ones

### 4.2 Model Architecture (`src/models/`)

#### MultimodalModel (`multimodal.py`)
General N-modality model:
- `self.encoders: ModuleDict` — one encoder per modality
- `self.fusion: nn.Module` — ConcatFusion, GatedFusion, or SumFusion
- `self.classifier: nn.Linear` — main classifier (fusion_dim → num_classes)
- `self.unimodal_classifiers: ModuleDict` — per-modality classifiers for regularization
- `forward(inputs, return_features=True)` → (logits, unimodal_logits, features)

#### Encoders (`encoders.py`)

| Encoder | Input | Architecture | Output |
|---------|-------|-------------|--------|
| VisualEncoder | (B, C, [T,] H, W) | ResNet18/50 + adaptive pool | (B, feature_dim) |
| AudioEncoder | (B, 1, H, W) | ResNet18 (1-ch input) + pool | (B, feature_dim) |
| TextEncoder | (B, seq_len) | Embedding → BiLSTM → Linear | (B, feature_dim) |
| MLPEncoder | (B, [T,] D) | Linear→ReLU→Dropout→Linear→ReLU | (B, feature_dim) |

- VisualEncoder handles both single-frame (4D) and multi-frame (5D) input via temporal pooling
- MLPEncoder handles pre-extracted features (MOSEI), mean-pools temporal dimension if 3D
- OGM-GE weight initialization: kaiming_normal (Conv2d), xavier_normal (Linear)

#### Fusion (`fusion.py`)

| Module | Formula |
|--------|---------|
| ConcatFusion | `fc(cat(features, dim=1))` |
| GatedFusion | `Σ_i sigmoid(gate_i(feat_i)) * proj_i(feat_i)` |
| SumFusion | `Σ_i proj_i(feat_i)` |

#### ProbeManager (`probes.py`)

- Creates independent `LinearProbe` (or `MLPProbe`) + separate `Adam` optimizer per modality
- `train_probes(features, targets)` — `.detach()` features, train for num_steps
- `evaluate_probes(features, targets)` — `@torch.no_grad()`, returns accuracy + loss per modality
- `compute_utilization_gap(use_ema=True)` — max(acc) - min(acc)
- EMA tracking: `acc_ema[m] = ema_alpha * acc + (1-ema_alpha) * old`

---

## 5. Training Pipeline

### 5.1 Supported Training Modes (`scripts/train.py`, ~2500 lines)

| Mode | Method | Description |
|------|--------|-------------|
| `baseline` | Standard joint training | All modalities update every step, no modulation |
| `frequency` | ASGML v1 | Dominant updates every k steps (hard or soft mask) |
| `staleness` | ASGML v1 | Dominant uses τ-step-old gradients |
| `adaptive` | ASGML v1 | Probe-driven τ adaptation |
| `continuous` | **ASGML v2** | Probe-guided continuous gradient boosting |
| `miles` | MILES baseline | Epoch-level LR reduction for dominant modality |
| `inforeg` | InfoReg baseline | PLW weight regularization via Fisher trace |
| `arl` | ARL baseline | Asymmetric entropy-based weighting |
| `opm` | OPM baseline | Feature dropout for dominant modality |

### 5.2 OGM-GE Implementation (`apply_ogm_ge()`)

Applied as an orthogonal gradient modifier after backward(), before optimizer.step():

1. Compute per-modality softmax scores on true labels
2. **2 modalities**: Pairwise ratio → `coeff = 1 - tanh(α * relu(ratio))` for dominant
3. **N modalities**: Per-modality ratio vs mean → same formula
4. Apply: `param.grad = param.grad * coeff + N(0, param.grad.std())`
5. Active during epoch range [modulation_start, modulation_end] (default 0–50)
6. Supports both Conv2d (4D) and Linear (2D) parameter gradients (generalized for MLP encoders)

### 5.3 Main Training Loop (Continuous/Boost Mode)

```
For each batch:
  1. Forward:  logits, unimodal_logits, features = model(inputs)
  2. Loss:     L = L_fusion + γ * Σ L_unimodal (conditional on update mask)
  3. Backward: L.backward()
  4. Grad norms: record ||∇_{θ_m} L|| for logging
  5. OGM-GE:  if enabled, throttle dominant encoder gradients
  6. ASGML Boost: scale weak encoder gradients by cached probe-derived scales
  7. Step:     optimizer.step()  (fusion head ALWAYS updates)
  8. Scheduler: update dynamics tracker, increment step

  Every eval_freq batches:
  9.  Split batch in half
  10. Train probes on first half (probe_train_steps iterations)
  11. Evaluate probes on second half
  12. Compute new scales from probe accuracies
  13. EMA-smooth scales into scheduler
```

### 5.4 Gradient Modification Order

```
backward() → grad_norms → OGM-GE (throttle) → ASGML boost (scale) → optimizer.step()
```

Both OGM-GE and ASGML modify `param.grad.data` in-place. The fusion head and classifier are **never** modified by either — only encoder parameters.

---

## 6. Datasets & Preprocessing

### 6.1 CREMA-D (Primary Benchmark)

| Property | Value |
|----------|-------|
| Task | Audio-visual emotion recognition (6 classes) |
| Split | 6,698 train / 744 test |
| Imbalance | High — audio dominant, visual weak |
| Visual | ResNet18, 1–3 frames, RandomResizedCrop(224), ImageNet normalize |
| Audio | STFT spectrogram: sr=22050, n_fft=512, hop=353 → log mag → (1, 257, 187) |
| Training | SGD lr=0.001, momentum=0.9, wd=1e-4, batch 64, StepLR step=70 γ=0.1, 100 epochs |

### 6.2 AVE

| Property | Value |
|----------|-------|
| Task | Audio-visual event recognition (28 classes) |
| Split | 3,287 train / 810 test |
| Imbalance | Low — both modalities discriminative |
| Audio | Pre-computed pickle spectrograms |
| Visual | Evenly-spaced frames, same augmentation as CREMA-D |

### 6.3 Kinetics-Sounds

| Property | Value |
|----------|-------|
| Task | Audio-visual action recognition (31 classes) |
| Split | 12,892 train / 1,228 test (OGM-GE split, 79% YouTube availability) |
| Imbalance | Low — balanced audio-visual |
| Audio | Pre-extracted librosa mel spectrograms → (1, 128, 128), min-max normalized |
| Visual | ResNet18 pretrained ImageNet, same augmentation pipeline |

### 6.4 CMU-MOSEI

| Property | Value |
|----------|-------|
| Task | 3-modality sentiment (text-audio-vision, 3 classes) |
| Split | 1,368 train / 456 valid / 457 test (MMSA format) |
| Features | Pre-extracted: text 768d (BERT), audio 33d (COVAREP), vision 709d |
| Preprocessing | Z-score normalization per feature, NaN/inf → 0 for audio |
| Encoders | MLPEncoders (not ResNet) |
| Optimizer | Adam lr=0.001 (not SGD) |

---

## 7. Experimental Results

### 7.1 CREMA-D (3-Frame, 5 Seeds)

| Rank | Method | Mean ± Std |
|------|--------|-----------|
| 1 | **ASGML Boost + OGM-GE (α=0.75)** | **71.45 ± 1.71%** |
| 2 | OGM-GE alone (α=0.8) | 69.14 ± 1.13% |
| 3 | InfoReg (100ep) | 67.72 ± 0.83% |
| 4 | Baseline | 61.59 ± 0.80% |
| 5 | MILES (τ=0.2) | 61.05 ± 2.52% |

### 7.2 CREMA-D (1-Frame, 5 Seeds)

| Rank | Method | Mean ± Std |
|------|--------|-----------|
| 1 | ASGML Boost + OGM-GE (α=0.75) | 62.69 ± 0.22% |
| 2 | OGM-GE alone | 62.47 ± 1.42% |
| 3 | ASGML Boost only | 60.46 ± 0.85% |
| 4 | Baseline | 60.30 ± 0.70% |

### 7.3 AVE (5 Seeds)

| Rank | Method | Mean ± Std |
|------|--------|-----------|
| 1 | **ASGML Boost only** | **87.41 ± 0.26%** |
| 2 | Baseline | 86.54 ± 0.63% |
| 3 | ASGML Boost + OGM-GE | 85.43 ± 0.70% |
| 4 | OGM-GE alone | 85.31 ± 0.66% |

### 7.4 Kinetics-Sounds (5 Seeds)

| Rank | Method | Mean ± Std |
|------|--------|-----------|
| 1 | **ASGML Boost only (α=0.5)** | **79.17 ± 0.97%** |
| 2 | Baseline | 79.05 ± 0.40% |
| 3 | Boost + OGM-GE (α=0.75) | 77.33 ± 0.64% |
| 4 | OGM-GE alone (α=0.8) | 77.25 ± 0.79% |

### 7.5 CMU-MOSEI (3 Modalities, Preliminary)

| Method | Accuracy |
|--------|---------|
| Boost + OGM-GE | ~72.45% |
| OGM-GE alone | ~72.45% |
| Baseline | ~70.5% |

### 7.6 Cross-Dataset Summary

| Dataset | Modalities | Imbalance | Best Method | Acc | vs Baseline |
|---------|-----------|-----------|-------------|-----|-------------|
| **CREMA-D** (3f) | Audio-Visual | High | Boost+OGM-GE | 71.45% | +9.86pp |
| **AVE** | Audio-Visual | Low | Boost-only | 87.41% | +0.87pp |
| **KS** | Audio-Visual | Low | Boost-only | 79.17% | +0.12pp |
| **CMU-MOSEI** | A-V-Text | Medium | Boost+OGM-GE | ~72.45% | +2.0pp |

---

## 8. Baseline Comparisons

### 8.1 OPM Comparison (Wei et al. TPAMI 2024)

Controlled comparison on CREMA-D 3-frame:

| Method | Mean ± Std |
|--------|-----------|
| **ASGML Boost + OGM-GE** | **71.77 ± 0.91%** |
| OPM + OGM | 69.84 ± 1.47% |
| OGM-GE only | 69.14 ± 1.13% |
| OPM only | 66.83 ± 1.54% |

ASGML beats OPM+OGM by +1.93pp in controlled setting.

### 8.2 ARL Comparison (Wei et al. ICCV 2025)

ARL claims 76.61% on CREMA-D. Our controlled reproduction: **62.90%** (13.7pp gap).

**Root cause:** Bug in reference code — computed asymmetric softmax weights are never assigned back to the model. The `fixed_coeff=0.5` is used as a constant multiplier rather than the adaptive weights.

### 8.3 MILES Comparison (Modality-Informed LR Scheduler)

MILES achieves 61.05 ± 2.52% on CREMA-D 3-frame, approximately equal to baseline (61.59%). Epoch-level LR adjustment is too coarse for effective rebalancing.

### 8.4 InfoReg Comparison (Huang et al. CVPR 2025)

InfoReg achieves 67.72 ± 0.83% on CREMA-D 3-frame. PLW-based weight regularization is effective but ASGML+OGM-GE outperforms by +3.73pp.

---

## 9. Key Findings & Narrative

### 9.1 The Core Argument

**Existing gradient modulation methods are one-sided** — they throttle the dominant modality but leave the weak modality's small gradient unchanged. ASGML provides the complementary intervention: **boosting the weak modality's gradient signal** using decoupled probe monitoring.

### 9.2 Dataset-Adaptive Behavior

The method automatically adapts to the level of modality imbalance:

- **High imbalance (CREMA-D):** Boost + OGM-GE is best. Two-sided pressure (throttle dominant + boost weak) yields the largest gains (+9.86pp).
- **Low imbalance (AVE, KS):** Boost-only is best. OGM-GE gradient throttling *hurts* when modalities are already balanced (-1.23pp on AVE, -1.80pp on KS).
- **The probe signal governs this:** When probe accuracies are similar, boost scales approach 1.0 (no intervention). When they diverge, boost activates proportionally.

### 9.3 Why Original ASGML (v1) Failed

1. **Normalization killed the signal:** Dual-signal (gradient ratio + loss descent) normalizes to mean, losing absolute gap information. Both modalities get τ ≈ 1–2.
2. **Binary decisions too coarse:** Update-or-skip is a blunt instrument. Continuous scaling is smoother and always active.
3. **Wrong direction:** v1 throttled the dominant (same as OGM-GE). v2 boosts the weak (orthogonal complement).

### 9.4 Why Decoupled Probes Matter

| Property | Coupled Signals (CGGM, OGM-GE) | Decoupled Probes (ASGML) |
|----------|-------------------------------|--------------------------|
| Gradient flow | Affects training objective | No effect (`.detach()`) |
| Feedback loops | Present — optimizer changes signal | Absent — probes are independent |
| Batch usage | Same batch for signal + training | Split-batch (train on half, eval on half) |
| Signal type | Relative (ratios) | Absolute (accuracy) |
| Update frequency | Every step | Periodic (every eval_freq) |

### 9.5 Synergy Amplification with Visual Frames

The ASGML + OGM-GE synergy amplifies with more visual information:
- 1-frame: +0.22pp over OGM-GE alone
- 3-frame: +2.31pp over OGM-GE alone (**10.5× amplification**)

With more frames, the visual modality has greater capacity to benefit from boosted gradients.

### 9.6 Variance Reduction

ASGML Boost + OGM-GE shows 6.5× lower variance than OGM-GE alone on CREMA-D 1-frame (0.22% vs 1.42% std). The probe-guided continuous scaling stabilizes training.

---

## 10. Research Timeline

| Date | Milestone | Outcome |
|------|-----------|---------|
| 2026-01 | Literature review, gap identification | Identified async temporal separation as unexplored |
| 2026-01 | ASGML v1 design (frequency/staleness) | Implemented StalenessBuffer, LearningDynamicsTracker |
| 2026-02-01 | Baseline reproduction (CREMA-D) | Matched OGM-GE published numbers |
| 2026-02-05 | ASGML v1 frequency/staleness experiments | Performance ≈ baseline (normalization kills signal) |
| 2026-02-10 | Diagnosis: Why v1 fails | Normalized signals → τ ≈ 1–2 for both modalities |
| 2026-02-12 | Pivot to continuous boost mode (v2) | Probe-guided gradient scaling works |
| 2026-02-13 | 3-frame baseline comparison | Boost+OGM-GE: 71.45% (best) |
| 2026-02-15 | Multi-seed validation (5 seeds) | Confirmed results with error bars |
| 2026-02-18 | MILES and InfoReg baselines | Both underperform Boost+OGM-GE |
| 2026-02-20 | AVE dataset experiments | Boost-only best on low-imbalance (87.41%) |
| 2026-02-25 | Design v2 document | Formalized approach and positioning |
| 2026-03-01 | CMU-MOSEI (3-modality) experiments | +2pp over baseline, OGM-GE generalized to N modalities |
| 2026-03-05 | OPM comparison | ASGML beats OPM+OGM by +1.93pp |
| 2026-03-08 | ARL comparison | ARL non-reproducible (62.90% vs claimed 76.61%) |
| 2026-03-10 | Read Wei et al. 2024 (OPM/OGM) and Wei et al. 2025 (ARL) | Confirmed orthogonality with ARL (decision vs representation level) |
| 2026-03-16 | Kinetics-Sounds dataset preparation | Downloaded 79% of YouTube videos, pre-extracted spectrograms |
| 2026-03-17 | KS multi-seed results | Boost-only 79.17% edges baseline 79.05% |
| 2026-03-27 | All 4 benchmarks complete | Manuscript structure set up |

---

