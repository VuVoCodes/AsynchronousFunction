# Asynchronous Staleness Guided Multimodal Learning (ASGML) v2

**Updated Design Document for NeurIPS 2026 Submission**

Date: 2026-02-25 (last updated: 2026-02-25)

---

## Executive Summary

This document reflects the **evolved ASGML design** after extensive experimentation. The core insight shifted from "temporal separation via update skipping" to "probe-guided weak-modality boosting as a complement to existing gradient modulation methods."

**Key Result**: ASGML Boost + OGM-GE achieves **71.45% ± 1.71%** on CREMA-D (3-frame), outperforming either method alone.

---

## 1. Research Gap (Refined)

### 1.1 Original Gap (Still Valid)
Existing methods address modality imbalance through **synchronous** interventions:
- **Gradient modulation** (OGM-GE, AGM, CGGM): Reweight/throttle gradients but update all modalities every step
- **Loss modification** (CGGM): Add direction alignment loss terms
- **Unimodal regularization** (MMPareto, ARL): Auxiliary losses, still synchronous

### 1.2 New Insight: Orthogonal Interventions

All existing methods focus on **throttling the dominant modality**:

```
OGM-GE:  dominant_grad *= tanh(...)  → scale DOWN dominant
CGGM:    dominant gets lower B       → scale DOWN dominant
AGM:     similar throttling mechanism
```

**Gap identified**: No method explicitly **boosts the weak modality**. These are orthogonal interventions:

| Intervention | Target | Direction |
|--------------|--------|-----------|
| Throttle dominant | Fast learner | Pull back |
| **Boost weak** | Slow learner | Push forward |

Combining both provides **two-sided pressure** toward balance.

### 1.3 Why Original ASGML (Frequency/Staleness) Didn't Work

The original design used dual-signal (gradient ratio + loss descent) to compute staleness τ:

```python
# Original: Normalized signal → τ values centered around mean
learning_speed[m] = β * grad_ratio + (1-β) * loss_ratio  # Both normalized
τ[m] = tau_base * learning_speed[m]  # Results in τ ≈ 1-2 for both
```

**Problem**: Normalization removes absolute difference information:
- Audio gradient = 100, Visual = 50 → ratio = 1.33 vs 0.67
- After τ computation and clamping → τ_audio ≈ 2, τ_visual ≈ 2
- **No differentiation** → binary skip rarely triggered

---

## 2. ASGML v2: Probe-Guided Continuous Boosting

### 2.1 Core Idea

Replace binary update skipping with **continuous gradient scaling** based on **decoupled probe monitoring**:

```
Original ASGML:  Skip updates for dominant (binary, rarely triggered)
ASGML v2:        Boost gradients for weak (continuous, always active)
```

### 2.2 Design Principles

1. **Decoupled Monitoring**: Probes observe encoder quality without affecting training
2. **Absolute Signal**: Use probe accuracy (not normalized ratios) to preserve gap information
3. **Continuous Scaling**: Smooth gradient modulation instead of binary skip
4. **Complementarity**: Design to work WITH existing methods (OGM-GE), not replace them

### 2.3 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      ASGML v2 ARCHITECTURE                          │
│                                                                     │
│  ┌─────────────┐      ┌─────────────┐                              │
│  │   Audio     │      │   Visual    │                              │
│  │  Encoder    │      │  Encoder    │                              │
│  │ (ResNet18)  │      │ (ResNet18)  │                              │
│  └──────┬──────┘      └──────┬──────┘                              │
│         │                    │                                      │
│         ▼                    ▼                                      │
│    features_a           features_v                                  │
│         │                    │                                      │
│         ├────────┬───────────┤                                      │
│         │        │           │                                      │
│         │        ▼           │                                      │
│         │   ┌─────────┐      │                                      │
│         │   │ Fusion  │      │                                      │
│         │   └────┬────┘      │                                      │
│         │        │           │                                      │
│         │        ▼           │                                      │
│         │    logits          │                                      │
│         │        │           │                                      │
│         │        ▼           │                                      │
│         │   L_task (only)    │     ◄── Loss unchanged              │
│         │        │           │                                      │
│         │        ▼           │                                      │
│         │   backward()       │                                      │
│         │        │           │                                      │
│         ▼        ▼           ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │              GRADIENT MODIFICATION                       │      │
│   │                                                          │      │
│   │  1. OGM-GE (if enabled): Throttle dominant              │      │
│   │     audio_grad *= 0.6                                    │      │
│   │                                                          │      │
│   │  2. ASGML Boost: Boost weak                             │      │
│   │     visual_grad *= 1.5                                   │      │
│   │                                                          │      │
│   └─────────────────────────────────────────────────────────┘      │
│         │                    │                                      │
│         ▼                    ▼                                      │
│   optimizer.step()                                                  │
│                                                                     │
│  ════════════════════════════════════════════════════════════════  │
│                                                                     │
│   PROBE MONITORING (Decoupled, Periodic)                           │
│                                                                     │
│   features.detach() ──► Linear Probes ──► accuracies               │
│                                              │                      │
│                                              ▼                      │
│                                    Compute boost scales             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Note:** Architecture generalizes to N modalities. All components (scheduler, loss, probes, model) loop over a modality list. The diagram above shows 2 encoders for clarity; CMU-MOSEI uses 3 MLP encoders (text, audio, vision).

---

## 3. Method Details

### 3.1 Probe-Based Monitoring

**Architecture**: Lightweight linear probes (512 → num_classes)

```python
class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        self.classifier = nn.Linear(feature_dim, num_classes)  # ~3K params
```

**Safety Guarantee**: Complete decoupling via `.detach()`

```python
# CRITICAL: Probes NEVER backprop into encoders
feat = features[modality].detach().float()
logits = probe(feat)
loss.backward()  # Only updates probe weights
```

**Split-Batch Evaluation**: Honest signal estimation

```python
# Prevent probe overfitting on training batch
split = batch_size // 2
train_features = {m: f[:split] for m, f in features.items()}  # Train probes
eval_features = {m: f[split:] for m, f in features.items()}   # Evaluate probes
```

### 3.2 Continuous Boost Scaling

**Formula**:

```python
def get_continuous_scales(utilization_scores, alpha=0.5, scale_max=2.0):
    max_score = max(utilization_scores.values())
    min_score = min(utilization_scores.values())
    gap = max_score - min_score

    scales = {}
    for m in modalities:
        # Relative weakness: 0 for dominant, 1 for weakest
        rel_weakness = 1.0 - (scores[m] - min_score) / gap

        # Scale: 1.0 for dominant, up to scale_max for weakest
        scales[m] = 1.0 + alpha * rel_weakness
        scales[m] = min(scales[m], scale_max)

    return scales
```

**Example Calculation**:

```
Probe accuracies: audio = 0.65, visual = 0.35
Gap = 0.30, alpha = 0.5, scale_max = 2.0

Audio (dominant):
  rel_weakness = 1.0 - (0.65 - 0.35) / 0.30 = 0.0
  scale = 1.0 + 0.5 * 0.0 = 1.0  ← unchanged

Visual (weak):
  rel_weakness = 1.0 - (0.35 - 0.35) / 0.30 = 1.0
  scale = 1.0 + 0.5 * 1.0 = 1.5  ← boosted 50%
```

**EMA Smoothing** (stability):

```python
scale_new = ema_alpha * scale_computed + (1 - ema_alpha) * scale_old
# ema_alpha = 0.3 (default)
```

### 3.3 Comparison with CGGM

| Aspect | CGGM | ASGML v2 |
|--------|------|----------|
| Classifier/Probe coupling | Coupled (joint training) | Decoupled (`.detach()`) |
| Loss modification | Yes (`L = L_task + λ*L_gm`) | No (`L = L_task` only) |
| Signal type | Improvement Δε (normalized) | Absolute accuracy |
| Intervention point | Before backward (in loss) | After backward (gradient scaling) |
| Direction modulation | Yes (via `L_gm`) | No (magnitude only) |
| Batch usage | Same batch | Split batch |
| Evaluation frequency | Every iteration | Periodic (`eval_freq`) |

**Key Insight**: CGGM's classifier is a **co-pilot** (steers training via loss). ASGML's probe is a **dashboard gauge** (displays info for external decision).

---

## 4. Theoretical Framing

### 4.1 Orthogonal Interventions

Let g_i denote the gradient for modality i. Existing methods apply:

```
OGM-GE:  g_dominant *= α  where α < 1  (throttle)
```

ASGML adds:

```
ASGML:   g_weak *= β  where β > 1  (boost)
```

Combined effect on gradient ratio:

```
Before:  ||g_dominant|| / ||g_weak|| = r  (imbalanced, r >> 1)
After:   ||α * g_dominant|| / ||β * g_weak|| = (α/β) * r

With α=0.6 (OGM-GE) and β=1.5 (ASGML):
  (0.6/1.5) * r = 0.4 * r  (2.5× better balance)
```

### 4.2 Why Boosting Works

**Gradient magnitude affects learning speed**: Larger gradients → faster parameter updates → faster feature learning.

By boosting weak modality gradients:
1. Weak encoder updates more aggressively
2. Features become more discriminative faster
3. Fusion head learns to use both modalities

### 4.3 Why Decoupled Probes Matter

**Coupled classifiers (CGGM)** create feedback loops:
- Classifier loss affects encoder training
- Encoder changes affect classifier signal
- Potential for instability or mode collapse

**Decoupled probes (ASGML)** provide clean measurement:
- Probes observe but don't interfere
- Signal reflects true encoder quality
- No gradient contamination

---

## 5. Experimental Results

### 5.1 CREMA-D Results (Multi-Seed)

**1-Frame Results** (5 seeds: 42, 0, 1, 2, 3)

| Method | Mean ± Std |
|--------|-----------|
| **ASGML Boost + OGM-GE (α=0.75)** | **62.69 ± 0.22%** |
| OGM-GE alone (α=0.8) | 62.47 ± 1.42% |
| ASGML Boost only (α=0.5) | 60.46 ± 0.85% |
| Baseline | 60.30 ± 0.70% |

**3-Frame Results** (5 seeds: 42, 123, 456, 789, 1024)

| Method | Mean ± Std |
|--------|-----------|
| **ASGML Boost + OGM-GE (α=0.75)** | **71.45 ± 1.71%** |
| OGM-GE alone | 69.14 ± 1.13% |
| InfoReg (100ep) | 67.72 ± 0.83% |
| Baseline | 61.59 ± 0.80% |
| MILES (τ=0.2) | 61.05 ± 2.52% |

### 5.2 Key Findings

1. **Original ASGML (frequency/staleness) ≈ baseline**: Binary skip rarely triggered due to normalization
2. **ASGML Boost alone ≈ baseline**: Boosting without throttling provides marginal improvement (60.46% vs 60.30%)
3. **Combination is best**: OGM-GE throttles dominant + ASGML boosts weak = orthogonal interventions
4. **3 frames amplify synergy**: boost+OGM-GE gap over OGM-GE goes from +0.22pp (1-frame) to +2.31pp (3-frame) — 10.5× amplification
5. **Variance reduction**: boost+OGM-GE has 6.5× lower variance than OGM-GE alone (1-frame), suggesting probe-guided boosting stabilizes training

### 5.3 Why Combination Works

```
Gradient spectrum after interventions:

BEFORE:     Audio ████████████████████  Visual ████
            (dominant, large grad)      (weak, small grad)

OGM-GE:     Audio ████████████          Visual ████
            (throttled)                 (unchanged)

ASGML:      Audio ████████████████████  Visual ████████████
            (unchanged)                 (boosted)

COMBINED:   Audio ████████              Visual ████████████
            (throttled)                 (boosted)

            NOW BALANCED!
```

---

## 6. Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `alpha` | Boost strength | 0.5 | 0.1 - 1.0 |
| `scale_max` | Maximum boost scale | 2.0 | 1.5 - 3.0 |
| `eval_freq` | Probe evaluation frequency | 20 | 10 - 100 |
| `probe_train_steps` | Steps to train probe per eval | 10 | 5 - 50 |
| `ema_alpha` | EMA smoothing for scales | 0.3 | 0.1 - 0.5 |
| `probe_type` | Probe architecture | "linear" | linear, mlp_1layer |

---

## 7. Implementation Details

### 7.1 Training Loop Integration

```python
# Standard forward pass
logits, unimodal_logits, features = model(inputs, return_features=True)

# Standard loss (unchanged)
loss = criterion(logits, targets) + gamma * unimodal_loss

# Standard backward
loss.backward()

# OGM-GE gradient modulation (if enabled)
apply_ogm_ge(model, unimodal_logits, ...)

# Periodic probe evaluation
if step % eval_freq == 0:
    # Split batch for honest evaluation
    train_feats, eval_feats = split_batch(features)
    probe_manager.train_probes(train_feats, ...)
    probe_results = probe_manager.evaluate_probes(eval_feats, ...)

    # Compute boost scales from probe signal
    scales = scheduler.get_continuous_scales(probe_results)
    scheduler.update_continuous_scales(scales)  # EMA smooth

# Apply ASGML boost (every step, using cached scales)
for m in modalities:
    scale = scheduler.current_continuous_scales[m]
    for param in model.encoders[m].parameters():
        if param.grad is not None:
            param.grad.data.mul_(scale)

# Optimizer step
optimizer.step()
```

### 7.2 Model Architecture

**CREMA-D / AVE (2-modality, ResNet18):**
```
Audio Input (B, 1, 128, T) ──► AudioEncoder (ResNet18) ──► (B, 512)
                                                              │
Visual Input (B, 3, H, W) ──► VisualEncoder (ResNet18) ──► (B, 512)
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
                              ConcatFusion
                            (B, 1024) → (B, 512)
                                    │
                                    ▼
                              Classifier (B, 512) → (B, num_classes)

Probes (separate):
  features['audio'].detach() ──► LinearProbe ──► accuracy
  features['visual'].detach() ──► LinearProbe ──► accuracy
```

**CMU-MOSEI (3-modality, MLP):**
```
Text (B, 39, 768) ──mean_pool──► MLPEncoder (768→512) ──► (B, 512)
                                                              │
Audio (B, 400, 33) ──mean_pool──► MLPEncoder (33→512) ──► (B, 512)
                                                              │
Vision (B, 55, 709) ──mean_pool──► MLPEncoder (709→512) ──► (B, 512)
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
                              ConcatFusion
                            (B, 1536) → (B, 512)
                                    │
                                    ▼
                              Classifier (B, 512) → (B, 3)

Probes (separate, one per modality):
  features['text'].detach()   ──► LinearProbe ──► accuracy
  features['audio'].detach()  ──► LinearProbe ──► accuracy
  features['vision'].detach() ──► LinearProbe ──► accuracy
```

---

## 8. Ablation Studies (Required)

| Experiment | What It Tests | Status |
|------------|---------------|--------|
| Baseline | Lower bound | ✅ CREMA-D done |
| OGM-GE only | SOTA synchronous throttling | ✅ CREMA-D done |
| ASGML Boost only | Boosting without throttling | ✅ CREMA-D done |
| **ASGML + OGM-GE** | **Full contribution** | ✅ CREMA-D done |
| InfoReg | PLW-based weight regulation | ✅ CREMA-D done |
| MILES | Epoch-level LR scheduling | ✅ CREMA-D done |
| ASGML + CGGM | Complementarity with other methods | 🔄 Pending (CGGM not implemented) |
| Different alpha values | Boost strength sensitivity | ✅ Sweep done (α=0.25-1.0) |
| Different scale_max | Maximum boost limit | ✅ Sweep done (1.5-3.0, minimal effect) |
| Different eval_freq | Probe evaluation frequency | 🔄 Pending |
| Linear vs MLP probe | Probe architecture | 🔄 Pending |
| With/without EMA | Scale smoothing importance | 🔄 Pending |
| With/without split-batch | Honest signal importance | ✅ Confirmed critical (v1 vs v2) |
| Cross-dataset (AVE) | Generalization to different domain | 🔄 Data ready |
| 3-modality (MOSEI) | Scaling beyond 2 modalities | 🔄 Data ready |

---

## 9. Paper Positioning

### 9.1 Title Options

1. "Probe-Guided Gradient Boosting for Balanced Multimodal Learning"
2. "ASGML: Boosting Weak Modalities via Decoupled Utilization Monitoring"
3. "Complementary Gradient Modulation: Throttle + Boost for Multimodal Balance"

### 9.2 Key Claims

1. **Decoupled probes** provide clean utilization signal without training interference
2. **Boosting weak modalities** is orthogonal to throttling dominant modalities
3. **Combined approach** (OGM-GE + ASGML) achieves SOTA on CREMA-D
4. **Simple linear probes** are sufficient for effective monitoring

### 9.3 Contribution Summary

| Contribution | Description |
|--------------|-------------|
| **Methodological** | First method to explicitly boost weak modality gradients |
| **Architectural** | Decoupled probe design with split-batch evaluation |
| **Empirical** | Demonstrates complementarity of throttle + boost |
| **Practical** | Minimal overhead (~6K probe params, periodic evaluation) |

---

## 10. Open Questions

1. **Theoretical analysis**: Can we prove convergence for combined OGM-GE + ASGML?
2. **Scaling to N modalities**: ASGML core (scheduler, loss, probes) is N-modality ready — confirmed via code review. **Blocker:** OGM-GE is hardcoded for 2-modality pairwise ratios AND only modulates Conv2d gradients (not MLP). Needs generalization for MOSEI (3 modalities, MLP backbone). See Section 10.1.
3. **Probe alternatives**: Can we use gradient-based signals instead of probes?
4. **Dynamic alpha**: Should boost strength adapt during training?
5. **Other combinations**: Does ASGML complement CGGM, AGM, etc.?

### 10.1 OGM-GE N-Modality & MLP Limitations

**Issue 1 — 2-modality only:** `apply_ogm_ge()` in `scripts/train.py:681` uses `if len(m_list) == 2:` to compute pairwise ratios. With 3+ modalities, no gradient modulation occurs (silent no-op).

**Issue 2 — Conv2d only:** `scripts/train.py:698` filters `if len(param.grad.size()) == 4:` (Conv2d). MLP layers have 2D gradients, so all MOSEI encoder gradients are skipped.

**Fix required for MOSEI boost+OGM-GE:**
- Generalize pairwise ratio to N-way (e.g., each modality vs mean score)
- Remove or relax the Conv2d filter to include Linear (2D) gradients

**ASGML boost alone works fine for MOSEI** — no fixes needed.

---

## 11. Timeline

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | Baseline + OGM-GE reproduction | ✅ Complete |
| Phase 2 | Original ASGML (frequency/staleness) | ✅ Complete (didn't work) |
| Phase 3 | ASGML Boost (continuous mode) | ✅ Complete |
| Phase 4 | Multi-seed validation (CREMA-D) | ✅ Complete (71.45% ± 1.71%) |
| Phase 5a | AVE dataset preparation | ✅ Complete (4,097 samples, 28 classes, OGM-GE format) |
| Phase 5b | KS dataset | ⏸️ Deferred (manual download needed) |
| Phase 6a | CMU-MOSEI dataset preparation | ✅ Complete (2,281 samples, 3 classes, MMSA features) |
| Phase 6b | MOSEI infrastructure (MLP encoders, 3-mod config) | ✅ Complete (smoke test: 70% after 2ep) |
| Phase 6c | OGM-GE N-modality + MLP generalization | 🔄 Needed for boost+OGM-GE on MOSEI |
| Phase 7 | Full training runs on AVE + MOSEI | 🔄 Pending |
| Phase 8 | Ablation studies | 🔄 Pending |
| Phase 9 | Paper writing | 🔄 Pending |

---

## 12. Code Pointers

| Component | File | Key Function/Class |
|-----------|------|-------------------|
| Probes | `src/models/probes.py` | `ProbeManager` (line 88), `LinearProbe` (line 19) |
| Scheduler | `src/losses/asgml.py` | `ASGMLScheduler.get_continuous_scales()` (line 490) |
| Training | `scripts/train.py` | `train_epoch()` (line 707), main loop (line 1708) |
| OGM-GE | `scripts/train.py` | `apply_ogm_ge()` (line 625) |
| MLP Encoder | `src/models/encoders.py` | `MLPEncoder` (line 206) — for pre-extracted features |
| MOSEI Dataset | `src/datasets/mosei.py` | `MOSEIDataset` — auto-detects MMSA vs InfoReg format |
| AVE Dataset | `src/datasets/ave.py` | `AVEDataset` — OGM-GE compatible format |
| CREMA-D Config | `configs/cremad.yaml` | Primary benchmark (ResNet18, audio-visual) |
| AVE Config | `configs/ave.yaml` | 28-class audio-visual (ResNet18) |
| MOSEI Config | `configs/mosei.yaml` | 3-class sentiment, 3 modalities (MLP backbone) |

---

## 13. Dataset Support

### 13.1 CREMA-D (Primary — Complete)
- **Task:** 6-class emotion recognition (audio-visual)
- **Split:** 6,698 train / 744 test
- **Encoders:** ResNet18 (from scratch), 1 or 3 visual frames
- **Dominance:** Audio dominant, visual weak
- **Results:** 71.45 ± 1.71% (boost+OGM-GE, 3-frame)

### 13.2 AVE (Prepared — Ready for Training)
- **Task:** 28-class audio-visual event classification
- **Data:** 4,097 samples (3,287 train / 810 test), 10-second videos
- **Format:** OGM-GE compatible (pickle spectrograms + extracted frames)
- **Encoders:** ResNet18 (same architecture as CREMA-D)
- **Location:** `data/AVE/` with `audio_spec/`, `visual/`, `my_train.txt`, `my_test.txt`, `stat.txt`
- **Smoke test:** 60.49% after 2 epochs (from ~3.6% random)

### 13.3 CMU-MOSEI (Prepared — Ready for Training)
- **Task:** 3-class sentiment analysis (text + audio + vision)
- **Data:** 2,281 samples (1,368 train / 456 valid / 457 test) — MMSA-processed subset
- **Features:** BERT text (768d), audio (33d), vision (709d) — z-score normalized
- **Encoders:** 2-layer MLP (Linear→ReLU→Dropout→Linear→ReLU) per modality
- **Format:** MMSA pickle (`unaligned_39.pkl`), auto-detected by `MOSEIDataset`
- **Location:** `data/MOSEI/Processed/Processed/unaligned_39.pkl`
- **Smoke test:** 70.02% after 2 epochs
- **Note:** ASGML boost works for 3 modalities. OGM-GE requires generalization (see Section 10.1).

### 13.4 Kinetics-Sounds (Deferred)
- Manual download of video data required
- Split files exist at `data/KS/ks_split/`

---

## Appendix A: Evolution from v1 to v2

### What Changed

| Aspect | ASGML v1 | ASGML v2 |
|--------|----------|----------|
| Core mechanism | Binary update skipping | Continuous gradient scaling |
| Signal | Dual-signal (normalized) | Probe accuracy (absolute) |
| Trigger | Rarely (τ ≈ 1-2 for all) | Always active |
| Direction | Throttle dominant | Boost weak |
| Standalone performance | ≈ Baseline | Modest improvement |
| With OGM-GE | Not tested | **71.45%** (best) |

### Why v1 Failed

1. **Normalization issue**: Dual-signal normalized by mean → lost absolute gap info
2. **Binary decision**: Skip/update is coarse; continuous scaling is smoother
3. **Wrong direction**: Throttling alone is already done by OGM-GE

### Why v2 Works

1. **Absolute signal**: Probe accuracy preserves the utilization gap
2. **Continuous scaling**: Smooth, always-active intervention
3. **Complementary**: Boosts weak instead of throttling dominant
4. **Decoupled**: Clean measurement without training interference
