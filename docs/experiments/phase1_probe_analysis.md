# Phase 1 Probe Diagnostics Analysis

**Date:** 2026-02-02
**Experiments:** Baseline, Frequency k=2, Staleness τ=2

---

## Overview

This document analyzes the per-modality probe diagnostics from Phase 1 experiments to understand how each ASGML mode affects modality learning dynamics.

---

## Test Set Probe Accuracies

Probe accuracies on the **test set** measure how discriminative each modality's learned features are for the classification task.

### Summary Statistics

| Experiment | Audio Probe (Avg) | Visual Probe (Avg) | Avg Gap | Interpretation |
|------------|-------------------|--------------------|---------|-|
| Baseline | 42.65% | 29.60% | **13.05%** | Reference imbalance |
| **Frequency k=2** | 39.67% | 30.03% | **9.64%** | Best balance |
| Staleness τ=2 | 44.10% | 30.29% | **13.80%** | Slight increase |

### Key Finding: Frequency Mode Achieves Best Modality Balance

The frequency k=2 mode reduced the audio-visual gap from 13.05% to **9.64%** (26% reduction).

- Audio probe accuracy **decreased** (42.65% → 39.67%): Audio learning was slowed
- Visual probe accuracy **increased** (29.60% → 30.03%): Visual had more opportunity
- **This is the intended ASGML effect** — but it hurt overall performance

---

## Epoch-by-Epoch Probe Trajectories

### Baseline
```
Epoch   Audio Probe   Visual Probe   Gap
1       31.41%        23.22%         8.19%
5       43.09%        31.01%         12.08%
10      42.15%        29.13%         13.02%
13      46.17%        29.53%         16.64%
```
**Pattern:** Gap increases over training — audio dominates more as training progresses

### Frequency k=2
```
Epoch   Audio Probe   Visual Probe   Gap
1       35.57%        23.62%         11.95%
5       34.36%        29.26%         5.10%   <- Best balance
10      41.61%        32.35%         9.26%
13      41.88%        31.54%         10.34%
```
**Pattern:** Gap reduced early, then stable — frequency skipping helps visual catch up

### Staleness τ=2
```
Epoch   Audio Probe   Visual Probe   Gap
1       34.63%        23.22%         11.41%
5       39.60%        31.54%         8.05%
10      47.92%        27.25%         20.67%  <- High gap
13      50.74%        32.62%         18.12%
```
**Pattern:** Gap increases sharply mid-training — staleness may not effectively slow audio

---

## Per-Modality Learning Dynamics

### Audio Encoder Performance

| Metric | Baseline | Frequency k=2 | Staleness τ=2 |
|--------|----------|---------------|---------------|
| Min Probe Acc | 31.41% | 34.36% | 34.63% |
| Max Probe Acc | 48.86% | 44.83% | **52.48%** |
| Final Probe Acc | 46.17% | 41.88% | 50.74% |

**Observations:**
- Frequency mode successfully **reduced** audio probe performance (max 44.83% vs 48.86%)
- Staleness mode **increased** audio probe performance (max 52.48%)
- This explains why staleness achieved better overall F1 but worse balance

### Visual Encoder Performance

| Metric | Baseline | Frequency k=2 | Staleness τ=2 |
|--------|----------|---------------|---------------|
| Min Probe Acc | 23.22% | 23.62% | 23.22% |
| Max Probe Acc | 35.03% | 34.50% | 33.29% |
| Final Probe Acc | 29.53% | 31.54% | 32.62% |

**Observations:**
- All methods achieved similar visual probe performance
- Frequency and staleness slightly improved final visual probe acc
- Visual learning appears bottlenecked regardless of audio intervention

---

## Gradient Dynamics

### Late-Phase Gradient Norms

| Experiment | Audio Grad Norm | Visual Grad Norm | Ratio (A/V) |
|------------|-----------------|------------------|-------------|
| Baseline | 145,989 | 539,047 | 0.27 |
| Frequency k=2 | 213,196 | 536,778 | 0.40 |
| Staleness τ=2 | 118,592 | 280,912 | 0.42 |

**Observations:**
- Visual gradients are larger than audio in all experiments
- This suggests visual encoder has more to learn but may be receiving suppressed signal
- Staleness mode has smallest overall gradient magnitudes (more converged)

---

## Interpretation & Insights

### 1. The Frequency-Balance Trade-off

Frequency k=2 successfully improved modality balance:
- Reduced audio-visual probe gap by 26% (13.05% → 9.64%)
- Audio probe accuracy decreased (intended effect)
- But overall F1 also decreased (58.66% vs 60.13% baseline)

**Conclusion:** k=2 (skipping 50% of audio updates) is too aggressive. The audio encoder is a critical contributor, and halving its updates hurts the joint representation.

### 2. Staleness Improves F1 Without Improving Balance

Staleness τ=2 achieved best F1 (61.72%) but:
- Did NOT reduce the audio-visual gap (13.80% vs 13.05% baseline)
- Actually increased audio probe performance

**Hypothesis:** Staleness acts more as a regularizer than a balancing mechanism. The delayed gradients may:
- Reduce overfitting to dominant modality patterns
- Smooth the optimization landscape
- But not fundamentally change the modality learning dynamics

### 3. Visual Encoder Appears Bottlenecked

Across all experiments:
- Visual probe accuracy capped around 30-35%
- Visual gradients are consistently larger than audio
- Interventions on audio had limited effect on visual learning

**Possible causes:**
- Visual encoder architecture may be suboptimal for CREMA-D
- Visual features may be inherently less discriminative for emotion
- Visual preprocessing may need improvement

---

## Recommendations for Phase 2

### 1. Softer Frequency Ratios
- Try k=3 (skip 33%) and k=4 (skip 25%)
- Goal: Better balance without sacrificing audio learning

### 2. Adaptive ASGML
- Let probes dynamically adjust staleness/frequency
- Trigger only when utilization gap exceeds threshold

### 3. Investigate Visual Encoder
- Try different visual backbones (ViT, EfficientNet)
- Check visual preprocessing pipeline
- Consider visual-specific augmentation

### 4. Combined Approaches
- OGM-GE + ASGML (gradient modulation + temporal separation)
- May achieve both balance AND improved performance

---

## Appendix: Data Sources

All data extracted from TensorBoard logs:
- `outputs/phase1_baseline/cremad_baseline_20260201_230340/tensorboard/`
- `outputs/phase1_frequency_k2/cremad_frequency_20260201_230345/tensorboard/`
- `outputs/phase1_staleness_t2/cremad_staleness_20260201_230353/tensorboard/`

Metrics logged:
- `test/probe_acc_audio` - Audio probe accuracy on test set
- `test/probe_acc_visual` - Visual probe accuracy on test set
- `test/utilization_gap` - Difference between probe accuracies
- `train/grad_norm_audio` - Audio encoder gradient L2 norm
- `train/grad_norm_visual` - Visual encoder gradient L2 norm
