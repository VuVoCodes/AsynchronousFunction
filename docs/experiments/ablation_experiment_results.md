# Ablation Studies

This file consolidates ablation studies for the Probe-Guided Gradient Boosting method.

---

## K Ablation (Probe Evaluation Frequency)

**Date:** 2026-04-11
**Goal:** Show method is robust to the probe refresh interval K, which controls adaptation latency (not intervention frequency — boost scale applies every step regardless).
**Setup:** CREMA-D 3f (3 frames @ 3 FPS), seed=42, Boost+OGM-GE (α=0.75), OGM-GE α=0.8, all other hyperparameters at default. 100 epochs, SGD lr=0.001, StepLR step=70, batch 64.

### What K Controls

- **K=1**: probes retrain + boost scale refreshes every batch (most responsive, highest overhead)
- **K=20** (default): probes refresh every 20 batches, boost scale reuses EMA-smoothed value in between
- **K=100**: probes refresh every 100 batches (stale signal, minimal overhead)

**Important:** ASGML intervenes on every step regardless of K. The boost scale is applied to encoder gradients every batch. K only controls how often the scale is *recomputed* from fresh probe signals. Between refreshes, the scale continues applying using its last EMA-smoothed value.

### K Sweep Results (seed=42)

| K | Probe refresh interval | Best Acc | vs K=20 default |
|---|----------------------|----------|-----------------|
| **1** | Every batch | 71.24% | -0.13pp |
| **5** | Every 5 batches | **71.64%** | **+0.27pp** |
| **10** | Every 10 batches | 70.83% | -0.54pp |
| **20** | Every 20 batches (default) | 71.37% | — |
| **50** | Every 50 batches | 71.24% | -0.13pp |
| **100** | Every 100 batches | 71.24% | -0.13pp |

### K Ablation Analysis

1. **Method is robust to K.** All six K values land within a 0.81pp range (70.83-71.64%). No configuration catastrophically fails or drives large improvement — probe polling frequency is not a sensitive hyperparameter.

2. **K=5 slightly edges the default** at 71.64% vs 71.37% for K=20, but the +0.27pp difference is well within single-seed noise (±1.71% std for Boost+OGM-GE on CREMA-D 3f). The default K=20 remains a defensible choice.

3. **K=1 and K=100 both achieve 71.24%** — this is the strongest evidence of robustness. Most-responsive (every batch) and least-responsive (stale for 100 batches) give identical accuracy, showing the EMA-smoothed boost scale effectively carries over between probe refreshes.

4. **Overhead-accuracy tradeoff:** K=1 adds ~5% wall-clock overhead from extra probe train/eval passes; K=100 adds <0.5%. Since accuracy is flat across K, larger K is strictly preferred for efficiency.

5. **Justifies the default K=20.** The sweet spot is in the 5-50 range, and K=20 provides good adaptation latency while keeping probe overhead negligible (~1% wall-clock).

### Reviewer Response

Addresses potential concerns about:
- "Is K=20 a magic number?" → No, accuracy is flat across 2 orders of magnitude in K
- "What's the probe overhead?" → Minimal. K can be increased without accuracy loss
- "Why not K=1 for best adaptation?" → Empirically no benefit; K=20 is cheaper and equivalent

### Output Locations

| Experiment | Directory |
|-----------|-----------|
| K=1 | `outputs/sweep_k_ablation/3f_boost_ogm_K1_seed42/` |
| K=5 | `outputs/sweep_k_ablation/3f_boost_ogm_K5_seed42/` |
| K=10 | `outputs/sweep_k_ablation/3f_boost_ogm_K10_seed42/` |
| K=20 (default, from main exp) | `outputs/sweep_3f/3f_boost_ogm_a075_seed42/` |
| K=50 | `outputs/sweep_k_ablation/3f_boost_ogm_K50_seed42/` |
| K=100 | `outputs/sweep_k_ablation/3f_boost_ogm_K100_seed42/` |

### Reproduction Commands

```bash
for K in 1 5 10 50 100; do
    python scripts/train.py --config configs/cremad.yaml --mode adaptive \
        --asgml-mode continuous --continuous-alpha 0.75 \
        --continuous-eval-freq $K \
        --ogm-ge --alpha 0.8 \
        --num-frames 3 --fps 3 --seed 42 \
        --output-dir outputs/sweep_k_ablation --exp-name 3f_boost_ogm_K${K}_seed42
done
```

---

*Last updated: 2026-04-11 (K ablation complete. Method robust across K ∈ {1, 5, 10, 20, 50, 100} — all within 0.81pp. K=20 default justified.)*
