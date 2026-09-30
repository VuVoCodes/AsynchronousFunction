---
name: project-camready-mosei-correction
description: 2026-09-25 audit of the genuine-CMU-MOSEI column swap in the camera-ready — what verified clean, and the four residual honesty/context issues
metadata:
  type: project
---

The CH-SIMS→genuine-CMU-MOSEI correction (camera-ready, 2026-09-25) is mechanically clean. Independently verified:

- Split sizes 16,265/1,869/4,643 match `data/MOSEI_true/mosei_senti_data.pkl` exactly.
- |0.5| 3-class bucketing matches `src/datasets/mosei.py` (InfoReg-format branch, threshold 0.5).
- Test-set majority class is 41.31% (neutral, 1918/4643), so 62% is ~+21 pp over trivial and the model is functioning. Neutral is the *majority* class under |0.5| bucketing — a non-standard MOSEI setup worth stating.
- No stale submitted-version numbers survive anywhere (grepped 70.42, 72.47, 72.43, 69.80, 68.05, 70.99, 1.44, 0.178, "7 classes", "BERT (768").
- Welch tests on the per-seed values: PGGB+OGM-GE vs G-Blend **p=0.89**, vs Baseline p=0.32. The column is a statistical tie except CGGM.

Residual issues (recurring, likely to reappear):

1. **Bolding over-reads ties.** Table 1 bolds a 0.04 pp lead (p=0.89); Appendix Table 9 (MSLR) bolds a **0.01 pp** lead (62.24 vs 62.23). The paper already has a "sub-σ" annotation convention — apply it in Table 1 or bold jointly. Same 0.04-pp-as-a-win pattern also underlies "PGGB alone is best on all four low-imbalance benchmarks" (Twitter15 +0.04, Sarcasm +0.04, KS +0.12), so the rhetorical treatment is inconsistent across columns.
2. **Metric context.** The 3-class disclosure sits only in the Table 1 caption. No chance/majority reference, no statement that GloVe+2-layer-MLP numbers are not comparable to published MOSEI Acc-2 (~82-86%) / Acc-7 (~50-54%). And **CMU-MOSI is binarized** (`src/datasets/cmu_mosi.py`: `labels_raw > 0`) yet its metric is stated nowhere — singling MOSEI out as 3-class implies MOSI is standard.
3. **δ=0.218 ± 0.005 is a double-edged gain.** It removes the fragile 0.178±0.086 straddle (strengthens the >0.15 categorization), but by Eq. (7) the weakest modality gets w=1 whenever δ >> ε, so Prop. 2's bound at Δ=0.218 is |s-1| ≤ ~α: the boost ran at **full strength** across all 5 seeds and produced +0.17 pp. The flat column is therefore a clean negative result, not attenuation, and the paper does not own this.
4. **"F1-macro confirms the Table 1 ranking" (App. E)** is unsupportable for MOSEI: 0.17 pp lead inside a 0.94 pp column.

**How to apply:** Do not re-litigate the correction's arithmetic — it checks out. Focus review effort on the bolding convention, metric disclosure symmetry with MOSI, and whether the text-sufficiency framing has been replaced by the measured Adam-absorption explanation (see [[project_camready_unapplied_commitments]]).
