# Missing Experiments & Architecture Notes

Last updated: 2026-04-11

## Table 1 vs Actual Data: What's in the paper now

Current Table 1 has 10 rows x 6 datasets = 60 cells.
- Filled: 38 cells
- Missing (---): 22 cells

### Numbers currently in Table 1

| Method | CREMA-D | AVE | KS | MOSI | MOSEI | BraTS |
|--------|---------|-----|-----|------|-------|-------|
| Baseline | 61.59 | 86.54 | 79.05 | 72.42 | 70.42 | 85.77 |
| OGM-GE | 69.14 | 86.96 | 77.25 | **72.68** | **72.47** | --- |
| MMPareto | 65.51 | 86.37 | 78.21 | --- | --- | --- |
| AGM | 57.42 | 84.42 | 77.85 | --- | --- | --- |
| G-Blend | 61.10 | 87.09 | 77.75 | --- | --- | --- |
| CGGM | 50.22 | 76.72 | 73.18 | 59.45 | 68.05 | 81.13 |
| InfoReg | 67.72 | --- | --- | --- | --- | --- |
| MILES | 61.05 | --- | --- | --- | --- | --- |
| Boost only | 62.72 | **87.41** | **79.17** | 71.89 | 69.80 | **85.98** |
| Boost+OGM-GE | **71.45** | 87.23 | 77.33 | 72.60 | 72.43 | --- |

---

## Architecture Compatibility Matrix

This is the KEY constraint. Not all methods can run on all datasets.

### Encoder types per dataset

| Dataset | Encoder Type | Input | Notes |
|---------|-------------|-------|-------|
| CREMA-D | ResNet18 CNN (from scratch) | Raw audio spectrograms + video frames | SGD, no pretrain |
| AVE | ResNet18 CNN (ImageNet pretrained) | Raw audio spectrograms + video frames | SGD |
| KS | ResNet18 CNN (ImageNet pretrained) | Raw audio spectrograms + video frames | SGD |
| MOSEI | 2-layer MLP (512 hidden) | Pre-extracted: BERT 768d + COVAREP + FACET | Adam |
| MOSI | 2-layer MLP (512 hidden) | Pre-extracted: GloVe 300d + COVAREP + FACET | Adam |
| BraTS | ResNet101 (DeepLab v3+) | 3D MRI volumes (FLAIR, T1ce, T1, T2) | SGD cosine LR, Dice+CE |

### Method architecture requirements

| Method | Works with CNN? | Works with MLP? | Works with DeepLab? | Implementation constraint |
|--------|----------------|-----------------|---------------------|--------------------------|
| OGM-GE | Yes | Yes | Yes (needs adaptation) | Operates on Conv2d/Linear weight grads; should work everywhere |
| MMPareto | Yes | Yes | Yes (needs adaptation) | Architecture-agnostic; needs per-modality param groups + MinNormSolver |
| AGM | Yes | Yes | Yes (needs adaptation) | Architecture-agnostic; loss-based scaling |
| G-Blend | Yes | Yes | Yes (needs adaptation) | Architecture-agnostic; needs per-modality val loss computation |
| CGGM | Yes | Yes | Yes | Already runs on all 6 datasets |
| **InfoReg** | **No** | **Yes** | **No** | **Requires MLP on pre-extracted features. Fisher trace computed via gradient norms of MLP layers. NOT designed for CNN/DeepLab.** |
| **MILES** | **Yes** | **Yes** | **Unclear** | Flexible algorithm, but released implementation targets audio-visual ResNet18. Needs unimodal classifiers + per-modality LR groups. |
| OPM | Yes | Yes | Unclear | Requires ConcatFusion with decomposable weight matrix |
| Boost/PGGB | Yes | Yes | Yes | Already runs on all 6 datasets |

---

## Missing Experiments: Detailed Analysis

### TIER 1: HIGH PRIORITY (reviewer will immediately notice)

#### 1. BraTS -- OGM-GE (5 seeds)
- **Why missing:** Never run
- **Architecture fit:** OGM-GE works on Conv2d/Linear grads. ResNet101 has both. Should work.
- **Adaptation needed:** Port `apply_ogm_ge` to DeepLab pipeline. Need to compute gradient magnitude ratio across 4 encoders (currently 2-modality formulation in code, but N-modality generalization exists).
- **Effort:** Medium. Main work is ensuring the 4-encoder OGM-GE ratio computation is correct.
- **Why it matters:** Table 1 shows OGM-GE on 5/6 datasets but not BraTS. Reviewer asks: "why skip OGM-GE on your segmentation benchmark?"

#### 2. BraTS -- Boost+OGM-GE (5 seeds)
- **Why missing:** Depends on BraTS OGM-GE working first
- **Architecture fit:** Same as above
- **Adaptation needed:** Same OGM-GE port + combine with existing boost
- **Effort:** Low (once OGM-GE works on BraTS)
- **Why it matters:** Our best config not tested on BraTS. Currently only boost-only shown.

#### 3. MOSEI -- MMPareto (5 seeds)
- **Why missing:** MMPareto only run on audio-visual datasets so far
- **Architecture fit:** Architecture-agnostic. Works with MLP encoders.
- **Adaptation needed:** Minimal. MMPareto needs per-modality parameter groups and unimodal losses. MLP encoders already have per-modality params. Need to add 3-modality unimodal loss computation.
- **Effort:** Low-Medium
- **Why it matters:** MMPareto is ICML 2024, a key baseline. Missing on both sentiment datasets.

#### 4. MOSI -- MMPareto (5 seeds)
- **Same as MOSEI MMPareto.** Once MOSEI works, MOSI is trivial (same pipeline, different features).

### TIER 2: MEDIUM PRIORITY (strengthens paper, not fatal if missing)

#### 5-6. AVE/KS -- InfoReg
- **Architecture fit:** **INCOMPATIBLE.** InfoReg requires MLP encoders on pre-extracted features. AVE/KS use ResNet18 CNNs on raw audio+video.
- **Adaptation needed:** Would need to either (a) pre-extract ResNet18 features and train InfoReg on those, or (b) rewrite InfoReg's Fisher trace computation for CNN layers.
- **Effort:** High. Not a simple port.
- **Decision:** **DROP.** Paper already says "InfoReg evaluated on CREMA-D only, as their released implementation targets audio-visual classification." This is honest but slightly misleading since InfoReg could theoretically work on MOSEI/MOSI (MLP encoders).
- **Alternative:** Run InfoReg on MOSEI/MOSI instead (where it's actually designed to work).

#### 7-8. AVE/KS -- MILES
- **Architecture fit:** MILES is more flexible than InfoReg. Uses per-modality LR adjustment based on utilization rates. Should work with ResNet18.
- **Adaptation needed:** Need unimodal classifiers for each modality. Per-modality optimizer groups. The conditional utilization rate computation needs unimodal accuracy tracking.
- **Effort:** Medium
- **Decision:** Worth doing if time permits. MILES on AVE/KS would show whether LR-based methods help on low-imbalance data.

#### 9-10. MOSEI/MOSI -- AGM
- **Architecture fit:** Architecture-agnostic. Works with MLP.
- **Adaptation needed:** Minimal. AGM uses loss-based exponential scaling. Need to compute per-modality CE scores on 3 modalities.
- **Effort:** Low
- **Decision:** Worth doing. AGM underperforms on CREMA-D (57.42%), but behavior on sentiment data is unknown.

#### 11-12. MOSEI/MOSI -- G-Blend
- **Architecture fit:** Architecture-agnostic. Works with MLP.
- **Adaptation needed:** Need per-modality validation loss computation for 3 modalities. OG-ratio calculation needs val set evaluation each epoch.
- **Effort:** Low-Medium
- **Decision:** Worth doing. G-Blend matched baseline on CREMA-D (61.10%), but it outperforms on AVE (87.09%). Behavior on sentiment unclear.

#### 13. BraTS -- MMPareto
- **Architecture fit:** Architecture-agnostic.
- **Adaptation needed:** Need 4-modality per-encoder param groups, 4 unimodal losses, MinNormSolver on 4 gradients. Also needs DeepLab-compatible unimodal classifiers.
- **Effort:** Medium-High (4 modalities + segmentation loss + DeepLab pipeline)
- **Decision:** Low priority. Nice to have but complex adaptation.

#### 14. BraTS -- AGM
- **Architecture fit:** Architecture-agnostic.
- **Adaptation needed:** 4-modality loss ratio computation with Dice+CE loss.
- **Effort:** Medium
- **Decision:** Low priority.

### TIER 3: LOW PRIORITY (skip unless time permits)

#### 15-16. MOSEI/MOSI -- InfoReg
- **Architecture fit:** **COMPATIBLE.** InfoReg was designed for MLP on pre-extracted features. MOSEI/MOSI use exactly this setup.
- **Adaptation needed:** Minimal. InfoReg's Fisher trace + PLW detection should work directly.
- **Effort:** Low
- **Decision:** Actually this SHOULD be higher priority. InfoReg on MOSEI/MOSI is its natural habitat. Currently we only show InfoReg on CREMA-D where it's architecturally mismatched (we use ResNet18 CNNs, not its intended MLP setup). **Consider promoting to TIER 2.**

#### 17-18. MOSEI/MOSI -- MILES
- **Architecture fit:** Compatible with MLP encoders.
- **Effort:** Low-Medium
- **Decision:** Nice to have.

#### 19-20. BraTS -- G-Blend, InfoReg, MILES
- **Effort:** High (DeepLab adaptation for each)
- **Decision:** Skip. BraTS already has Baseline, CGGM, and Boost. Adding OGM-GE and Boost+OGM-GE (Tier 1) is sufficient.

---

## Recommended Action Plan

### Must-do (before submission)

| # | Experiment | Seeds | GPU hours (est.) | Effort |
|---|-----------|-------|-----------------|--------|
| 1 | BraTS OGM-GE | 5 | ~25h (5x5h) | Medium (port to DeepLab) |
| 2 | BraTS Boost+OGM-GE | 5 | ~25h | Low (once #1 works) |
| 3 | MOSEI MMPareto | 5 | ~5h (5x1h) | Low-Medium |
| 4 | MOSI MMPareto | 5 | ~5h | Low (same as #3) |

### Should-do (if time permits)

| # | Experiment | Seeds | GPU hours (est.) | Effort |
|---|-----------|-------|-----------------|--------|
| 5 | MOSEI AGM | 5 | ~5h | Low |
| 6 | MOSI AGM | 5 | ~5h | Low |
| 7 | MOSEI G-Blend | 5 | ~5h | Low-Medium |
| 8 | MOSI G-Blend | 5 | ~5h | Low-Medium |
| 9 | MOSEI InfoReg | 5 | ~5h | Low (natural habitat!) |
| 10 | MOSI InfoReg | 5 | ~5h | Low |

### Can skip

| # | Experiment | Reason |
|---|-----------|--------|
| AVE/KS InfoReg | Architecture incompatible (CNN vs required MLP) |
| AVE/KS MILES | Medium effort, low-imbalance datasets show minimal differences |
| BraTS MMPareto/AGM/G-Blend/InfoReg/MILES | High effort (DeepLab adaptation), diminishing returns |

---

## Paper Framing Decisions

### Current Section 4.2 text says:
> "InfoReg and MILES are evaluated on CREMA-D only, as their released implementations target audio-visual classification."

### Problem with this:
- InfoReg actually works BEST on pre-extracted features (MOSEI/MOSI). Saying it "targets audio-visual" is misleading.
- MILES is more flexible than implied.

### Suggested revision:
> "InfoReg and MILES are evaluated on CREMA-D only. InfoReg's Fisher-trace computation is designed for MLP encoders on pre-extracted features; adapting it to CNN-based audio-visual pipelines required approximations that may not reflect its intended use. MILES uses per-modality learning rate adjustment and was validated by its authors on audio-visual benchmarks."

OR (if we run InfoReg on MOSEI/MOSI):
> "InfoReg is evaluated on CREMA-D and the sentiment datasets (CMU-MOSEI, CMU-MOSI), where MLP encoders on pre-extracted features match its intended architecture. MILES is evaluated on CREMA-D."

### CGGM caveat (already in paper, keep as-is):
> "CGGM was originally proposed for Transformer-based encoders... we adapt it to our CNN/MLP pipeline for controlled comparison."

### Missing "---" cells in Table 1:
A reviewer seeing many "---" cells will ask why. Options:
1. **Fill as many as possible** (recommended, Tier 1+2 above)
2. **Reduce the table** to only methods with full coverage (drops InfoReg, MILES, and possibly AGM/G-Blend)
3. **Split into two tables**: one for audio-visual (full coverage), one for sentiment/segmentation (fewer baselines)

**Recommendation:** Option 1 for Tier 1 experiments. For remaining gaps, add a footnote: "Methods marked --- were not evaluated on that dataset due to architecture incompatibility or implementation constraints."

---

## Table 1 After Completing Tier 1

| Method | CREMA-D | AVE | KS | MOSI | MOSEI | BraTS |
|--------|---------|-----|-----|------|-------|-------|
| Baseline | 61.59 | 86.54 | 79.05 | 72.42 | 70.42 | 85.77 |
| OGM-GE | 69.14 | 86.96 | 77.25 | 72.68 | 72.47 | **TBD** |
| MMPareto | 65.51 | 86.37 | 78.21 | **TBD** | **TBD** | --- |
| AGM | 57.42 | 84.42 | 77.85 | --- | --- | --- |
| G-Blend | 61.10 | 87.09 | 77.75 | --- | --- | --- |
| CGGM | 50.22 | 76.72 | 73.18 | 59.45 | 68.05 | 81.13 |
| InfoReg | 67.72 | ---* | ---* | --- | --- | --- |
| MILES | 61.05 | --- | --- | --- | --- | --- |
| Boost only | 62.72 | 87.41 | 79.17 | 71.89 | 69.80 | 85.98 |
| Boost+OGM-GE | 71.45 | 87.23 | 77.33 | 72.60 | 72.43 | **TBD** |

*Architecture incompatible (InfoReg requires MLP on pre-extracted features)

Missing cells drops from 22 to 17 (Tier 1 fills 5 cells).
Adding Tier 2 MOSEI/MOSI experiments fills 4-8 more.
