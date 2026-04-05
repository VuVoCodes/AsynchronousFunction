# Experimental Credibility Audit

**Date:** 2026-04-04
**Purpose:** Track flags for transparency disclosures in the paper and items to revisit before submission.

---

## Cross-Validation (Strong Points)

| Our Result | Published Match | Source | Gap |
|-----------|----------------|--------|-----|
| OGM-GE CREMA-D 69.14% | 69.19% | MMPareto (ICML 2024) Table 1 | +0.05pp |
| OGM-GE CREMA-D 69.14% | 68.70% | InfoReg (CVPR 2025) Table 2 | +0.44pp |
| OGM-GE CREMA-D 69.14% | 70.4% | MILES Table II | -1.26pp |
| Baseline CREMA-D 61.59% | 62.6% | MILES Table II | -1.01pp |

These confirm our pipeline is producing credible numbers on CREMA-D.

---

## Flag 1: Pretrained vs From-Scratch Encoders (KS, AVE)

**Issue:** Our KS and AVE baselines (79.05%, 86.54%) are 10-20pp higher than all published numbers because we use pretrained ImageNet ResNet18. Published papers (MMPareto, ARL, InfoReg, OGM-GE) all train from scratch.

**Impact:** Absolute numbers not comparable cross-paper. Relative improvements (ASGML vs baseline) remain valid.

**Required disclosure:** State clearly in Section 4.1 whether each dataset uses pretrained or from-scratch encoders. Add sentence: "Absolute numbers may differ from those in original publications due to encoder initialization; all methods are compared under identical conditions."

**Action:** [ ] Verify pretrained status in configs for each dataset
**Action:** [ ] Write disclosure paragraph in experimental setup

---

## Flag 2: CGGM Architecture Mismatch

**Issue:** CGGM was designed/tuned for Transformer encoders (MSA). We run it on ResNet18 (CNN) and MLP encoders. Our CGGM numbers (50% CREMA-D, 59% MOSI) are far below their published results on different datasets with Transformers.

**Additional issue:** L_gm (gradient direction loss) is likely non-functional — `get_l_gm_for_loss()` returns a Python float, not a differentiable tensor. The direction modulation half of CGGM may be disabled.

**Impact:** Our "ASGML beats CGGM" claim is architecture-specific, not general.

**Required disclosure:** "CGGM was originally proposed for Transformer-based encoders; we adapt it to our CNN/MLP pipeline for controlled comparison. The performance gap may partly reflect this architectural mismatch."

**Action:** [ ] Fix L_gm bug (return tensor, not float) and re-evaluate — if CGGM improves significantly, re-run all 6 datasets
**Action:** [ ] If CGGM remains poor after fix, document that the architecture mismatch is the primary cause
**Action:** [ ] Frame CGGM finding as "does not transfer to CNN architectures" rather than "we beat CGGM"

---

## Flag 3: MILES Large Gap (61% vs published 75.1%)

**Issue:** 14pp gap explained by their 80-config LR search (Adam) vs our single config (SGD). Their published number is best-of-80-trials.

**Impact:** Low. Controlled comparison is more informative than cross-paper.

**Required disclosure:** Note in baselines paragraph that MILES paper uses extensive hyperparameter search with Adam optimizer, while our controlled comparison uses a shared SGD configuration.

**Action:** [ ] Optional: try Adam + a few more LR configs for MILES to narrow the gap

---

## Flag 4: ARL Non-Reproducible (63% vs published 76.6%)

**Issue:** 13.7pp gap. Found bug in their reference code (GradScale stays fixed at init value). Their baseline (58.83%) is 7pp below consensus. Even exact architecture match yields 62.90%.

**Impact:** ARL published numbers appear unreliable. We should not compare against their published number.

**Required disclosure:** If including ARL as a baseline, note it was reimplemented following reference code and report our controlled number. Do NOT copy their published 76.61% into our comparison table.

**Action:** [ ] Decide whether to include ARL in the paper at all — it's from the same lab as OGM-GE/OPM/MMPareto
**Action:** [ ] If included, mention reproducibility issue diplomatically

---

## Flag 5: InfoReg Gap (67.7% vs published 71.9%)

**Issue:** 4.2pp gap from LR scheduler difference (our StepLR step=70 vs their step=30, plus β/K params).

**Impact:** Low. Gap is modest and well-explained.

**Required disclosure:** Note scheduler difference if InfoReg appears in comparison table.

**Action:** [ ] Optional: try their exact scheduler (StepLR step=30) to close the gap

---

## Flag 6: BraTS Cross-Paper Numbers

**Issue:** Our baseline (85.77%) far exceeds CGGM paper's baseline (69.21%) and CGGM result (73.94%). Likely different data splits and preprocessing.

**Impact:** Cross-paper BraTS numbers are not comparable. Our controlled comparison (ASGML 85.98% vs baseline 85.77% vs CGGM 81.13%) is valid.

**Required disclosure:** "Cross-paper BraTS numbers are not directly comparable due to differences in data splits and preprocessing. We report controlled comparisons on a fixed split."

**Action:** [ ] Document exact BraTS split details (1000/125/126) in the paper

---

## Flag 7: MOSI/MOSEI — Method Does Not Clearly Help

**Issue:** On MOSEI, OGM-GE alone (72.47%) ≈ boost+OGM-GE (72.43%). On MOSI, OGM-GE (72.68%) > boost+OGM-GE (72.60%) > baseline (72.42%). Boost-only hurts on MOSI (-0.53pp).

**Impact:** Cannot claim ASGML "generalizes" to 3-modality if it doesn't add value beyond OGM-GE.

**Required disclosure:** Be transparent that on MOSEI/MOSI, OGM-GE drives the improvement and ASGML provides marginal additional benefit (variance reduction).

**Action:** [ ] Frame MOSEI/MOSI as "ASGML does not interfere" rather than "ASGML helps"

---

## Flag 8: KS Improvement Within Noise

**Issue:** Boost-only 79.17% vs baseline 79.05% = +0.12pp, within std (±0.97%).

**Impact:** Cannot claim significant improvement on KS.

**Required disclosure:** Present KS as "matches baseline" or "boost-only avoids the degradation caused by OGM-GE (-1.80pp)."

**Action:** [ ] Frame KS result around OGM-GE degradation, not ASGML improvement

---

## Summary: Paper Disclosure Checklist

- [ ] Encoder initialization (pretrained vs from-scratch) per dataset in Section 4.1
- [ ] "All methods compared under identical conditions" statement
- [ ] CGGM architecture caveat
- [ ] Fix CGGM L_gm bug and re-evaluate
- [ ] Transparent MOSEI/MOSI framing
- [ ] KS framing around OGM-GE degradation
- [ ] BraTS split details
- [ ] Cross-paper comparison caveat paragraph
