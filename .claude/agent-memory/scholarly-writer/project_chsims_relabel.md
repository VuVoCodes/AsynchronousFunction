---
name: chsims-relabel-evidence-scoping
description: The sentiment column once labeled CMU-MOSEI is CH-SIMS (relabeled in Manuscript_ICLR 2026-09-14); which citations support which sentiment dataset; do not call CH-SIMS text-dominant
metadata:
  type: project
---

In `Manuscript_ICLR/main.tex` the benchmark formerly labeled "CMU-MOSEI" is **CH-SIMS** (\citep{yu2020chsims}, ACL 2020, pp. 3718--3727, verified on ACL Anthology). Numbers were kept unchanged (valid CH-SIMS results). The authors chose relabel-only, with no correction narrative in the double-blind ICLR submission. `zadeh2018mosei` stays in references.bib but is no longer cited. `Manuscript/main.tex` (NeurIPS) was NOT relabeled as of that date.

**Why:** A preprocessed MMSA CH-SIMS pickle (`unaligned_39.pkl`) was loaded as MOSEI (see [[mosei-column-is-actually-chsims]] in user auto-memory).

**How to apply (evidence scoping, checked against the PDFs in Papers/):**
- `li2023agm` text-only within about 1 pp claim = CMU-MOSEI only (AGM Intro, p.1). Never attach it to CH-SIMS or CMU-MOSI.
- `wei2025dgl` Table 2 (right), CMU-MOSI, MLP fusion: text-only 77.12, multimodal baseline 76.83, best (DGL) 79.78, i.e., +2.7 pp over text-only. Balancing methods +0.7 to +2.0 pp over baseline.
- `guo2024cggm` Table 4 (CMU-MOSI Acc-2): text-only 76.83, baseline 81.23, CGGM 82.84. `wei2024opm` Table 11 (CMU-MOSI): concat 75.9, OPM 77.6, OGM 76.8. Both support "modest gains" on CMU-MOSI only.
- CH-SIMS high-imbalance status rests on measured delta = 0.178 +/- 0.086 alone. Nothing in the paper supports "text-dominant" for CH-SIMS.
- CH-SIMS setup: 2,281 segments, official split 1,368/456/457, Chinese BERT-base 768-d (39 tokens), LibROSA 33-d, OpenFace 2.0 709-d, 3-class labels from MMSA, z-score and mean pooling, same 2-layer MLP (512-d, dropout 0.3).
- Open issue flagged, not fixed: `src/datasets/cmu_mosi.py` loads `mosi_raw.pkl` first (1,283/214/686, audio 74-d, vision 35-d, binary labels). The appendix CMU-MOSI entry states 1,284/229/686 with 5-d/20-d features (that is `mosi_data.pkl`). Authors must confirm which file the runs used.
