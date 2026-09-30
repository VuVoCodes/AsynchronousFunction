---
name: sentiment-column-provenance
description: Provenance of the second sentiment benchmark column. Both the NeurIPS camera-ready and the ICLR port now label it CH-SIMS with the original CH-SIMS numbers; the brief 2026-09-25 "genuine CMU-MOSEI" swap was reverted
metadata:
  type: project
---

The paper's second sentiment benchmark was originally labelled CMU-MOSEI but was computed on a CH-SIMS pickle. Both manuscript copies are now corrected by the **label** route: the numbers are the original ones and the name is CH-SIMS.

**`Manuscript_NEURIPS_CAMREADY/` (verified 2026-09-25, post-revert).** A short-lived attempt on 2026-09-25 replaced the column with genuine CMU-MOSEI numbers (baseline 62.07 ... PGGB+OGM-GE 62.24, 3-class, delta 0.218). **The user reverted that.** The camera-ready is back to the submitted numbers under a CH-SIMS name:
- Table 1 sentiment column: Baseline 70.42, OGM-GE **72.47** (strongest baseline), MMPareto 70.20, AGM 69.28, G-Blend 70.15, CGGM 68.05, PGGB 69.80, PGGB+OGM-GE 72.43, i.e. **$-0.04$~pp** below OGM-GE.
- Utilization gap delta = $0.178 \pm 0.086$ (high-imbalance).
- MSLR comparison (Table 11): MSLR 70.99, PGGB 69.80, PGGB+OGM-GE 72.43, reported as $+1.44$~pp.
- F1-macro (Table 5): PGGB+OGM-GE 62.36 top.
- Appendix B.1 descriptors still read 16,326/1,871/4,659, BERT 768-d / COVAREP 74-d / FACET 35-d, 7 classes, CMU-MultimodalSDK URL, "YouTube opinion videos". **These describe CMU-MOSEI, not CH-SIMS.** The user knows and is handling the descriptor correction separately. Do not silently fix them.

**`Manuscript_ICLR/main.tex`:** same CH-SIMS label with the same CH-SIMS numbers (relabelled 2026-09-14).

**Evidence scoping for the text-dominance claim (important):**
- `liu2022chsimsv2` (CH-SIMS v2.0, ICMI 2022) is the on-point citation for CH-SIMS: acoustic and visual modalities "contribute much less than the textual modality", termed text-predominant. Cite it for **predominance only**.
- Do **not** claim text-*sufficiency* for CH-SIMS. The CH-SIMS paper's Table 4 has multimodal MLF-DNN 82.28 Acc-2 vs text-only 75.19 and vision-only 74.44, so there is real multimodal headroom and vision is nearly level with text.
- `li2023agm`'s "text-only within roughly 1 pp of multimodal" is **CMU-MOSEI-only evidence**. It must never be attached to CH-SIMS. Removed from Section 4.1 on 2026-09-25.
- `wei2025dgl` (+2.7 pp multimodal gain) is **CMU-MOSI-only**.
- **Residual A (wording kept by explicit user decision, 2026-09-25):** Introduction l.126 and Section 4.2 l.471 read "text alone is near-multimodally sufficient" for "(CMU-MOSI, CH-SIMS)". The citation was swapped `li2023agm` -> `liu2022chsimsv2`, so these now cite `\citep{liu2022chsimsv2,wei2025dgl}`. The user was told that `liu2022chsimsv2` supports *predominance* only and that CH-SIMS Table 4 shows real multimodal headroom, and **chose to keep the wording**. The sufficiency clause therefore rests on `wei2025dgl` (CMU-MOSI) alone. **Do not "fix" this wording back unless asked** -- it is a deliberate decision, not an oversight.
- **Residual B (raised 2026-09-25, partially addressed by explicit user decision the same day):** Appendix `app:delta` (l.1002, "Per-dataset utilization gap") reads "CMU-MOSI and CH-SIMS imbalance reflects text-sufficiency, where text-only baselines are reported within roughly $1$~pp of multimodal accuracy". The user asked only for `liu2022chsimsv2` to be **added**, so the group is now `\citep{li2023agm,liu2022chsimsv2,wei2025dgl}` and the prose is unchanged. `li2023agm` was deliberately **kept**, so the MOSEI-only ~1 pp figure is still grouped with CH-SIMS here, and unlike Section 4.1 (l.438) this site does not scope evidence per dataset. Treat this like Residual A: a deliberate user decision, **do not "fix" the wording or drop `li2023agm` unless asked**. The clean per-dataset phrasing already exists at l.438 if the user ever wants l.1002 aligned to it.

**Framing:** all eight sentiment accuracies span ~4.4 pp but PGGB+OGM-GE sits $-0.04$~pp below OGM-GE. Standing framing: "remains within seed standard deviation of the strongest baseline". Never upgrade it to a win. CMU-MOSI is $-0.08$~pp below its strongest baseline, same framing.

**Bib keys:** `yu2020chsims` (ACL 2020, pages 3718--3727) and `liu2022chsimsv2` (ICMI 2022, pages omitted, unverified) are in the camera-ready `references.bib`. `zadeh2018mosei` remains in the file but is now uncited.
