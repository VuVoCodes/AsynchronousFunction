---
name: CGGM/AGM reproduction anomalies
description: Baseline accuracy mismatches vs. published numbers, plus inventory-aggregation caveat
type: project
---

**Why:** The current Table 1 shows several baselines performing worse than the paper's own joint-training baseline on CREMA-D, which is inconsistent with their published numbers. A reviewer cross-checking will flag this, and it creates an apparent double-standard ("my method is good because the baselines are bad").

**How to apply:** Either add rebuttal-ready explanations for each anomaly, or re-run these baselines under stricter matched conditions.

**CRITICAL CAVEAT — inventory parsing artifact (discovered 2026-04-25):** When deriving claims from aggregated inventory across runs, *always filter `_archived_mixed/` and similar archive subdirectories* before computing seed statistics. A seed-42 run from `_archived_mixed/` was contaminating the `ave_scratch_baseline` aggregate, inflating std from 0.54 pp to 7.82 pp and producing a spurious "13× std-reduction" story for one of the proposed revisions. The story dissolved the moment the archive was filtered. **Future reviews of any inventory-derived claim should start from this caveat** and re-aggregate from current (non-archived) seed pools before treating large variance/effect numbers as load-bearing evidence.

Anomalies in Table 1 (CREMA-D 3-frame):
- **CGGM**: 50.22% vs. paper baseline 61.59% (−11 pp). Paper says it was adapted from Transformer to CNN/MLP via footnote. Published CGGM on CREMA-D (Guo et al. NeurIPS 2024) achieves ~77% with their native architecture. A reviewer will suspect an unfair adaptation.
  - **2026-04-25 update:** Footnote now extended to fold in cross-benchmark regression evidence (incl. Food101 −37 pp) with neutral attribution: "we attribute to backbone mismatch rather than the underlying method." Closes the prosecutorial-tone risk while making the cross-benchmark consistency visible.
- **AGM**: 57.42% vs. baseline 61.59% (−4.2 pp). Published Li et al. ICCV 2023 reports significant gains on CREMA-D. No adaptation caveat is offered.
- **MILES**: 61.05% vs. baseline 61.59% (−0.54 pp). Published Guerra et al. 2025 reports gains. No caveat.
- **OGM-GE**: 69.14% vs. published ~72-75% on CREMA-D. Paper uses 3-frame@3fps following OGM-GE but still trails. Seed variance explains part.

Additional tension in App. B.8 (per-modality probe):
- AGM visual probe: 17.72% (near chance = 16.67%) but its aggregate is 57.42% ≠ catastrophic.
  - If visual representation is near random, fusion accuracy must come from audio alone. The audio unimodal ceiling on CREMA-D is ~60%, so AGM is essentially an audio-only model. This is actually a finding the paper could lean into: AGM's balancing backfires into a unimodal model. Currently this is buried.
- MILES visual probe: 19.62% — same story.

These are defensible if the paper frames them as "methods designed for a different pipeline (single-layer, Transformer) do not transfer to this multi-layer MLP + CNN setting" — a reviewer-facing disclaimer that matches the CGGM footnote but generalizes to AGM/MILES.
