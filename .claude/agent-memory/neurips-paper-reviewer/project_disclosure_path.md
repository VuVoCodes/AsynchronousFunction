---
name: project-disclosure-path
description: 2026-07-25 rebuttal — authors chose proactive full-disclosure of the CMU-MOSEI/CH-SIMS mislabel; AC role-play leaned accept-conditional
metadata:
  type: project
---

As of 2026-07-25 the rebuttal package (Reviews/responses/) takes the PROACTIVE FULL-DISCLOSURE path on the mislabeled sentiment benchmark: the column labeled CMU-MOSEI was CH-SIMS data, now disclosed openly, relabeled CH-SIMS, with a genuine CMU-MOSEI re-run (5 seeds, ~62%, flat text-dominant) and re-measured delta 0.218 +/- 0.005.

**Why:** The mislabel (see [[project_mosei_is_chsims]] in user auto-memory) was caught during rebuttal verification prompted by reviewer gN93/Q3. Disclosing proactively rather than quietly correcting was judged the higher-trust move.

**How to apply:** In an AC role-play stress-test (submission 12365, AC wmcn), this disclosure resolved the last stuck meta-review point (#5, imbalance threshold — 0.178 inside the fragile empty interval is replaced by a clean 0.218 +/- 0.005) and raised trust rather than lowering it. Residual AC asks for the camera-ready: (1) per-benchmark provenance-audit statement / checksums, because Appendix B.1 described genuine CMU-MOSEI params (16,326 train, 7-class) while the CH-SIMS data actually run had 1,368 train / 3-class — a 12x sample mismatch that survived to submission; (2) sanity-check the genuine-MOSEI ~62% absolute number against literature (Acc-7 typically ~50-54%) and reconcile Q3 probe gap (~15pp) vs delta=0.218. Camera-ready CMU-MOSEI table numbers move ~8-10 pp, so recommend AC-shepherded (not unsupervised) camera-ready.
