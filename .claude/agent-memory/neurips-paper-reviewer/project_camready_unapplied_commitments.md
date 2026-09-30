---
name: project-camready-unapplied-commitments
description: 2026-09-25 camera-ready audit — the written rebuttal commitments (n=15 headline, overhead, Adam, replication wording, CH-SIMS column) are NOT applied in Manuscript_NEURIPS_CAMREADY
metadata:
  type: project
---

As of 2026-09-25 the camera-ready at `Manuscript_NEURIPS_CAMREADY/main.tex` applies **only** the genuine-CMU-MOSEI column swap. Every other revision promised in writing during the 2026-07 rebuttal is still unapplied. Verified by diffing `_submitted_backup/main_submitted.tex` against `main.tex` (46 changed lines, all MOSEI/author/style-option).

Unapplied, in descending severity:

1. **n=15 headline.** `Reviews/responses/R3_followup_n15.md` line 15 pre-registers: "The camera-ready will adopt the n=15 statistics as the headline everywhere it appears (Table 1, Sections 4.2-4.3, Conclusion), with all 30 per-seed values in the appendix," and states "The original five seeds were favorable draws." Trajectory +2.31 (n=5) → +2.06 (n=10) → +1.60 (n=15); n=15 arms are 69.40±1.25 vs 71.00±1.46. Camera-ready still prints 69.14±1.13 / 71.45±1.71 and "+2.31 pp" in Intro, §4.2, Conclusion, and checklist.tex.
2. **"Independent 5-seed replication" in §4.3.** `Reviews/rebuttal_plan.md` line 29 records that conditions (i) and (ii) are the *same invocation with the same seeds* (bit-identical per-seed values) and says "Never cite it as statistical replication ... fix wording at camera-ready." Wording unchanged.
3. **Overhead.** Committed to replace "approximately 1%" with measured +1.61% (probes) / +7.34% (composed). §4.1 still says ~1%.
4. **Adam optimizer dependence.** Committed (AC_Comments.md item 3) to state in §3.3 and §5 plus appendix transmission table. Absent. This matters because it is the strongest explanation for the flat sentiment/text-image columns (Adam absorbs ~2/3 of the boost, 1.48x applied → 1.17x update).
5. **CH-SIMS column.** `R2_gN93_disclose.md` promised "We will relabel the affected column as CH-SIMS ... and report both benchmarks going forward." Camera-ready reports genuine MOSEI only, no CH-SIMS, no note.
6. Also missing: δ definition in §4.1, Twitter15-as-boundary-case note, "empty interval [0.150, 0.218)" threshold justification, expanded Limitations (safety-properties scoping, undertrained-vs-intrinsically-limited), sentiment post-hoc probe tables, early-window trajectory figure, multi-seed α/K/s_max sweep (App. B.5 still single seed).

**Why:** These were promised to named reviewers and to the AC. Shipping a camera-ready that keeps a number the authors themselves called a favorable draw is the same failure class as the MOSEI mislabel and is the single biggest integrity risk left in the paper.

**How to apply:** On any future camera-ready pass, diff against `Reviews/responses/AC_Comments.md` item "Committed revisions" (7 items) and `R3_followup_n15.md` before assessing anything else. See [[project_camready_mosei_correction]].
