---
name: 2026-05-02 Appendix audit findings
description: Reviewer-style audit of App. A (proofs) and App. B.1-B.8, focused on what NeurIPS reviewers would actually flag
type: project
---

Audit of /home/main/AsynchronousFunction/Manuscript/main.tex lines 634-995.

**Top blocking risks (will draw direct reviewer comment):**
1. App. B.5 hyperparameter sweeps are single-seed on a 1.5 pp noise floor — needs 3+ seeds at default points minimum.
2. App. B.1 datasets section has no download URLs/DOIs — fails NeurIPS reproducibility checklist.

**High-priority flags:**
- App. A Step 2 (L725): $\mathcal{F}_t$ measurability is asserted *after* the expectation step that depends on it. Reorder.
- App. A Step 2: "fresh independent sample" assumption (mini-batch sampling independent of probe state) should be in Assumption 2, not asserted mid-proof.
- App. B.4 missing F1 entries (`---`) need a one-sentence explanation of the lost-logs cause to defuse cherry-pick suspicion.
- App. B.6 audio "EMA std > batch std" inversion — explanation is post-hoc; the §3.2 "$19\times$ bound" should be weakened to match the empirical $\sim 2\times$ on visual.
- App. B.7 "non-degrading" framing collides with AVE row showing $-0.25$ on G-Blend and CGGM — clarify scope ("on CREMA-D in mean accuracy under this protocol").

**Cross-paper consistency:**
- L939 OGM-GE base "69.14±1.13" must match main Table 4.3 OGM-GE row.
- L725 measurability claim is contingent on probes never backproping — depends on `continuous_scale_ema` no-op fix tracked in `project_section3_pending_fixes.md`.

**Recurring style:** No semicolon violations found in App. A/B (project style upheld). A few comma-joined run-on clauses (L922, L948) are borderline.

**Layout:** All 8 appendix tables use `[H]` — risks stranded captions; consider `[!htbp]` for some.
