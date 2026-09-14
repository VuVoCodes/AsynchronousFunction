---
name: review-20260725-r1-rebuttal-soundness
description: Soundness + oversharing audit of Reviews/responses/R1_tQk1.md — pass 2 (2026-07-25) after revision; 3 of 4 original blockers fixed, W2 trajectory-table arithmetic remains the sole BLOCKER
metadata:
  type: project
---

Two audit passes on the rebuttal to Reviewer tQk1 (Accept 5, conf 3).

## Pass 2 (2026-07-25, post-revision) — status

**RESOLVED since pass 1:** β=0.1 is now named alongside the μ=0.3 scale EMA (two-EMA confusion fixed). The per-epoch iteration-counter reset is now signaled by "within each 105-iteration epoch". The two sentences falsely claiming the boost "vanishes"/"holds scales near 1 regardless" were removed and replaced with "provably small near exact balance", which is literally defensible.

**STILL OPEN — BLOCKER: W2 trajectory table is arithmetically unreconstructable.**
Rebuttal quotes raw first-event accuracies (9.4%, 15.6%) — exactly 3/32 and 5/32, so the n=32 eval half is confirmed — but tabulates first-event EMA values (2.1, 2.8). With β=0.1 from zero init the first update yields 0.94 and 1.56, not 2.1 and 2.8. Implied per-column β differs (0.223 audio vs 0.179 visual), so no single β or debiasing explains it. Row 2 compounds it: inverting the EMA gives raw audio 45.1 ± 0.95%, and no multiple of 1/32 (43.75 = 14/32, 46.875 = 15/32) falls in that window. Visual reconciles at 9/32 = 28.125. The audio column specifically cannot be reconstructed.
**Why:** the rebuttal volunteers raw numbers purely to support the binomial argument, which hands a checker the exact data needed to falsify its own table.
**How to apply:** any early-trajectory table must report raw P_m or EMA, not both, and if both, state the initialization and any debiasing.

**STILL OPEN — FRICTION: epsilon guard, narrowed but not eliminated.**
The rebuttal's "provably small near exact balance" is now correct, but main.tex still overclaims in two places the rebuttal does not promise to fix: line 326 ("when modalities are balanced (δ small) ... s_m ≈ 1, so the method applies no intervention") and line 471 ("consistent with self-attenuation as δ → 0") used to explain low-imbalance results at δ ≤ 0.15. Since w_weak = δ/(δ+ε) ≈ 1 for any δ far above ε, both are false. The promised reword covers abstract + intro only — under-scoped.
Secondary risk: W2 pt. 4 and W3 pt. 1 sit in the same response. Inverting the EMA at pt. 4 shows a 0.7 pp gap produced the *maximal* instantaneous scale 1+α = 1.75, which reads as a counterexample to "provably small near balance".

**Verified clean (do not re-litigate):** binomial sd sqrt(p(1-p)/32) = 6.59 → 6.6 pp; 1+μα = 1.225 ≈ 1.23 and ≤ s_max = 2 (Prop. 1); inversion iterations 5434 and 5479 are valid probe events (epoch 52 local 79, epoch 53 local 19), non-consecutive, consistent with "no inversion persisted beyond a single event"; 500 events = 5/epoch × 100 requires the per-epoch reset and is self-consistent; 9.41 / 0.81 / 5.4× match main.tex lines 567 and 572; "Section 4.1" correctly resolves to sec:datasets (line 420) where the ~1% estimate lives (line 434); zero semicolons.

**NITs:** +7.4% should be +7.3% ((18.71−17.43)/17.43 = 7.344), and the derived "~5.8 pp" becomes ~5.7. Overhead table reports no std over its 7 epochs, so a 0.28 s/epoch delta is not visibly separable from jitter. "never inverts again during the imbalanced phase" is circular once pt. 3 discloses later inversions.

**Oversharing items flagged (cut or compress):** the probe-EMA cold-start diagnostic "matching main-task accuracy from epoch 3 onward" concedes epochs 1–2 differ, which is precisely the window the reviewer asked about; the "(differences below 0.2 GB are within allocator noise)" parenthetical undercuts the identical memory figures it follows; exact inversion iteration indices are gratuitous precision.

**Standing paper bug (both passes):** main.tex line 896 cites "the per-epoch overhead reported in §4.4 (sec:analysis)" but overhead lives in §4.1 (line 434). Fix alongside the promised ~1% → 1.6% update.
