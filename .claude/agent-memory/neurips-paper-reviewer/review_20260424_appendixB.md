---
name: Review 2026-04-24 Appendix B pass
description: Deep read-only review of Appendix B.1-B.8 post-framing rework (near-neutral, causally inert, operationally identical)
type: project
---

Post-rework appendix-only pass. Main body already reframed to "near-neutral with two isolated regressions", α=0 as "causally inert", (i)/(ii) as "operationally identical replication".

**Overall appendix verdict:** MINOR REVISE — appendix is substantively coherent with the newly-scoped main body. No HIGH severity contradictions. Top issues are cosmetic-to-medium and already flagged in prior pass.

**Cross-appendix consistency findings:**

1. **B.7 "never-hurts" language (L1000) carries forward from pre-rework wording.** Main-body abstract/intro now says "near-neutral with isolated small regressions". B.7's phrase "supports the 'never-hurts' compositional property" is categorically stronger and directly contradicts the honestly-reported regressions on CREMA-D proper (KS −1.72, Sarcasm −0.64 for boost-only). The B.7 protocol is narrow (CREMA-D 3-frame only, 5 seeds, one dataset) — "never-hurts" is not demonstrable from this scope. MEDIUM. Fix: "supports the non-destructive compositional property observed on CREMA-D" or "we observe no accuracy degradation under this protocol" — strip the absolute quantifier.

2. **B.7 L1000 semicolons.** Two prose semicolons joining independent clauses (same violation flagged in review_20260423 #7). LOW. Replace both with periods.

3. **B.5 single-seed caveat is disclosed (L921 "illustrative... not a statistical comparison") — acceptable.** Keep as is.

4. **B.6 probe-stability framing is now well-scoped.** Caption at L962 explicitly invokes "i.i.d. steady-state" and explains the audio non-stationarity artifact. Consistent with §3 Eq. 5 "19× smaller" under i.i.d. steady-state. APPROVE.

5. **B.2 OPM scope.** L853 claim "composes favorably with a non-throttling paradigm" is on one dataset (CREMA-D). L875 honestly restricts the comparison to CREMA-D. Wording is defensible because the claim is scoped to the comparison protocol. LOW risk.

6. **B.6 Pearson r=0.96/Spearman ρ=0.95 claim (L976)** — verifiable from saved instrumentation per caption ("500 probe-eval checkpoints"). No mismatch flag.

7. **B.8 probe accuracy table** numerically matches Figure 2(a) of §4.5 (gap 2.31 pp for Boost+OGM-GE, 12.53 pp for baseline — identical to main text). APPROVE.

8. **B.1 datasets — licensing/URLs missing.** No dataset URLs, no license statements. NeurIPS reviewers sometimes flag this for reproducibility. LOW-MEDIUM.

9. **B.4 F1 table "---" entries.** Caption at L896 explains "runs where training logs were not retained" — acceptable disclosure, though F1 OGM-GE/CGGM both missing on key datasets is a reproducibility hole. Not inconsistent, just incomplete.

10. **B.7 Δ std column.** G-Blend Δstd −0.88 (−47%) is highlighted with \mathbf, but with n=5 this is a single-sample std ratio — no CI given. Caption at L1002 admits "should be interpreted with caution given n=5", good disclosure.

11. **Statistical power on B.7.** CGGM "+0.10 ± 1.03" with n=5 has MDE roughly ±2 pp at 80% power. The paper calls it "statistically flat" which is the right framing, but power limitation is not explicitly disclosed. Previously flagged in prior review — unresolved.

**Numerical consistency check:**
- B.2 Table 2 OGM-GE alone: 63.74±0.45 (single-layer fusion) differs from main Table 1's 69.14 (multi-layer). Disclosed (L855: different native pipeline). Consistent.
- B.7 Table 4 OGM-GE base 69.14±1.13 matches main Table 1 — consistent.
- B.7 Table 4 Boost+OGM-GE 71.45±1.71 matches main Table 1 — consistent.
- B.8 Table (post-hoc probe) baseline gap 12.53 and Boost+OGM-GE gap 2.31 match Figure 2(b) annotation (5.4× reduction = 12.53/2.31) — consistent.
- B.4 F1 Boost+OGM-GE CREMA-D 71.85 vs accuracy 71.45 — plausible (F1 can slightly exceed or trail accuracy on balanced multiclass).

**No residual "passive"/"unbiased without qualifier" leaks in B.1–B.8.** B.6 caption correctly says "conditionally unbiased" is implicit via reference to Eq. ema_probe with i.i.d. qualifier.

**Top 3 revise items by reviewer-score impact:**
1. B.7 "never-hurts" (L1000) — replace with scoped wording. Contradicts honestly-reported regressions.
2. B.7 L1000 semicolons — replace with periods.
3. B.1 — add dataset URLs/licenses (one line each).

**Score projection impact:** current projection (Nov 5 / Tech 8 / Expt 8 / Clar 8 / Rel 7 / Rep 8) is unchanged by these appendix findings. All three top items are cosmetic/near-cosmetic.
