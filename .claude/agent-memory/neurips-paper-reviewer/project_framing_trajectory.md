---
name: Framing trajectory decision
description: Tracks framing debates and fixes across 2026-04-23 hybrid decision, 2026-04-24 never-hurts / causal-inert verification, and 2026-04-24 (i)↔(ii) identity resolution
type: project
---

**2026-04-23 decision:** Hybrid framing (co-equal monitor+boost, three-way ablation added).
- Three-way ablation now in paper: OGM-GE / Monitor+OGM-GE α=0 / Boost+OGM-GE α=0.75.

**2026-04-24 verification pass (framing fixes applied to main.tex):**

Pillar 1 — Never-hurts / near-neutral framing. Abstract L55, §1 P5 L99, contribution bullet 4 L110. MOSEI scoping rule stated: abstract/§1 scoped to "five low-imbalance benchmarks", §5 broader. Reconciled.

Pillar 2 — Causal-inert at α=0 (L595). Original parenthetical listed "probe-optimizer interactions, shared batch-norm drift, dataloader ordering" — last two factually wrong. Replaced with "probe training on detached features, Adam steps on probe parameters, and EMA state updates. Only the multiplicative scale $s_m$ differs between α=0 and α=0.75." VERIFIED in L595 as of 2026-04-24.

Pillar 3 (new, 2026-04-24 late pass) — Operational identity of (i)↔(ii) disclosed in §4.3. Paper now states that Table 1 OGM-GE alone (69.14±1.13, seed set A) and Monitor+OGM-GE α=0 (69.14±1.18, seed set B) are the SAME invocation (`--mode adaptive --continuous-alpha 0.0 --ogm-ge`, probes active in both), run on different seed sets. The 69.14=69.14 identity is a reproducibility check, not a counterfactual. (iii) α=0.75 is the only condition that differs, isolating boost actuation. Abstract edited to say "setting α=0 while keeping probes active yields the same mean as OGM-GE alone" — no longer implies probes-off comparator.

**Resolved tension:** Prior framings risked the claim "probes-off → 69.14, probes-on-α=0 → 69.14, probes+boost → 71.45 shows boost causally" when the first arm was actually probes-on-α=0. Fix converts the arm pair into a reproducibility check + an α-only contrast. This is defensible and honest.

**Residual hole:** No true "probes-off while keeping OGM-GE" comparator exists in Table 1. Meta-reviewer could ask whether mere presence of probe pipeline nudges OGM-GE's trajectory. Unlikely to block acceptance given the fix framing, but worth preempting in rebuttal ("probes are fully detached, add no gradient path into encoders, and their presence at α=0 is thus causally inert by construction").

**Score projection after all three pillars (2026-04-24 final pass):**
- Novelty 5/10 (unchanged — framing fixes don't add novelty).
- Technical Soundness: 8/10 (L595 now accurate; (i)↔(ii) identity now honestly disclosed rather than implicitly overclaimed).
- Experimental Rigor: 7→7.5/10 (three-way ablation remains, honesty-of-framing improved; no new data).
- Clarity: 8/10 (maintained; pillar 3 adds a sentence but clarifies the causal story).
- Related Work 7, Repro 8 (unchanged).
- Recommendation: **Weak Accept**, now cleanly cleared. Abstract + §4.3 no longer overclaim.

**Remaining blockers to move Weak Accept → Accept:**
- (A) Proposition 3 acknowledged as standard-SGD-with-scalar-inflation, not a new convergence result. Partially reframed in abstract ("a standard-SGD descent bound") — OK.
- (B) RESOLVED 2026-04-24 evening. Reproduction-protocol paragraph at L525 §4.1 now states: matched pipeline (shared encoders, 3-frame CREMA-D, identical augmentation/optimizer), explicit per-method caveats (OGM-GE 3-6pp below follow-up due to frame sampling, AGM uses our ResNet-18, MILES ported without specialized architecture, CGGM from Transformer setting), and final sentence attributes sub-baseline AGM/MILES CREMA-D numbers to 3-frame visual protocol sensitivity rather than method weakness, explicitly declining per-method retuning to preserve comparability. VERIFIED in main.tex L525.
- (C) AVE from-scratch is framed as OGM-GE sensitivity, explicit and honest. Acceptable.

**2026-04-24 final framing state (post-reproduction-protocol fix):**
- (B) was flagged as the single most impactful remaining action. It is now addressed in a defensible form: the paper owns the reproduction gap, localizes it to the 3-frame protocol, and justifies non-retuning as a comparability choice rather than an oversight. This converts what was a reviewer attack surface into a pre-empted methodological decision.
- Residual ceiling items that remain binding (will NOT move with further text-level edits):
  (1) Novelty 5/10 — method is a scheduling intervention on top of OGM-GE, not a standalone SOTA mechanism. Structural, not fixable by writing.
  (2) CREMA-D-only headline (+2.31 pp) — other datasets show near-neutral. Already honestly scoped in abstract/§1.
  (3) OGM-GE dependency — method is operationalized as a scale on OGM-GE's ratio, not a free-standing optimizer. Already disclosed in §3.
- These three items together cap the paper at **Weak Accept** from this reviewer's lens. Moving to Accept would require either (a) a non-OGM-GE backbone showing the same boost gain, or (b) a dataset beyond CREMA-D with >1 pp gain. Neither is a text fix.

**Score projection after reproduction-protocol fix (2026-04-24 final):**
- Novelty 5/10 (unchanged, structural ceiling).
- Technical Soundness 8/10 (unchanged, already cleared by Pillar 2+3).
- Experimental Rigor 7.5 → 8/10 (reproduction protocol now explicit, pre-empts a predictable reviewer complaint).
- Clarity 8/10 (unchanged).
- Related Work 7/10 (unchanged).
- Reproducibility 8/10 (unchanged; the fix is framing, not new reproducibility material).
- Recommendation: **Weak Accept (cleanly cleared)** — solidly above the acceptance threshold but not yet Accept due to structural ceiling items.

**How to apply:**
- Check L595 list — VERIFIED correct 2026-04-24.
- Check abstract L55 boost phrasing — VERIFIED correct.
- Three-way ablation paragraph §4.3 — VERIFIED discloses (i)↔(ii) operational identity.
- MOSEI scoping — VERIFIED in L99 and L110, §5 retains full list.
- Reproduction protocol L525 §4.1 — VERIFIED 2026-04-24 evening. Final sentence explicitly attributes AGM/MILES sub-baseline numbers to 3-frame protocol sensitivity.
- Novelty ceiling remains 5/10 — framing fixes do not move it. Further Accept-direction movement requires new empirical material, not writing.
