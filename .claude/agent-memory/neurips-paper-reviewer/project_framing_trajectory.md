---
name: Framing trajectory decision
description: Tracks framing debates and fixes across 2026-04-23 hybrid decision, 2026-04-24 never-hurts / causal-inert / (i)↔(ii) / App. A+B closure passes
type: project
---

**2026-04-23 decision:** Hybrid framing (co-equal monitor+boost, three-way ablation added).
- Three-way ablation now in paper: OGM-GE / Monitor+OGM-GE α=0 / Boost+OGM-GE α=0.75.

**2026-04-24 verification pass (framing fixes applied to main.tex):**

Pillar 1 — Never-hurts / near-neutral framing. Abstract L55, §1 P5 L99, contribution bullet 4 L110. MOSEI scoping rule stated: abstract/§1 scoped to "five low-imbalance benchmarks", §5 broader. Reconciled.

Pillar 2 — Causal-inert at α=0 (L595). Original parenthetical listed "probe-optimizer interactions, shared batch-norm drift, dataloader ordering" — last two factually wrong. Replaced with "probe training on detached features, Adam steps on probe parameters, and EMA state updates. Only the multiplicative scale $s_m$ differs between α=0 and α=0.75." VERIFIED in L595 as of 2026-04-24.

Pillar 3 — Operational identity of (i)↔(ii) disclosed in §4.3. Paper now states that Table 1 OGM-GE alone (69.14±1.13, seed set A) and Monitor+OGM-GE α=0 (69.14±1.18, seed set B) are the SAME invocation (`--mode adaptive --continuous-alpha 0.0 --ogm-ge`, probes active in both), run on different seed sets. The 69.14=69.14 identity is a reproducibility check, not a counterfactual.

Pillar 4 (2026-04-24 late evening, App. A+B closure pass) — Theory and appendix attack surfaces closed.
- **App. A H1 (σ² block-additivity):** Added explicit per-block σ_m^2 derivation at L782 showing $\sum_m \sigma_m^2 \leq \sigma^2$ via block decomposition of the stochastic-gradient deviation. This removes the hand-wave that had the second-moment bound use a global σ² without justifying block separability. VERIFIED.
- **App. A H2 (σ-algebra filtration):** Added $\mathcal{F}_t$ definition at L779 including probe parameters + EMA state, and proved $\bar{s}_m^{(t)}$ is $\mathcal{F}_t$-measurable via detached-features construction, so the conditional-expectation step in Eq. (inner_bound) is rigorous. VERIFIED.
- **App. A M3 (Eq. 6 vs code hard-branch):** Footnote at L382 discloses the $s_m = 1$ hard branch when gap $< \epsilon$, and notes this coincides with the continuous Eq. 6 at $\Delta = 0$. Paper-vs-code fidelity now honest. VERIFIED.
- **App. A M4 (partial sum tightening):** Proposition 2 proof at L741-743 correctly uses $\mu \sum_{k=0}^{t-t_0-1}(1-\mu)^k = 1 - (1-\mu)^{t-t_0} \leq 1$. VERIFIED.
- **App. A M5 (index range):** Propositions 1–3 all index correctly over $t \geq t_0$ / $t \geq 0$. VERIFIED.
- **App. A M6 (LR algebra step):** L803 shows the explicit $\eta \leq 1/(L s_{\max}^2) \Rightarrow L\eta s_{\max}^2/2 \leq 1/2$ step, no longer skipped. VERIFIED.
- **App. B.1 licensing:** L850 dedicated "Licensing" paragraph lists CC-BY 4.0 for CREMA-D and references NeurIPS checklist. VERIFIED.
- **App. B.2 OPM softening:** L855 claim narrowed to CREMA-D under OPM's native single-layer fusion pipeline. VERIFIED.
- **App. B.3 "approximately 5 pp":** L885 uses soft "approximately $5$~pp" wording for Food101 trainable-vs-frozen contrast. VERIFIED.
- **App. B.4 F1 claim scoped:** L918 only claims F1 confirms ranking on CREMA-D + CMU-MOSEI, honestly notes G-Blend top on Twitter15 and partial `---` entries. VERIFIED.
- **App. B.5 noise-floor sentence:** L948 adds "Trends smaller than the $\approx 1.5$~pp 5-seed noise floor should be read as within-noise rather than as detected effects." VERIFIED.
- **App. B.6 53%-vs-95% gap:** L976 explicitly states the $53\%$ observed is below the $\approx 95\%$ i.i.d. steady-state bound "as expected under non-stationary training." VERIFIED.
- **App. B.7 semicolons removed + MDE sentence + "never-hurts" renamed + (i)↔(ii) cross-ref:** L1002 now reads "support the non-degrading composition behavior observed on CREMA-D under this protocol" (no "never-hurts"), includes MDE sentence "At n=5 with σ≈1 pp, the 80%-power minimum detectable effect is approximately 2 pp", cross-refs the (i)↔(ii) operational-identity claim from §4.3. No prose semicolons in the paragraph. VERIFIED.
- **App. B.8 chance-line footnote:** L1013 caption states "Chance level is $1/6 \approx 16.67\%$ for CREMA-D's 6-class classification", correctly contextualizing AGM/MILES visual probe numbers ($17.72$, $19.62$) as near-chance. VERIFIED.

**Score projection after App. A+B closure (2026-04-24 final, final):**
- **Novelty 5/10** (unchanged — structural ceiling, not a writing item).
- **Technical Soundness 8/10 → 8.5/10** (H1+H2 close the two residual theory handwaves; Prop. 3 remains a standard-SGD-with-scalar-inflation result so it does not rise to 9/10, but the proof is now self-contained and rigorous under the stated assumptions).
- **Experimental Rigor 8/10** (unchanged — App. B fixes tighten framing/scope but add no new data).
- **Clarity 8/10 → 8.5/10** (semicolons removed, MDE sentence added, (i)↔(ii) cross-ref deepens causal story readability).
- **Related Work 7/10** (unchanged).
- **Reproducibility 8/10** (unchanged — licensing paragraph helps, but not materially).
- **Recommendation: Weak Accept (cleanly cleared, at the top of the band).** Defensible against the most aggressive AC review lens, modulo the three structural ceilings listed below.

**Structural ceilings that remain binding (will NOT move with further text-level edits):**
1. **Novelty 5/10** — method is a scheduling/scaling intervention on top of OGM-GE, not a standalone SOTA mechanism.
2. **CREMA-D-only headline (+2.31 pp)** — other datasets show near-neutral. Already honestly scoped.
3. **OGM-GE dependency** — method operationalized as a scale on OGM-GE's ratio, not free-standing.

Moving Weak Accept → Accept requires either (a) a non-OGM-GE backbone showing the same boost gain, or (b) a dataset beyond CREMA-D with >1 pp gain, or (c) a second high-imbalance benchmark where the +2.31 pp magnitude replicates. None are text fixes.

**Residual items worth flagging before submission (each LOW):**
- Abstract word-budget: "composition behavior observed" is slightly verbose, but within budget.
- F1 `---` entries in Table \ref{tab:f1_macro}: caption disclosure is acceptable but reviewers may still dock Repro slightly.
- Prop 3 interpretation paragraph (L823) ends with "recovering the standard SGD convergence rate" — true, but a reviewer may ask about the rate DURING imbalance (before $\Delta \to 0$). This is standard-SGD-with-scalar-inflation; a one-sentence acknowledgement at the end of the interpretation paragraph would fully close it, but is NOT required for Weak Accept.

**Final state-of-paper (2026-04-24 evening):** Paper is at the highest score achievable by writing edits alone. All identified attack surfaces (framing, causal-inert justification, (i)↔(ii) identity, theory handwaves, appendix overclaims) are closed. Remaining gap to Accept is purely experimental and structural. Verdict: **APPROVE for submission.**

**How to apply:**
- Check L595 (causal-inert list), L99/L110 (MOSEI scoping), §4.3 three-way ablation paragraph, L525 reproduction protocol — all VERIFIED 2026-04-24.
- Check L779 ($\mathcal{F}_t$ definition), L782 (block additivity), L382 (hard-branch footnote), L803 (LR algebra) — all VERIFIED 2026-04-24 late pass.
- Check L850 (licensing), L855 (OPM scope), L976 (53%/95% gap), L1002 ("non-degrading" replacing "never-hurts", MDE sentence, (i)↔(ii) cross-ref, no semicolons), L1013 (chance-line) — all VERIFIED 2026-04-24 late pass.
- Novelty ceiling remains 5/10 — further Accept-direction movement requires new empirical material, not writing.
