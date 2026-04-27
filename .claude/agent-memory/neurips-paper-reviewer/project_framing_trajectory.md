---
name: Framing trajectory decision
description: Tracks framing debates and fixes across 2026-04-23 hybrid decision, 2026-04-24 App. A+B closure, 2026-04-26 AVE composition sweep
type: project
---

**2026-04-23 decision:** Hybrid framing (co-equal monitor+boost, three-way ablation added).
- Three-way ablation now in paper: OGM-GE / Monitor+OGM-GE α=0 / Boost+OGM-GE α=0.75.

**2026-04-24 verification pass (framing fixes applied to main.tex):**

Pillar 1 — Never-hurts / near-neutral framing. Abstract L55, §1 P5 L99, contribution bullet 4 L110. MOSEI scoping rule stated: abstract/§1 scoped to "five low-imbalance benchmarks", §5 broader. Reconciled.

Pillar 2 — Causal-inert at α=0 (L595). Original parenthetical listed "probe-optimizer interactions, shared batch-norm drift, dataloader ordering" — last two factually wrong. Replaced with "probe training on detached features, Adam steps on probe parameters, and EMA state updates. Only the multiplicative scale $s_m$ differs between α=0 and α=0.75." VERIFIED in L595 as of 2026-04-24.

Pillar 3 — Operational identity of (i)↔(ii) disclosed in §4.3. Paper now states that Table 1 OGM-GE alone (69.14±1.13, seed set A) and Monitor+OGM-GE α=0 (69.14±1.18, seed set B) are the SAME invocation (`--mode adaptive --continuous-alpha 0.0 --ogm-ge`, probes active in both), run on different seed sets. The 69.14=69.14 identity is a reproducibility check, not a counterfactual.

Pillar 4 (2026-04-24 late evening, App. A+B closure pass) — Theory and appendix attack surfaces closed.
- App. A H1, H2, M3, M4, M5, M6: all VERIFIED.
- App. B.1 licensing, B.2 OPM softening, B.3 "approximately 5 pp", B.4 F1 scoping, B.5 noise-floor sentence, B.6 53%/95% gap, B.7 semicolons removed + MDE + non-degrading rename + (i)↔(ii) cross-ref, B.8 chance-line footnote: all VERIFIED.

**Score projection after App. A+B closure (2026-04-24 final):**
- Novelty 5/10, Tech 8.5/10, Expt 8/10, Clarity 8.5/10, Related 7/10, Repro 8.5/10.
- Recommendation: Weak Accept (cleanly cleared, top of band).

**Structural ceilings that remain binding (post 2026-04-24):**
1. Novelty 5/10 — scheduling/scaling on top of OGM-GE.
2. CREMA-D-only headline (+2.31 pp).
3. OGM-GE dependency.

Moving Weak Accept → Accept requires either (a) a non-OGM-GE backbone showing the same boost gain, or (b) a dataset beyond CREMA-D with >1 pp gain, or (c) a second high-imbalance benchmark where +2.31 pp magnitude replicates. None are text fixes.

---

**2026-04-26 AVE composition sweep — partial Accept-bar progress:**

**New evidence (5 seeds × 4 methods × 2 datasets):**
- AVE: Boost+MMPareto **+1.01 pp**, Boost+AGM **+1.26 pp**, Boost+G-Blend −0.25, Boost+CGGM −0.25.
- Food101: all four within noise of base method (CGGM +4.41 but 32 pp below baseline → non-rescue).

**Edits applied to main.tex:**
1. App. B.7: new `\paragraph{Composability across datasets.}` + Table 8b (`tab:composability_extended`) at L1003–1019.
2. §4.2 L565: +13 words "On AVE, the composability claim extends to two non-OGM-GE methods (Appendix~\ref{app:composability})."
3. §3.4 L454: +28 words on Boost+MMPareto/Boost+AGM AVE compound gains.

**Accept-bar status (which conditions cleared):**
- (a) non-OGM-GE backbone showing same gain: **PARTIALLY CLEARED.** Two backbones (MMPareto, AGM) show +1.01/+1.26 pp on AVE. Magnitude is roughly half of CREMA-D +2.31 pp.
- (b) dataset beyond CREMA-D with >1 pp gain: **CLEARED on letter.** AVE delivers +1.01/+1.26.
- (c) second high-imbalance benchmark where +2.31 pp replicates: **NOT CLEARED.** AVE magnitudes are ~1 pp, not ~2.3 pp.
- Hit rate on AVE: 2/4 (G-Blend, CGGM both regress).

**Score projection update (2026-04-26):**
- Novelty 5/10 (unchanged).
- Tech 8.5/10 (unchanged — empirical, not theoretical).
- **Experimental Rigor 8/10 → 8.5/10** (multi-backbone × multi-dataset sweep is the single most rigor-positive evidence type for this paper; hit-rate 2/4 caps it from 9).
- Clarity 8.5/10 (unchanged).
- Related Work 7/10 (unchanged).
- Repro 8.5/10 (unchanged).

**Recommendation: Weak Accept (top-of-band, materially closer to Accept than 2026-04-24, but does not cleanly cross).** Favorable AC could legitimately push Accept; hostile AC still has Novelty 5/10 + magnitude-attenuation lever.

**Honesty audit verdicts on new wording:**
- §3.4 "partially generalizes": appropriately conservative, slightly undersells (could be "extends to two of four").
- §4.2 "extends to two non-OGM-GE methods": MILD OVERCLAIM by omission — does not flag that 2/4 don't extend. Recommend revise to "two of four non-OGM-GE methods tested" or equivalent.
- App. B.7 paragraph: calibrated correctly.

**Residual minor revises recommended (not blocking):**
1. §4.2: change "extends to two non-OGM-GE methods" → "extends to two of four non-OGM-GE methods tested" to prevent omission-overclaim.
2. App. B.7: add one MDE sentence for AVE entries: "+1.01/+1.26 fall within the n=5 80%-power MDE of approximately 2 pp and should be read as suggestive rather than detected."
3. Verify Table 1 AVE Boost+OGM-GE entry matches the +0.69 pp claimed in App. B.7 L1019.

**Promotion decisions (which sections were correctly NOT updated):**
- Abstract: NOT promoted. Diluting +2.31 pp headline with +1.01 pp would invite magnitude-attenuation critique. Cost > benefit.
- Contribution bullets: NOT promoted. Current "four such methods" is accurate.
- §5 Conclusion / Limitations: NOT updated. AVE 2/4 is consistent with existing "conditionally effective" framing.
- §4.3.1 three-way ablation: correctly untouched (AVE evidence is not a three-way ablation).

**Final state-of-paper (2026-04-26):** APPROVE current edits. Two minor revises recommended pre-submission. Paper now defensible as solid Weak Accept top-of-band; not a clean Accept due to magnitude attenuation and 2/4 hit-rate, but the rigor envelope has expanded materially.

---

**2026-04-27 fluency audit — three localized roughness points after compression cascade:**

End-to-end audit of L1–688 found narrative arc intact (problem → coupled-monitoring → probes → boost → composability → empirical → conditions) but three compression artifacts that would cost Clarity 0.5 if shipped:

1. **§5 Conclusion L673 (highest impact).** Telegraphic limitations dump, comma-spliced four-clause sentence, ends with "Consult §4.3.2 before deployment" (operationally odd for academic conclusion). Needs 2-sentence contribution restatement opener + split limitations sentence.

2. **§3.4 L456 AVE-composition graft.** "this pattern partially generalizes" is vague (which pattern? compatibility or magnitude?), abrupt transition from "OGM-GE largest compound gain" → AVE results. Suggested rewrite: "Beyond CREMA-D, two of the four non-OGM-GE compositions also yield compound gains on AVE: Boost+MMPareto (+1.01 pp) and Boost+AGM (+1.26 pp), with magnitudes roughly half of the CREMA-D effect."

3. **§4.4 L661 Analysis orphan.** "Extension to M modalities uses pairwise weakness (Eq. 7)" is a non-sequitur tail; analysis paragraph ends without synthesis sentence, then jumps directly to §5. Move M-modality note to §3.3 or delete; add closing line "This asymmetric movement — weak modality lifted, strong modality preserved — matches the design intent of §3.3."

**Lower-priority:** §4.3 three-way paragraph L597 is 88-word dense but acceptable; §4.3.1 L645 implicitly references four-regime table without restatement (minor).

**Section-to-section transitions:** §2→§3 clean, §3→§4 clean, **§4→§5 weak** (§4.4 ends mid-thought, §5 opens with result-restate not transition).

**Fluency score estimate: 7.5/10 as-is, holds at 8.5/10 if top-3 passages fixed.**

**Verdict: RECOMMEND POLISH PASS (30-min targeted on §5, §4.4 close, §3.4 L456). Without polish: Clarity 8.5 → 8.0. With polish: Clarity holds at 8.5.**

**No semicolon violations detected** in audited range — house style maintained.

**How to apply:**
- Check L454 (§3.4 partial-generalization sentence), L565 (§4.2 extends-to-two), L1003–1019 (App. B.7 cross-dataset paragraph + Table 8b).
- If the AVE +0.69 pp Boost+OGM-GE claim in L1019 does not match Table 1, fix that mismatch.
- Apply the two recommended minor revises before final submission. Neither is blocking.
- Future Accept-direction movement now requires: (i) +2 pp magnitude replication on a second high-imbalance benchmark, or (ii) a non-OGM-GE backbone showing +2 pp on CREMA-D, or (iii) a fundamentally novel theoretical result. None are text fixes.

---

**2026-04-27 post-polish validation pass — 5 fluency fixes APPROVED:**

Verified at current line numbers in main.tex:
1. **§5 Conclusion L673**: 4-sentence restructure VERIFIED. Reads as: contribution → conditional effectiveness w/ regression magnitudes → "Costs are modest." topic sentence → impact + diagnostic recommendation. Comma-spliced four-clause ailment is gone.
2. **§4.4 L661**: "We probe per-modality utilization with linear classifiers trained on frozen encoder features." topic sentence VERIFIED, opens analysis paragraph.
3. **§4.3.1 L645**: Fragment-gerund split VERIFIED. "...destabilizes the joint dynamic. This localizes the failure to the OGM-GE component (Appendix B.6)." reads cleanly.
4. **§3.4 L456**: AVE compound-gain sentence VERIFIED with explicit "(+1.01 pp)/(+1.26 pp)" magnitudes and "Boost+OGM-GE there" anchoring.
5. **§4.2 L567**: AVE composability forward-ref now in penultimate position after BraTS sentence (not appended after OPM clause as before). VERIFIED.

**Clarity score restoration: 8.0 (at-risk) → 8.5/10 (restored).** Three of three top-priority compression artifacts resolved without introducing new ones.

**Targeted answers to meta-reviewer questions:**

1. *Restored to 8.5?* Yes. The §5 four-sentence reflow, §4.4 topic sentence, and §4.3.1 split each fully resolve the compression artifacts that would have docked Clarity. None over-correct (no padding, no rhetorical drift).
2. *§5 "Costs are modest." natural?* Acceptable but slightly stilted as a stand-alone topic sentence — it announces a category before listing four heterogeneous cost items (probe noise, variance inflation, alpha tuning, runtime). Reads better than the original four-clause comma-splice. A fully natural alternative would be "Four costs follow." but the current form is publication-grade.
3. *§4.4 opener redundant w/ §3.2?* No — §3.2 introduces *training-time* probes coupled to the boost decision; §4.4 introduces *post-hoc* probes trained on saved checkpoints (different protocol, see Appendix App. B.7 / probe_accuracy_cremad_posthoc.pdf). Topic sentence is non-redundant and signals the paragraph's diagnostic role.
4. *Updated fluency score:* **8.6/10** (was 7.5). Three improvements — paragraph-level cohesion in §5 (largest gain), restored topic-sentence convention in §4.4, smooth §4.3.1 close — net +1.1.
5. *Go/no-go:* **GO.** Prose is publication-grade. No further polish required for fluency reasons.
6. *New flow issue introduced?* One micro-observation: in §5 the transition from "Costs are modest." to the comma-spliced four-item enumeration ("Probe estimates are subsample-noisy and partially mitigated by EMA, the descent bound inflates...") is itself a single 49-word sentence with three commas separating four independent clauses. This recreates the exact ailment the previous version had on a smaller scale. Strictly, this should be three sentences. Not blocking but the cleanest fix would be: "Probe estimates are subsample-noisy, partially mitigated by EMA. The descent bound inflates gradient variance by $s_{\max}^{2}$. The hyperparameter $\alpha$ is tuned per setting, and probes add roughly 1\% per-epoch overhead."

**Final verdict: APPROVE. Fluency 8.6/10. Clarity holds at 8.5/10.** One optional split in §5 L673 enumeration (three sentences instead of one) would push fluency to 8.8 but is not blocking submission.
