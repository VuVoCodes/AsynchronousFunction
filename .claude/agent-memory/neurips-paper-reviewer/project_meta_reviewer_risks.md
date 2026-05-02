---
name: Meta-reviewer risks
description: Likely AC/SAC objections to anticipate in rebuttal after async→boost pivot
type: project
---

Meta-reviewer (AC/SAC) risks the paper must survive. Updated 2026-04-23.

**Why:** NeurIPS ACs discount reviewer enthusiasm when the contribution looks like engineering. These are the wedge points a skeptical AC will probe first.

**How to apply:** Before rebuttal, the authors should have ready answers for each of these. Each question should map to a concrete paragraph or table the paper can point to.

1. **"Is this mechanism genuinely new or is it PMR/IPRM with a different scaling rule?"**
   Weakness: both prior methods already use some form of modality-specific unimodal signal; differentiation must be crisp. PMR uses prototypes (non-parametric, different mechanism); IPRM uses live-model two-pass (coupled). The paper makes this case in §2.4 but could be sharper.

2. **"Is the headline effect larger than the seed noise of the best alternative?"**
   CREMA-D: boost+OGM-GE 71.45±1.71 vs OGM-GE 69.14±1.13. The confidence intervals overlap (1-sigma), and a paired t-test might still be significant with 5 seeds if seed-matched. Paper does not report seed-matched paired test; this is fragile.

3. **"Why does AGM collapse the weak probe to 17.72% on the diagnostic but still achieve 57.42% aggregate accuracy?"**
   This raises the question of whether post-hoc probe accuracy is a valid utilization proxy at all, which undermines the paper's core analytical tool. Needs defense.

4. **"On 5 of 8 benchmarks the method is statistically indistinguishable from baseline — is a paper this dataset-specific worth NeurIPS?"**
   Honest answer: the paper's "self-attenuation" framing converts a negative into a positive (non-destructive), but a meta-reviewer may see it as admission of narrow applicability.

5. **"Why is the convergence theorem more than a standard SGD bound with a constant?"**
   It's not. The $s_{\max}^2$ factor is the only non-trivial content; the probe mechanism is not in the bound. Should be reframed as "descent with bounded variance inflation" not "convergence guarantee."

6. **"How robust is this to the $K$ and $\mu$ choices?"** App. B.6 sweeps $\alpha$, $K$, $s_{\max}$ but NOT $\mu$ or $\beta$. User has μ EMA as a known pre-submission code fix; reviewer scrutiny on this is a real risk.

7. **"If it's loss-side scaling applied to encoder gradients, why isn't this equivalent to per-modality learning rates (MSLR)?"**
   Paper cites MSLR (yao2022mslr) but doesn't show a head-to-head. A reviewer will notice.

8. **Citation paraphrase risks (added 2026-05-01 audit).**
   - L98 "Under mild assumptions on latent structure" cites huang2021multimodal — Huang et al. require non-trivial latent-space conditions, not "mild." Soften to "structural assumptions."
   - L98 "modality laziness ... wang2020gblending,peng2022ogmge" — terminology is from du2023suppression (already cited in §2). Wrong attribution.
   - L150 "exact durations ... huang2022modality" — paper proves bounds, not exact values. Replace with "explicit bounds."
   - L170 "OGM-GE ... refined by AGM, CGGM, MLGM" misrepresents AGM (Shapley-based, not refinement of OGM-GE). Reword as "extends with alternative imbalance signals."
   - L104 "wei2024opm's boost-only variant fails" — citation refers to OPM's OGM* ablation, not the parent paper's main contribution. Add "(OGM* ablation)" qualifier.

9. **Key-point delivery gaps (added 2026-05-01 audit).**
   - §3.4 composability is asserted, not proved for the general class of throttling methods. Add a bounded-scaling lemma or weaken to OGM-GE-only.
   - §4.4 utilization-gap analysis is 2.5 sentences — under-supports headline contribution bullet 4. The "5.4× reduction" lacks a baseline-gap reference number in the figure caption.
   - §4.3.1 "four operating regimes" — Food101 frozen regime has zero analytical text in main body.
   - §5 conflates AVE main-results (+0.87) with AVE from-scratch (-4.56) in the "$-0.6$ to $-4.6$ pp" range. Misleading.
   - MILES baseline is cited only in §4 (Table 1, footnote) — no §1 or §2 mention. Reviewer-objection trigger.

10. **§3 footnote at L306 confesses code-vs-paper mismatch on $\epsilon$-guard.** Hard branch in code coincides with Eq. 6 only at $\Delta=0$, not over $0 < \Delta < \epsilon$. Either fix equation to piecewise definition or weaken footnote claim. (Tracked separately in project_section3_pending_fixes.md as Fix #5.)
