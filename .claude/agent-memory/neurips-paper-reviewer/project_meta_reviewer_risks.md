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
