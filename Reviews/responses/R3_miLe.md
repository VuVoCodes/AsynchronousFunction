# Response to Reviewer miLe (Reject, 2)

Thank you for a technically sharp review. The Adam observation (W3) is correct and we now quantify it with new measurements. The reproduction question (Q7) appears to stem from a figure we could not locate in the OGM-GE paper, and we can resolve the comparison completely, including a reproduction at the original paper's own operating point. We address the four weaknesses and all questions below.

**[Q7, MAJOR] The OGM-GE paper reports 61.59 on CREMA-D. You report 69.14. Where is this discrepancy coming from?**

**Response.** The OGM-GE paper (Peng et al., CVPR 2022) reports **61.9%** on CREMA-D (their Tables 1 and 2, concatenation fusion), not 61.59% (61.59 is our joint-training baseline in our Table 1, which may be the source of the confusion). Their protocol uses **one visual frame** per clip (their Section 4.2: "For CREMA-D, we extract 1 frame from each of the clip"). Three facts resolve the comparison:

1. **We reproduce their number at their operating point.** Our 1-frame ablation (Table 2) yields OGM-GE at $62.47 \pm 1.42$, statistically consistent with their published 61.9.
2. **Our main table uses 3 frames at 3 fps**, following the protocol of more recent baselines (MILES, InfoReg), as stated in Section 4.1. Richer visual input raises all methods, and OGM-GE benefits most (+6.7 pp over its 1-frame result), because gradient modulation is more effective when the weak modality carries more information. Follow-up works under richer sampling report OGM-GE at 72-75%, and our 69.14 sits between the two published operating points, as disclosed in the Reproduction protocol paragraph.
3. **The direction of the difference works against us, not for us.** Our reproduced OGM-GE baseline is 7.2 pp stronger than the original paper's number, which makes the +2.31 pp margin of our composition harder to achieve, not easier.

**[W3] The proposed method of scaling the gradients should not be expected to give significant gains for the Adam optimizer, which is scale-invariant / unit-less. 4 out of 8 datasets use Adam and as expected show barely any improvement. OGM-GE has plausible improvement even with Adam because the GE component goes beyond mere gradient scaling.**

**Response.** The reviewer is right about the mechanism, and we measured it. New per-step instrumentation records the applied scale, post-scaling gradient norm, and the norm of the actual parameter update per encoder ($\alpha=0.75$ versus $\alpha=0$, matched seeds):

| Pipeline (optimizer) | boost scale | grad-norm ratio | **update-norm ratio** |
|---|---|---|---|
| CREMA-D (SGD), weak modality | 1.64 | 1.52 | **1.50** |
| CMU-MOSI (Adam), weak modality | 1.48 | 1.38 | **1.17** |

1. Under SGD the boost transmits to parameters essentially one-to-one.
2. Under Adam roughly two-thirds of the applied boost is absorbed by second-moment normalization (a $1.48\times$ scale yields only a $1.17\times$ change in actual parameter updates), with a residual effect from the time-varying scale interacting with the moment estimates (a constant scale would cancel asymptotically, but $\bar{s}$ is refreshed every $K$ steps and EMA-smoothed).
3. This is fully consistent with our results: all headline effects arise on SGD pipelines, and CMU-MOSI, the one high-imbalance Adam pipeline, is the cleanest demonstration of the attenuation, with the boost engaged in gradient space but muted in parameter space.
4. We note that OGM-GE's Adam results similarly rely on its GE noise term rather than pure scaling, as the reviewer observes.
5. We will state the optimizer dependence explicitly in Sections 3.3 and 5, add the measurement table to the appendix, and list optimizer-state-aware actuation (for example, scaling the update rather than the gradient) as future work.

**[W2] The main results table shows improvements of < 0.5 pp over the best baseline on 7 out of 8 datasets, which is extremely minor.**

**Response.** We respectfully suggest the per-dataset best-versus-best framing understates what the table shows, for two reasons:

1. **On the one benchmark with severe imbalance and full gradient flow, the margin is large and now firmly established.** With $n=10$ seeds per arm (5 new seeds added during rebuttal), PGGB+OGM-GE exceeds the strongest baseline by **+2.06 pp, 95% CI [0.73, 3.38], Welch $p=0.0044$** (Mann-Whitney $p=0.0029$, Cohen's $d=1.46$, 8 of 10 seed-matched pairs positive). For full transparency, the five fresh seeds alone give +1.80 pp in the same direction (4 of 5 pairs positive, one tie), which at $n=5$ does not reach significance on its own (CI [-0.42, 4.02]); the pooled $n=10$ estimate is the appropriate test and it is decisive.
2. **Near-neutrality elsewhere is the designed behavior, and the relevant contrast is that prior methods regress there.** By self-attenuation (Prop. 2), PGGB withdraws when the utilization gap is small. On the four low-imbalance benchmarks PGGB is the best-performing method on all four, while OGM-GE regresses on three of them (KS -1.80 pp, Twitter15 -0.27, Sarcasm -0.59). A method that gains where imbalance exists and provably does not intervene where it does not is the intended contribution, and we will make this framing sharper in Section 4.2.

**[W4] The ablation table (Table 2) shows that OGM-GE alone recovers most of the gap between PGGB+OGM-GE and the baseline (2.17 pp) and PGGB's contribution is minor (0.22 pp), well within the per-seed standard deviation.**

**Response.**

1. The 0.22 pp figure comes from the 1-frame ablation (Table 2), where the visual modality is deliberately information-starved. Boosting amplifies the gradient signal of the weak encoder, and it cannot create information the input does not carry: with a single frame there is little for the boosted encoder to learn, which the ablation shows.
2. Under the main 3-frame protocol the same decomposition gives +2.31 pp (now +2.06 pp at $n=10$ with CI excluding zero, see W2), a $10\times$ larger increment attributable to boost actuation alone: the $\alpha=0$ arm holds the entire probe pipeline active and differs only in the multiplicative scale.
3. We will make the information-availability reading of the 1-frame ablation explicit.

**[W1] This paper would be better served by contribution type "General" instead of "Concept and Feasibility", as the scope of the proposed method is small enough to be validated in a single paper.**

**Response.** We are content to defer to the AC on the contribution-type designation and note only that it was selected independently by Reviewer tQk1 as well.

**[Q1-Q6] Clarity questions.**

**Response.**

1. **Q1 (L17, "2-4 modalities"):** 2 = audio+visual (CREMA-D, AVE, KS) and text+image (Twitter15, Sarcasm), 3 = text+audio+vision (CMU-MOSI and the sentiment benchmark), 4 = four MRI sequences (BraTS 2021). We will enumerate this in Section 4.1.
2. **Q2 (L131, $g$ undefined):** correct, the fusion classifier $g$ and prediction $\hat{y} = g([z_1; \dots; z_M]; \phi)$ are used before being formally defined. We will add the definition after Eq. 1.
3. **Q3 (last paragraph of 3.1):** we will rewrite it as: "All encoder gradients share the factor $\partial L / \partial g$. When one modality dominates the fused prediction, this shared factor is shaped primarily by that modality. Any imbalance signal computed from $L$ or its gradients therefore measures the weak modality through a channel the strong modality controls, which is what makes amplifying the weak encoder's gradient unreliable under coupled monitoring."
4. **Q4 (L147-148):** the sentence "not influenced by the dominant modality as the influence can be both ways" is indeed unclear and we will replace it with: "$P_m$ therefore reflects the representation quality of $z_m$ alone, independent of how the fusion head weights modality $m$."
5. **Q5 (L172, $s_m$ at balance):** $s_m$ is well-defined at exact balance. When all $\bar{P}$ are equal, the numerator of Eq. 7 is zero for every $m$ while the denominator equals $\epsilon > 0$, so $w_m = 0$ and $s_m = 1$ exactly (no intervention). The implementation guards the same degenerate case with an explicit branch. We will add one sentence stating this.
6. **Q6 (takeaways of Section 3.5):** the propositions are safety properties: the intervention is bounded (Prop. 1), it vanishes when balance is reached so it cannot destabilize balanced training (Prop. 2), and training retains the standard nonconvex SGD descent guarantee with variance inflated by at most $s_{\max}^2 = 4$ (Prop. 3). They intentionally do not claim faster convergence or guaranteed gap closure, and we will reword the abstract and introduction to match (see response to tQk1).

Given the resolution of Q7, the quantified confirmation of your W3 mechanism, and the $n=10$ statistics for W2/W4, we hope the reviewer will consider revisiting the assessment.
