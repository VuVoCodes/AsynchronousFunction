# Global response (all reviewers and AC)

Thank you all for the careful reading. The reviews converge on a positive view of the core idea (decoupled probe monitoring as an online optimization signal) while asking for stronger empirical attribution, an explanation of the CREMA-D reproduction, and clearer scoping of the theory. During the discussion period we ran new experiments that address every point the AC lists. We summarize them here, keyed to the AC's numbered points, and give details in the individual responses.

**[AC point 1] Explain the CREMA-D baseline discrepancy raised by miLe.**

**Response.** The figure 61.59 is our joint-training baseline, not the OGM-GE paper's number (they report 61.9 under a 1-frame protocol). Our 1-frame ablation reproduces OGM-GE at $62.47 \pm 1.42$, statistically consistent with their published result, and our main table uses the richer 3-frame protocol of more recent baselines. Full resolution in the response to Reviewer miLe (Q7).

**[AC point 2] Clarify whether and how PGGB changes effective updates under Adam.**

**Response.** We instrumented training to record, at every optimizer step, the applied boost scale, the post-scaling gradient norm, and the norm of the actual parameter update per modality encoder, comparing $\alpha=0.75$ against $\alpha=0$ at matched seeds:

| Pipeline (optimizer) | boost scale | gradient-norm ratio | **update-norm ratio** |
|---|---|---|---|
| CREMA-D (SGD), weak modality | 1.64 | 1.52 | **1.50** |
| CMU-MOSI (Adam), weak modality | 1.48 | 1.38 | **1.17** |

1. Under SGD the boost transmits to parameter updates essentially one-to-one.
2. Under Adam, second-moment normalization absorbs roughly two-thirds of the applied boost ($1.48\times$ scale, $1.17\times$ actual updates), so the actuation is attenuated but not eliminated (the residual arises from the time-varying scale interacting with Adam's moment estimates).
3. Reviewer miLe's mechanism argument is correct, and it aligns with our results: all headline effects arise on SGD pipelines, and CMU-MOSI, our one high-imbalance Adam pipeline, is the cleanest demonstration of the attenuation, with the boost engaged in gradient space but muted in parameter space. We will state this optimizer dependence explicitly and add the measurement to the appendix.

**[AC point 3] Isolate the contribution of PGGB from OGM-GE, with uncertainty over their difference.**

**Response.** We tripled the seed count for both arms of the central comparison on CREMA-D 3-frame: five original seeds, five fresh seeds, then a final five fixed in advance under a pre-registered stopping rule (all 15 reported, no further additions):

| | OGM-GE + probes active, boost off ($\alpha=0$) | PGGB+OGM-GE ($\alpha=0.75$) |
|---|---|---|
| Accuracy, $n=15$ | $69.40 \pm 1.25$ | $71.00 \pm 1.46$ |

1. The two arms hold the entire probe pipeline identical and differ only in the multiplicative scale, so the contrast isolates boost actuation.
2. Difference **+1.60 pp, 95% Welch CI [0.59, 2.62]**, Welch $p=0.0032$, Mann-Whitney $p=0.0054$, Cohen's $d=1.18$. Seed-matched tests agree: paired t $p=0.016$, Wilcoxon $p=0.022$, sign 11+/3-/1=.
3. **We surface for all reviewers that this revises the paper's headline downward.** The trajectory is +2.31 (original 5 seeds), +2.06 ($n=10$), +1.60 ($n=15$): the original seeds were favorable draws and the stabilized effect is smaller than first reported. No 5-seed batch is significant alone (~38% power at $d \approx 1.2$), which is why we pre-committed to the pooled endpoint (88% power), whose confidence interval excludes zero. The camera-ready adopts the $n=15$ statistics everywhere the +2.31 currently appears.

**[AC point 4, R-tQk1] Early probe trajectories and warm-up.**

**Response.** From per-event instrumentation of all 500 probe evaluations ($K=20$) on CREMA-D: at the first probe event (iteration 19) both probes are at chance level and the smoothed ordering is briefly inverted by 0.7 pp. From the second event (iteration 39) the dominant modality is identified correctly (+1.2 pp, growing to +12.8 pp by the end of epoch 3) and the ordering inverts only twice more in the entire run, exactly where the smoothed gap passes through zero and self-attenuation holds the scales near 1. A per-event table is in the response to tQk1. Worst-case misdirection exposure is a single $K$-step window with smoothed scale at most $1+\mu\alpha$ (approximately 1.23). No explicit warm-up is needed: scales initialize at 1 and the EMA ramp acts as an implicit warm-up. We will add the early-window figure to the appendix.

**[R-tQk1] Wall-clock time and memory.**

**Response.** CREMA-D 3-frame on one RTX 4090 (mean s/epoch over 7 epochs): baseline 17.43, PGGB 17.71 (**+1.6%**), PGGB+OGM-GE 18.71 (+7.4%, of which about 5.8 pp is OGM-GE's own per-modality contribution computation, not the probes). Peak GPU memory is unchanged across the three configurations (~12.2 GB, differences within allocator noise). Each linear probe adds ~3K parameters.

**[AC point 5] Clarify the dataset-level imbalance score and the choice of the 0.15 threshold.**

**Response.** The dataset-level $\delta$ is the final-epoch EMA-smoothed probe-accuracy gap during baseline training, averaged over 5 seeds (Appendix B.9). We will move the definition into Section 4.1 and add the threshold-sensitivity discussion: every threshold in $[0.150, 0.178)$ yields the identical categorization under the paper's rule that $\delta$ exceed the threshold, with Twitter15 ($0.150 \pm 0.096$) and CMU-MOSEI ($0.178 \pm 0.086$) as the seed-sensitive boundary cases. Details in the response to gN93.

**[AC point 6] Revise the theoretical claims so they match what the propositions establish.**

**Response.** We agree the propositions are safety properties, not gap-closure or rate results, and Section 3.5 already labels them as such. We will align the abstract and introduction wording with this framing in the camera-ready ("we establish three safety properties" rather than any suggestion of a convergence-improvement claim). Details in the responses to tQk1 and gN93.

**Summary of camera-ready revisions**

So the committed changes are auditable in one place, the camera-ready will include:

1. **Table 1 (CREMA-D headline):** replaced by the pre-registered $n=15$ statistics ($71.00 \pm 1.46$, +1.60 pp, 95% CI [0.59, 2.62]) everywhere the +2.31 currently appears (abstract, introduction, main results, conclusion, figure captions), with all 30 per-seed values in the appendix, and explicit row labeling so the baseline row cannot be mistaken for a published OGM-GE number (miLe Q7).
2. **Section 4.1:** measured wall-clock overhead (+1.6%) replaces the ~1% estimate, and the dataset-level $\delta$ definition moves into the main text with the threshold-sensitivity discussion (gN93 Q2, AC point 5).
3. **Sections 3.3 and 5:** explicit statement of the optimizer dependence of gradient scaling, with the Adam update-norm measurement added to the appendix and optimizer-state-aware actuation listed as future work (miLe W3, AC point 2).
4. **Abstract and introduction:** theory characterized as three safety properties, with the rate-level analysis stated as open (tQk1 W3, miLe Q6, AC point 6).
5. **Appendix (probe diagnostics):** early-window trajectory figure and warm-up discussion from the per-event instrumentation (tQk1 W2, AC point 4).
6. **Limitations:** expanded to cover safety-guarantee scoping at high imbalance, threshold sensitivity with the Twitter15 and CMU-MOSEI boundary cases, and the undertrained-versus-intrinsically-less-informative distinction (gN93).
7. **Clarity fixes** (miLe Q1-Q6): definition of the fusion classifier $g$ after Eq. 1, rewritten closing paragraph of Section 3.1, replacement of the unclear L147 sentence, one sentence on $s_m$ at exact balance, and enumeration of the 2-4 modality configurations in Section 4.1.
