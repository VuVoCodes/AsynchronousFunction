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

**Response.** We doubled the seed count for both arms of the central comparison on CREMA-D 3-frame (five original seeds plus five fresh seeds, identical protocol):

| | OGM-GE + probes active, boost off ($\alpha=0$) | PGGB+OGM-GE ($\alpha=0.75$) |
|---|---|---|
| Accuracy, $n=10$ | $69.25 \pm 1.34$ | $71.30 \pm 1.48$ |

1. The two arms hold the entire probe pipeline identical and differ only in the multiplicative scale, so the contrast isolates boost actuation.
2. Difference **+2.06 pp, 95% Welch CI [0.73, 3.38]**, Welch t-test $p=0.0044$, Mann-Whitney $p=0.0029$, Cohen's $d=1.46$. Seed-matched tests agree: paired t $p=0.026$, Wilcoxon $p=0.020$, with 8 of 10 seed pairs favoring the composition (1 tie, 1 inversion).
3. For transparency, the five fresh seeds alone give +1.80 pp in the same direction (4 of 5 pairs positive, one tie), not significant at $n=5$ in isolation (CI [-0.42, 4.02]); at the pooled $n=10$ the confidence interval excludes zero.

**[AC point 4, R-tQk1] Early probe trajectories and warm-up.**

**Response.** From per-event instrumentation of all 500 probe evaluations ($K=20$) on CREMA-D: at the first probe event (iteration 19) both probes are at chance level and the smoothed ordering is briefly inverted by 0.7 pp. From the second event (iteration 39) the dominant modality is identified correctly and the ordering never inverts again until late training, when the utilization gap has been closed by design. Worst-case misdirection exposure is a single $K$-step window with smoothed scale at most $1+\mu\alpha$ (approximately 1.23). No explicit warm-up is needed: scales initialize at 1 and the EMA ramp acts as an implicit warm-up. We will add the early-window figure to the appendix.

**[R-tQk1] Wall-clock time and memory.**

**Response.** CREMA-D 3-frame on one RTX 4090 (mean s/epoch over 7 epochs): baseline 17.43, PGGB 17.71 (**+1.6%**), PGGB+OGM-GE 18.71 (+7.4%, of which about 5.8 pp is OGM-GE's own per-modality contribution computation, not the probes). Peak GPU memory is unchanged across the three configurations (~12.2 GB, differences within allocator noise). Each linear probe adds ~3K parameters.

**[AC point 5] Clarify the dataset-level imbalance score and the choice of the 0.15 threshold.**

**Response.** The dataset-level $\delta$ is the final-epoch EMA-smoothed probe-accuracy gap during baseline training, averaged over 5 seeds (Appendix B.9). We will move the definition into Section 4.1 and add the threshold-sensitivity discussion (every threshold in $[0.150, 0.222)$ yields the identical categorization under the paper's rule that $\delta$ exceed the threshold, Twitter15 is the boundary case, and the CMU-MOSEI value is under re-measurement on re-verified data). Details in the response to gN93.

**[AC point 6] Revise the theoretical claims so they match what the propositions establish.**

**Response.** We agree the propositions are safety properties, not gap-closure or rate results, and Section 3.5 already labels them as such. We will align the abstract and introduction wording with this framing in the camera-ready ("we establish three safety properties" rather than any suggestion of a convergence-improvement claim). Details in the responses to tQk1 and gN93.
