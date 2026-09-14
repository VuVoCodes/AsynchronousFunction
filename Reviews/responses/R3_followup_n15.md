# Follow-up to Reviewer miLe: pre-registered n=15 results and the full transmission table

Dear Reviewer miLe,

As committed in our response, we post the pre-registered $n=15$ extension and the full per-modality transmission table, and answer your remaining points in order.

**1. The pre-registered $n=15$ isolation result.** The five batch-3 seeds (1111/2222/3333/4444/6666) were fixed in advance of any run, all results are reported regardless of outcome, and no further seeds will be added. Pooled over all 15 seeds per arm (sample std, ddof=1):

| | $\alpha=0$ (OGM-GE + probes, boost off) | $\alpha=0.75$ (PGGB+OGM-GE) |
|---|---|---|
| Accuracy, $n=15$ | $69.40 \pm 1.25$ | $71.00 \pm 1.46$ |

Difference **+1.60 pp, 95% Welch CI [0.59, 2.62]**, Welch $p=0.0032$, Mann-Whitney $p=0.0054$, Cohen's $d=1.18$. Seed-matched tests agree: paired t $p=0.016$, Wilcoxon $p=0.022$, sign test 11+/3-/1=.

We state the full trajectory plainly rather than leave it to be inferred: the estimate has moved +2.31 (original 5 seeds) to +2.06 ($n=10$) to +1.60 ($n=15$). The original five seeds were favorable draws, and the stabilized effect is smaller than first reported. Batch 3 alone gives +0.70 and is not significant on its own, which is expected: at $d \approx 1.2$ a 5-seed batch has roughly 38% power, which is precisely why we pre-committed to the pooled endpoint (88% power at $n=15$) rather than to any single batch. What is stable at every accumulation point is the direction and the significance: the confidence interval excludes zero at $n=10$ and at $n=15$ under the stopping rule, and 11 of 15 seed-matched pairs favor the composition. The camera-ready will adopt the $n=15$ statistics as the headline everywhere it appears (Table 1, Sections 4.2-4.3, Conclusion), with all 30 per-seed values in the appendix.

**2. The full per-modality transmission table.** As promised, CMU-MOSI under both optimizers, identical data, architecture, and seed, $\alpha=0.75$ versus $\alpha=0$, ratios of means over matched post-warmup steps:

| Modality | Adam: scale | Adam: grad ratio | Adam: **update ratio** | SGD: scale | SGD: grad ratio | SGD: **update ratio** |
|---|---|---|---|---|---|---|
| text | 1.12 | 1.06 | **0.97** | 1.45 | 1.53 | **1.47** |
| audio | 1.48 | 1.38 | **1.17** | 1.29 | 1.27 | **1.18** |
| visual | 1.31 | 1.31 | **1.11** | 1.29 | 1.31 | **1.32** |

This resolves the label question from our earlier table at the level of individual encoders. Under SGD, every modality's update ratio tracks its applied scale (1.45/1.47, 1.29/1.18, 1.29/1.32). Under Adam, every update ratio is pulled toward 1 regardless of scale, including text at scale 1.12 transmitting nothing (0.97). The controller assigns its largest boost to different modalities under the two optimizers because it reacts to the learning dynamics it observes (audio lags under Adam, text lags earliest under SGD at this learning rate). The transmission property, which is what your W3 concerns, holds or fails modality-by-modality with the optimizer, not with the modality identity.

**3. On stacking PGGB on an OGM-GE "tuned to the 72-75% ceiling."** Those figures come from follow-up works that use richer frame-sampling protocols, not from tuning OGM-GE within a fixed protocol, so there is no 72-75% OGM-GE to stack on under a matched pipeline: reaching that ceiling means changing the input protocol, and changing it for one method would un-match exactly the comparison Q7 asked us to make fair. We verified both ends of the protocol axis instead: at OGM-GE's own 1-frame operating point we reproduce its published number (62.47 +/- 1.42 vs 61.9), and at the matched 3-frame point the isolation contrast above holds with both arms sharing the identical protocol. The isolation design is protocol-invariant by construction (both arms always share whatever protocol is chosen), so if you would find a richer-protocol version decisive, we are glad to run the same $\alpha=0$ versus $\alpha=0.75$ contrast at a higher frame rate and report it, within this window if time permits or in the camera-ready appendix otherwise.

**4. Primary ablation.** As stated in our response, the camera-ready promotes the 3-frame decomposition, now with the $n=15$ statistics above, to the primary ablation table, with the 1-frame variant retained as the information-availability ablation and its throttling-amplified versus boost-starved reading stated explicitly.

Your review shaped three of the strongest additions this paper gained during the discussion period: the Adam transmission measurement, the SGD control, and the pre-registered seed extension. If these results and commitments resolve the remaining concerns, we would be grateful if you would consider revisiting the assessment, and we remain glad to run further analysis in the time remaining.
