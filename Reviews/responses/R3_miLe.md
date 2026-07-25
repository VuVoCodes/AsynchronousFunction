# Response to Reviewer miLe (Reject, 2)

Thank you for a technically sharp review. Your W3 pushed us to run a measurement we should have had all along, and Q7 helped us realize our Table 1 layout invites a misreading.

**[Q7, MAJOR] The OGM-GE paper reports 61.59 on CREMA-D. You report 69.14. Where is this discrepancy coming from?**

**Response.** The OGM-GE paper (Peng et al., CVPR 2022) reports **61.9%** on CREMA-D (their Tables 1 and 2, concatenation fusion). The 61.59 figure is the joint-training baseline row of our own Table 1, and we apologize that the adjacency of these rows invites this reading. Their protocol uses **one visual frame** per clip (their Section 4.2). Three facts resolve the comparison:

1. **We reproduce their number at their operating point.** Our 1-frame ablation (Table 2) yields OGM-GE at $62.47 \pm 1.42$, consistent with their published 61.9.
2. **Our main table uses 3 frames at 3 fps**, following more recent baselines (MILES, InfoReg, Section 4.1). Richer visual input raises all methods, and OGM-GE benefits most (+6.7 pp over its 1-frame result). Follow-up works under richer sampling report OGM-GE at 72-75%, and our 69.14 sits between the two published operating points.
3. **The direction of the difference works against us, not for us.** Our reproduced OGM-GE baseline is 7.2 pp stronger than the original paper's number, which makes our composition's margin harder, not easier.

The revised Table 1 note is presented as follows.

*Baseline denotes joint training with unmodified gradients (not a published OGM-GE result). All methods are re-run under the matched 3-frame protocol of Section 4.1. OGM-GE's original 1-frame result (61.9) is reproduced at $62.47 \pm 1.42$ in Table 2.*

**[W1] This paper would be better served by contribution type "General" instead of "Concept and Feasibility", as the scope of the proposed method is small enough to be validated in a single paper.**

**Response.** We appreciate the perspective and defer to the AC's judgment on the appropriate designation.

**[W2] The main results table shows improvements of < 0.5 pp over the best baseline on 7 out of 8 datasets, which is extremely minor.**

**Response.** You are right that seven of the eight per-dataset margins are small. We suggest, though, that best-versus-best framing understates the table, for two reasons:

1. **On the one benchmark with severe imbalance and full gradient flow, the margin is established under a pre-registered protocol.** During this discussion period we extended the comparison to $n=15$ per arm under a pre-registered stopping rule (final five seeds fixed in advance, all 15 reported, no further additions). PGGB+OGM-GE gives $71.00 \pm 1.46$ versus $69.40 \pm 1.25$ for the $\alpha=0$ arm: **+1.60 pp, 95% Welch CI [0.59, 2.62], Welch $p=0.0032$** (Mann-Whitney $p=0.0054$, Cohen's $d=1.18$, paired t $p=0.016$, sign 11+/3-/1=). We state the trajectory plainly: +2.31 (original 5), +2.06 ($n=10$), +1.60 ($n=15$): the original seeds were favorable draws and the stabilized effect is smaller than first reported. No 5-seed batch is significant alone (~38% power at $d \approx 1.2$), which is why we pre-committed to the pooled endpoint (88% power), whose CI excludes zero. The camera-ready adopts the $n=15$ statistics as the headline throughout, with all 30 per-seed values in the appendix.
2. **Near-neutrality elsewhere is the designed behavior, and the relevant contrast is that prior methods regress there.** By self-attenuation (Prop. 2), PGGB withdraws when the utilization gap is small. On the four low-imbalance benchmarks PGGB is best on all four, while OGM-GE regresses on three (KS -1.80 pp, Twitter15 -0.27, Sarcasm -0.59). A method that gains where imbalance exists and by design withdraws where it does not is the intended contribution.

In summary:

| Regime | Outcome |
|---|---|
| Severe imbalance, full gradient flow (CREMA-D) | +1.60 pp over strongest baseline at pre-registered $n=15$, CI [0.59, 2.62], $p=0.0032$ |
| Low imbalance (AVE, KS, Twitter15, Sarcasm) | PGGB best method on all four (margins within seed noise); OGM-GE regresses on three |
| High imbalance under Adam (sentiment) | PGGB's marginal effect within seed std; actuation attenuated (measured, W3) |

**[W3] The proposed method of scaling the gradients should not be expected to give significant gains for the Adam optimizer, which is scale-invariant / unit-less. 4 out of 8 datasets use Adam and as expected show barely any improvement. OGM-GE has plausible improvement even with Adam because the GE component goes beyond mere gradient scaling.**

**Response.** You are right about the mechanism, and we measured it. New per-step instrumentation records the applied scale, post-scaling gradient norm, and the norm of the actual parameter update per encoder ($\alpha=0.75$ versus $\alpha=0$, matched seeds):

| Pipeline (optimizer) | boost scale | grad-norm ratio | **update-norm ratio** |
|---|---|---|---|
| CREMA-D (SGD), most-boosted modality | 1.64 | 1.52 | **1.50** |
| CMU-MOSI (Adam), most-boosted modality | 1.48 | 1.38 | **1.17** |
| CMU-MOSI (SGD control, new), most-boosted modality | 1.45 | 1.53 | **1.47** |

1. Under SGD the boost transmits to parameters essentially one-to-one.
2. Under Adam roughly two-thirds of the applied boost is absorbed by second-moment normalization ($1.48\times$ scale, $1.17\times$ actual updates).
3. **New control run testing your mechanism directly.** If the attenuation is an optimizer property rather than a method or data property, SGD on the identical MOSI pipeline should restore transmission. It does, modality by modality: under SGD every update-norm ratio tracks its applied scale (the most-boosted modality transmits at **1.47** against its 1.45 scale, and the modality boosted hardest under Adam transmits at 1.18 against its 1.29 scale), whereas under Adam every ratio is pulled toward 1 regardless of scale. The controller boosts different modalities under the two optimizers (it reacts to observed dynamics), so each row reports that run's most-boosted modality. Ratios are stochastic means and momentum-SGD is not a pure multiply, so tracking is approximate. We will provide the full per-modality table for both optimizers.
4. Consistently, all headline effects arise on SGD pipelines, and OGM-GE's Adam results similarly rely on its GE noise term rather than pure scaling, as you observe.
5. We will state the optimizer dependence in Sections 3.3 and 5, add both tables to the appendix, and list optimizer-state-aware actuation as future work.

**[W4] The ablation table (Table 2) shows that OGM-GE alone recovers most of the gap between PGGB+OGM-GE and the baseline (2.17 pp) and PGGB's contribution is minor (0.22 pp), well within the per-seed standard deviation.**

**Response.**

1. The 0.22 pp figure comes from the 1-frame ablation (Table 2), where the visual modality is deliberately information-starved. Boosting amplifies the weak encoder's gradient signal but cannot create information the input does not carry: with a single frame there is little to learn. This complements rather than contradicts Section 4.3's "amplifies method differences" rationale: throttling acts on the information-rich dominant modality and is amplified at 1-frame, while boosting needs weak-modality input and is starved.
2. Under the main 3-frame protocol the same decomposition gives +1.60 pp at the pre-registered $n=15$ (CI excluding zero, see W2), a $7\times$ larger increment isolating boost actuation within the OGM-GE composition: the $\alpha=0$ arm holds the entire probe pipeline active and differs only in the multiplicative scale. We do not claim this increment generalizes across throttlers (App. B.7 marks non-OGM-GE compositions within noise).
3. We agree the isolating decomposition should live at the main operating point: in the camera-ready we will promote the 3-frame decomposition (with the $n=15$ statistics) to the primary ablation table and retain the 1-frame variant as the information-availability ablation, with its reading made explicit.

**[Q1-Q6] Clarity questions.**

**Response.**

1. **Q1 (L17):** 2 = audio+visual (CREMA-D, AVE, KS) and text+image (Twitter15, Sarcasm), 3 = text+audio+vision (CMU-MOSI and the sentiment benchmark), 4 = four MRI sequences (BraTS 2021). We will enumerate this in Section 4.1.
2. **Q2 (L131, $g$ undefined):** correct, the fusion classifier $g$ is used before being formally defined. We will add the definition after Eq. 1.
3. **Q3 (last paragraph of 3.1):** we will rewrite it as: "All encoder gradients share the factor $\partial L / \partial g$, which a dominant modality shapes. Imbalance signals computed from $L$ therefore measure the weak modality through a channel the strong modality controls."
4. **Q4 (L147-148):** the sentence is indeed unclear and we will replace it with: "$P_m$ therefore reflects the representation quality of $z_m$ alone, independent of how the fusion head weights modality $m$."
5. **Q5 (L172, $s_m$ at balance):** we are glad to clarify this subtle case. $s_m$ is well-defined at exact balance: when all $\bar{P}$ are equal, the numerator of Eq. 7 is zero for every $m$ while the denominator equals $\epsilon > 0$, so $w_m = 0$ and $s_m = 1$ exactly (no intervention). We will add one sentence stating this.
6. **Q6 (takeaways of Section 3.5):** the propositions are safety properties: bounded intervention (Prop. 1), self-attenuation at balance (Prop. 2), and the standard nonconvex SGD descent guarantee with variance inflated by at most $s_{\max}^2 = 4$ (Prop. 3). They intentionally claim neither faster convergence nor guaranteed gap closure, and the abstract and introduction will be reworded.

Your review directly produced the Adam measurement and the pre-registered seed extension. We would welcome any further questions and are glad to run additional analysis in this window.
