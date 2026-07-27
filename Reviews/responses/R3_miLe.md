# Response to Reviewer miLe (Reject, 2)

Thank you for a technically sharp review. Your W3 pushed us to run a measurement we should have had all along, and Q7 helped us realize our Table 1 layout invites a misreading. Supporting results are from the submitted paper unless tagged **[NEW]** (run in this window, for the revision).

**[Q7, MAJOR] The OGM-GE paper reports 61.59 on CREMA-D. You report 69.14. Where is this discrepancy coming from?**

**Response.** The OGM-GE paper (Peng et al., CVPR 2022) reports **61.9%** on CREMA-D (their Tables 1 and 2, concatenation fusion). The 61.59 figure is the joint-training baseline row of our own Table 1, and we apologize that the adjacency of these rows invites this reading. Their protocol uses **one visual frame** per clip (their Section 4.2). Two facts resolve the comparison:

1. **We reproduce their result at their operating point.** Our 1-frame ablation (Table 2) yields OGM-GE at $62.47 \pm 1.42$, and their published 61.9 lies within one seed std of our mean.
2. **Our main table uses 3 frames at 3 fps**, a matched pipeline shared by all methods in Table 1. The richer input raises OGM-GE by +6.7 pp over its 1-frame result. We will correct Section 4.1's protocol attribution ("following OGM-GE").

The revised Table 1 note is presented as follows.

*Baseline denotes joint training with unmodified gradients (not a published OGM-GE result). All methods are re-run under the matched 3-frame protocol of Section 4.1. Their 1-frame result (61.9) is reproduced at $62.47 \pm 1.42$ in Table 2.*

**[W1] This paper would be better served by contribution type "General" instead of "Concept and Feasibility", as the scope of the proposed method is small enough to be validated in a single paper.**

**Response.** We appreciate the perspective. We selected "Concept and Feasibility" because probe-guided control opens scope beyond what this paper validates, but we are comfortable with the "General" criteria and defer to the AC's judgment.

**[W2] The main results table shows improvements of < 0.5 pp over the best baseline on 7 out of 8 datasets, which is extremely minor.**

**Response.** We thank you for this sharp observation. We agree that margins under 0.5 pp appear small at first glance, and seven of eight are indeed of that order. We respectfully suggest, however, that Table 1's best-versus-best presentation understates it.

For perspective, on the low-imbalance benchmarks competing methods regress rather than gain (OGM-GE: KS -1.80 pp, Twitter15 -0.27, Sarcasm -0.59), while standalone PGGB is best on all four, at +1.61% measured overhead with 3K-parameter probes **[NEW]**, a favorable efficiency trade-off.

On the one benchmark with severe imbalance and trainable encoders, the margin is larger, and we confirmed its robustness: **[NEW]** we extended the comparison to $n=15$ per arm, fixing the final five seeds in advance under a stopping rule (all 15 reported, no further additions). PGGB+OGM-GE gives $71.00 \pm 1.46$ versus $69.40 \pm 1.25$ for the $\alpha=0$ arm: **+1.60 pp, 95% Welch CI [0.59, 2.62], Welch $p=0.0032$**, Mann-Whitney $p=0.0054$, $d=1.18$. The $\alpha=0$ arm matches standalone OGM-GE within seed noise (69.14, Table 1), so this is also the margin over the best baseline. We state the trajectory plainly: +2.31 (original 5), +2.06 ($n=10$), +1.60 ($n=15$): the original seeds were favorable draws and the pooled estimate is smaller than first reported; the two later batches are not individually significant, and the pooled CI excludes zero. The updated version adopts the $n=15$ statistics as the headline, with all 30 per-seed values in the appendix.

In summary:

| Regime | Outcome |
|---|---|
| Severe imbalance, trainable encoders (CREMA-D) | +1.60 pp over the $\alpha=0$ arm at $n=15$ (final 5 seeds pre-registered), CI [0.59, 2.62], $p=0.0032$ |
| Low imbalance (AVE, KS, Twitter15, Sarcasm) | PGGB best method on all four (margins within seed noise); OGM-GE regresses on three |
| High imbalance under Adam (sentiment) | Effect within seed std; attenuation measured, and reversed under SGD (+1.69 pp, W3) |
| Dense prediction (BraTS 2021) | Reported separately as segmentation (Section 4.1); margin small, as you note |

**[W3] The proposed method of scaling the gradients should not be expected to give significant gains for the Adam optimizer, which is scale-invariant / unit-less. 4 out of 8 datasets use Adam and as expected show barely any improvement. OGM-GE has plausible improvement even with Adam because the GE component goes beyond mere gradient scaling.**

**Response.** Thank you for this observation. You are right about the mechanism, and we measured it. **[NEW]** Additional per-step instrumentation records the applied scale, post-scaling gradient norm, and actual update norm per encoder ($\alpha=0.75$ versus $\alpha=0$, matched seeds; each row reports that run's most-boosted modality, with the full per-modality table to follow in the appendix):

| Pipeline (optimizer) | boost scale | grad-norm ratio | **update-norm ratio** |
|---|---|---|---|
| CREMA-D (SGD), most-boosted modality | 1.64 | 1.52 | **1.50** |
| CMU-MOSI (Adam), most-boosted modality | 1.48 | 1.38 | **1.17** |
| CMU-MOSI (SGD control), most-boosted modality | 1.45 | 1.53 | **1.47** |

The table confirms your prediction: under SGD the applied boost reaches the parameter updates (1.50 update ratio at 1.64 applied), under Adam second-moment normalization absorbs roughly two-thirds of it (1.17 at 1.48), and switching the identical MOSI pipeline to SGD restores transmission (1.47 at 1.45). **[NEW]** To test the accuracy consequence, we ran the same composed contrast at full budget under both optimizers, 5 matched seeds.

| MOSI ($n=5$, composed) | $\alpha=0$ | $\alpha=0.75$ | Boost effect |
|---|---|---|---|
| Adam | $72.68 \pm 0.99$ | $72.60 \pm 1.05$ | $-0.09$ pp ($p=0.90$) |
| SGD (control) | $55.89 \pm 1.03$ | $57.58 \pm 1.04$ | **+1.69 pp** ($p=0.032$, 5/5 seeds) |

SGD is not retuned for this pipeline (hence its lower absolute accuracy), so the contrast is within-optimizer only. This changes no reported number; it scopes the mechanism: all headline effects arise on SGD pipelines, and OGM-GE's Adam results similarly rely on its GE noise term rather than pure scaling, as you observe. We will state the optimizer dependence in Sections 3.3 and 5, add both tables to the appendix, and list optimizer-state-aware actuation as future work.

**[W4] The ablation table (Table 2) shows that OGM-GE alone recovers most of the gap between PGGB+OGM-GE and the baseline (2.17 pp) and PGGB's contribution is minor (0.22 pp), well within the per-seed standard deviation.**

**Response.**

1. You are right that Table 2 as presented makes PGGB's contribution look marginal. The 0.22 pp figure comes from the 1-frame ablation, where the visual modality is information-starved: boosting amplifies the weak encoder's gradient signal but cannot create information the input does not carry. Throttling, which acts on the information-rich dominant modality, is amplified at 1-frame, while boosting is starved there.
2. Under the main 3-frame protocol the same decomposition gives +1.60 pp at $n=15$ (CI excluding zero, see W2), versus 0.22 pp (within noise) at 1-frame, isolating boost actuation within the OGM-GE composition: the $\alpha=0$ arm holds the entire probe pipeline active and differs only in the multiplicative scale. We do not claim this increment generalizes across throttlers (App. B.7 marks non-OGM-GE compositions within noise).
3. We agree the isolating decomposition should live at the main operating point: in the updated version, we will promote the 3-frame decomposition (with the $n=15$ statistics) to the primary ablation table and retain the 1-frame variant as the information-availability ablation.

**[Q1-Q6] Clarity questions.**

**Response.** We appreciate the reviewer's careful reading and constructive feedback, which helped us identify several minor technical details requiring clarification. Each is addressed below.

1. **Q1 (L17):** The 2-4 modalities are: 2 for audio+visual (CREMA-D, AVE, KS) and text+image (Twitter15, Sarcasm), 3 for text+audio+vision (CMU-MOSI and CMU-MOSEI), 4 for four MRI sequences (BraTS 2021). We will enumerate this in Section 4.1.
2. **Q2 (L131, $g$ undefined):** Thank you for pointing out this subtle omission: the fusion classifier $g$ is used before being formally defined. We will add the definition after Eq. 1.
3. **Q3 (last paragraph of 3.1):** we will rewrite it as: "Each encoder's fusion-loss gradient passes through the fusion head's per-modality weighting $\partial \hat{y} / \partial z_m$, which a dominant modality shapes. Imbalance signals computed from $L$ therefore measure the weak modality through a channel the strong modality controls."
4. **Q4 (L147-148):**  the sentence could be a bit clearer. The full replacement sentence (closing the probe paragraph of Section 3.2): *"This is precisely why decoupling matters: the probe accuracy $P_m$ reflects the representation quality of $\mathbf{z}_m$ alone, independent of how the fusion head $g$ weights modality $m$."*
5. **Q5 (L172, $s_m$ at balance):** our text did not state the $\epsilon$-guarded case, so this could not be inferred from the paper as written. $s_m$ is in fact well-defined at exact balance: when all $\bar{P}$ are equal, the numerator of Eq. 7 is zero for every $m$ while the denominator equals $\epsilon > 0$, so $w_m = 0$ and $s_m = 1$ exactly (no intervention).  We will add one sentence to the manuscript for clarifying this.
6. **Q6 (takeaways of Section 3.5):** the intended takeaway for Propositions 1-3 are safety properties (bounded intervention, neutrality at exact balance, standard SGD descent preserved), so adding PGGB cannot destabilize the training it composes with.

Thank you again for a review that directly produced the optimizer controls and the seed extension. We welcome further questions in this window.
