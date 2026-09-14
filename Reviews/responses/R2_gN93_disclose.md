# Response to Reviewer gN93 — full-disclosure version

Thank you for the careful and constructive review, and for recognizing PGGB as novel, lightweight, and easy to combine with existing methods. The four concerns all target the right places. We first disclose a data-provenance correction that your questions helped us catch.

**[Data provenance correction, proactive disclosure]** During rebuttal-period verification prompted by Q3, we discovered that the preprocessed file used for the column labeled CMU-MOSEI is in fact CH-SIMS (Yu et al., ACL 2020), a Chinese three-modality benchmark from the same MMSA toolkit: 1,368/456/457 splits, 33-d audio, 709-d vision, 3-class labels, not the CMU-MOSEI setup described in Appendix B.1. The retrieval error is ours. All rows of that column were trained and evaluated identically, so the within-column comparison is internally valid, but as a CH-SIMS result. We re-verified the provenance of every other benchmark against canonical sources (including CMU-MOSI, which is genuine); only this column is affected. We re-ran the full method suite on verified genuine CMU-MOSEI (MulT-processed GloVe 300-d text, COVAREP 74-d audio, FACET 35-d vision, 16,265/1,869/4,643 splits), 5 seeds: baseline $62.07 \pm 0.25$, OGM-GE $62.03 \pm 0.25$, PGGB $62.13 \pm 0.18$, PGGB+OGM-GE $62.24 \pm 0.26$. All four lie within 0.21 pp and no pairwise difference is significant: the flat, text-dominant profile, on which PGGB does not interfere. We will relabel the affected column as CH-SIMS, correct every affected descriptor (Tables 1, 8, 9 and Appendix B.1), and report both benchmarks going forward. The re-measured quantities below use the corrected data.

**[Q1] In Section 3.5, the safety properties are described as ensuring that the method does not harm low-imbalance training. What should we expect in high-imbalance settings? Are there any stability guarantees or diagnostic criteria for when boosting may become harmful?**

**Response.** The propositions cover both regimes, in the following division of labor.

1. The descent bound (Prop. 3) is **uniform in the imbalance level**: it holds for any utilization gap. In high-imbalance training the guarantee is therefore that PGGB cannot cause divergence and inflates gradient variance by at most $s_{\max}^2 = 4$ in the worst case.
2. Prop. 2 (self-attenuation) is the complementary low-imbalance guarantee: the intervention vanishes as the gap closes.
3. What is deliberately **not** guaranteed is that boosting closes the gap in high imbalance. That claim is empirical (Section 4.4: weak probe +9.41 pp, strong probe -0.81 pp, post-hoc gap reduced $5.4\times$).
4. On diagnostic criteria: the operating-conditions analysis (Section 4.3.1) identifies the observed failure regime (thin data budget with weak inherent asymmetry, failure localizing to the throttling component), and the monitored $\delta$ and $\bar{s}$ are themselves the online warning signal, loggable at no extra cost.
5. We will state this scoping explicitly in Section 3.5 and expand the Limitations below.

**[Q2] How exactly is the dataset-level delta value computed for imbalance categorization? Is it the final-epoch EMA-smoothed gap averaged over seeds, or some statistic over the full training trajectory? Please clarify this in the main text and justify the threshold (delta > 0.15).**

**Response.** Both halves of this question deserve a precise answer.

1. **Definition.** $\delta$ is the final-epoch EMA-smoothed probe-accuracy gap measured during **baseline** training (no intervention), averaged over 5 seeds (Appendix B.9, Table 9). It is not a statistic over the full trajectory. We will state this definition in Section 4.1 as requested.
2. **Corrected value.** Table 9's $\delta = 0.178 \pm 0.086$ was measured on the mislabeled data. On genuine CMU-MOSEI the re-measured value is $\delta = 0.218 \pm 0.005$ (5 seeds), so CMU-MOSEI remains firmly **high-imbalance** and the paper's categorization is unchanged.
3. **Why 0.15.** The corrected values separate into a low group (AVE 0.017, KS 0.038, Sarcasm 0.076, Twitter15 0.150) and a high group (CMU-MOSEI 0.218, CMU-MOSI 0.222, CREMA-D 0.268), leaving an empty interval between 0.150 and 0.218. Under the paper's rule (high imbalance if $\delta$ exceeds the threshold), every threshold in $[0.150, 0.218)$ produces the identical categorization, so 0.15 is not a tuned quantity: it is a round value inside this gap, and since $\delta$ is measured on baseline training alone, the categorization is fixed independently of any method's results.
4. **Boundary case.** Twitter15 ($0.150 \pm 0.096$) is the boundary case with the largest seed variance, and we will note this explicitly.
5. **Sensitivity.** The trajectory of $\delta$ stabilizes well before the final epoch in our instrumented runs (Appendix B.6 records it every 20 iterations), the per-seed spread is reported in Table 9, and the categorization does not change under linear versus one-hidden-layer probes in preliminary checks. We will add this discussion to the appendix.

**[Q3] For CMU-MOSI and CMU-MOSEI, can the authors provide a deeper unimodal analysis showing whether PGGB improves the audio and visual representations even when final multimodal accuracy changes little?**

**Response.** We ran the requested analysis on all three sentiment benchmarks: post-hoc per-modality linear probes on the saved final checkpoints, mirroring the Appendix B.8 protocol (mean over 5 seeds).

**Genuine CMU-MOSEI**: baseline probes are text 60.2, audio 44.9, vision 46.5. No method moves any modality's probe by more than 1 pp (PGGB: text -0.3, audio +0.7, vision +0.2), and the gap is nearly static (15.3 → 14.3). The weak modalities are intrinsically limited for the task (audio ~45% vs text ~60%), and PGGB correctly self-attenuates rather than forcing uninformative signal.

**CMU-MOSI**: text 70.99 → 71.72, audio 48.86 → 49.24, vision 54.49 → 53.47. No movement exceeds 1 pp, within seed noise, and the gap is static (~22 pp).

**CH-SIMS** (the corrected column): text 64.95 → 66.43, audio 51.29 → 53.17, vision 59.78 → 60.09. PGGB improves the weaker audio (+1.9 pp) and vision (+0.3 pp), but text also rises (+1.5 pp) and OGM-GE shows the same all-modality pattern (+2.7/+2.9/+1.6), with the gap barely moving (13.65 → 13.26). In this 1.4K-sample regime the interventions act as regularizers rather than targeted rebalancers.

The contrast with CREMA-D (App. B.8: weak probe +9.41 pp, strong probe -0.81 pp, gap reduced $5.4\times$) is the direct answer to your question: targeted weak-modality recovery occurs where the weak modality is undertrained (trainable encoders, SGD, severe imbalance), whereas on text-dominant frozen-feature pipelines PGGB self-attenuates, the behavior Prop. 2 is designed to give. Adam additionally absorbs roughly two-thirds of the applied boost there (update-norm measurement in the global response). We will add the full probe tables to the appendix.

**[Q4] On text-dominant sentiment tasks, would increasing alpha help the model learn more from weaker audio/visual modalities, or does it degrade performance because those modalities are intrinsically less informative? An alpha-sensitivity study on CMU-MOSI/MOSEI would be useful.**

**Response.** Neither helps nor degrades, in every setting tested ($\alpha$-sweep, boost-only, $\alpha \in \{0.25, 0.5, 0.75, 1.0, 1.5\}$, seed 42):

| Dataset | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | Range |
|---|---|---|---|---|---|---|
| CMU-MOSI | 73.32 | 72.74 | 72.89 | 73.03 | 72.59 | 0.73 pp |
| CMU-MOSEI (genuine) | 61.90 | 61.96 | 61.96 | 61.96 | 62.01 | 0.11 pp |
| CH-SIMS | 70.24 | 70.24 | 70.68 | 71.33 | 70.68 | 1.09 pp |

CMU-MOSI spans 0.73 pp with no monotone trend, within the ~0.3-0.8 pp seed-noise floor. Genuine CMU-MOSEI is essentially invariant. CH-SIMS is mildly positive to flat (+1.1 pp bump at $\alpha=1.0$, at the edge of noise). Accuracy does not degrade at any $\alpha$ up to 1.5 on any benchmark, so we find no evidence that stronger boosting amplifies noisy, less informative modalities into accuracy loss, consistent with the bounded scaling $s \le s_{\max}$ and Adam absorbing roughly two-thirds of what remains. Conversely, raising $\alpha$ cannot unlock additional audio/visual learning here, because (per Q3) those representations are intrinsically limited rather than suppressed. We will add the sweep (extended to multiple seeds) to the appendix.

**[Limitations] The limitations section should more directly address high-imbalance safety, sensitivity of the imbalance categorization, and the possibility that some weak modalities may be intrinsically less task-informative rather than merely undertrained.**

**Response.** We fully agree and will expand the Limitations paragraph to cover all three items. The enhanced limitation text is presented as follows.

*Limitations.* Per-setting tuning is required, probes add small compute overhead, and composition with throttling can regress when the dominance gap is small (worst case AVE-from-scratch -4.56 pp, Section 4.3.1), a regime our theory does not cover. The three propositions are safety properties: they bound the intervention and preserve the standard SGD descent guarantee at any imbalance level, but they do not guarantee gap closure, which is established empirically. The imbalance categorization is insensitive to the threshold within $[0.150, 0.218)$, with Twitter15 ($0.150 \pm 0.096$) as the boundary case. Finally, PGGB does not distinguish modalities that are undertrained from modalities that are intrinsically less task-informative. Boosting recovers the former (CREMA-D) and self-attenuates on the latter (the sentiment benchmarks), but this diagnosis is currently post-hoc, and developing an online criterion that separates the two cases is a promising direction for future work.

We thank you again for questions that directly improved the paper, and we would welcome any follow-up during the discussion period.
