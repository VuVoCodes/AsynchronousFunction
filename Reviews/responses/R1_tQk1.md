# Response to Reviewer tQk1 (Accept, 5)

Thank you very much for the positive assessment and for recognizing the repurposing of linear probes as real-time, stop-gradient meta-controllers as a novel algorithmic concept. All three requested items are addressed below with new measurements.

**[W1] Could the authors provide a detailed wall-clock time and memory consumption for PGGB against baseline training?**

**Response.** Thank you for this practical question. Measured on one RTX 4090, CREMA-D 3-frame, mean seconds/epoch over 7 epochs, identical data pipeline:

| Configuration | s/epoch | overhead | peak GPU memory |
|---|---|---|---|
| Baseline (joint training) | 17.43 | reference | ~12.2 GB |
| PGGB (probes + boost) | 17.71 | **+1.6%** | ~12.2 GB |
| PGGB+OGM-GE | 18.71 | +7.3% | ~12.2 GB |

1. The probe-attributable overhead is +1.6%, slightly above the ~1% estimate in Section 4.1, which we will update to the measured figure. The remaining ~5.7 pp in the composed row is OGM-GE's own per-modality contribution computation, incurred by OGM-GE with or without PGGB.
2. Peak memory is unchanged across the three configurations (12.1-12.3 GB): each probe is a single linear layer (~3K parameters, or ~0.03% of one ResNet-18 encoder), and the EMA state is a handful of scalars (one smoothed accuracy and one smoothed scale per modality).
3. We will add this table to the appendix.

**[W2] Could the authors provide early-training probe trajectory plots showing whether the utilization gap correctly identifies the dominant modality from the start, and if any early misdirection occurs? Or is there any warm-up period?**

**Response.** Thank you for this suggestion. We instrumented all 500 probe evaluations of a full CREMA-D training run (every $K=20$ iterations within each 105-iteration epoch, so 5 events per epoch over 100 epochs) and examined the early window. Since plots cannot be attached in this format, we tabulate the EMA-smoothed probe accuracies $\bar{P}_m$ (%) at the early probe events, and will add the corresponding figure to the appendix:

| Probe event | Iteration (epoch) | $\bar{P}_{\text{audio}}$ | $\bar{P}_{\text{visual}}$ | Gap (pp) |
|---|---|---|---|---|
| 1st | 19 (ep. 1) | 2.1 | 2.8 | -0.7 |
| 2nd | 39 (ep. 1) | 6.4 | 5.3 | +1.2 |
| 3rd | 59 (ep. 1) | 9.2 | 7.5 | +1.7 |
| 4th | 79 (ep. 1) | 11.7 | 9.4 | +2.3 |
| 5th | 99 (ep. 1) | 17.1 | 10.8 | +6.3 |
| end of epoch 2 | 204 | 25.9 | 16.1 | +9.8 |
| end of epoch 3 | 309 | 29.3 | 16.5 | +12.8 |

1. At the **first** probe event (iteration 19), both probes are statistically at chance on the 32-sample evaluation half, and the smoothed ordering is inverted by 0.7 pp, which is noise around zero. (The tabulated values are small because the probe-accuracy EMA is still ramping from its zero initialization.)
2. From the **second** probe event (iteration 39) onward, the utilization gap correctly identifies audio as dominant (+1.2 pp, growing to +12.8 pp by the end of epoch 3) and **never inverts again through epoch 51**.
3. The only two later inversions (2 of the 500 probe events, both after epoch 50) occur exactly where the smoothed gap passes through zero (-0.3 and -1.3 pp), where the ordering is uninformative by construction. No inversion persisted beyond a single probe event in the instrumented run.
4. Exposure to a single misdirected window is small by construction: scales initialize at 1 and are refreshed only at probe events, so a misdirected first window carries a smoothed scale of at most $1+\mu\alpha \approx 1.23$ ($\mu=0.3$, $\alpha=0.75$ in this composed run), against the global cap $\bar{s}_m \le s_{\max}=2$ of Prop. 1.
5. No explicit warm-up period is used: first-window exposure is capped as above, and a diagnostic comparing probe-EMA cold-start initializations (zero versus first measurement) produced indistinguishable final accuracy.

**[W3] Given that the paper claims theoretical contributions in the abstract and introduction, could the authors clarify how the propositions help prove whether PGGB actually closes the modality gap or improves convergence speed? Or the abstract and introduction could be reworded to more precisely characterize the theoretical results.**

**Response.** We agree with your reading and will reword accordingly.

1. The three propositions are **safety properties**: the intervention is bounded (Prop. 1), provably small near exact balance (Prop. 2), and training under the intervention retains the standard nonconvex SGD descent guarantee with a quantified worst-case variance inflation of $s_{\max}^2$ (Prop. 3).
2. They do not establish that PGGB closes the modality gap or improves convergence speed. Gap closure is supported empirically (Section 4.4: the weak-modality probe rises +9.41 pp while the strong falls only 0.81 pp, and the post-hoc utilization gap shrinks $5.4\times$ versus baseline).
3. We will reword the abstract, the introduction, and the corresponding statements in Sections 3.3 and 4.2 in the camera-ready to characterize the results precisely as safety guarantees. The revised sentence is presented as follows.

*We establish three safety properties: bounded scaling, self-attenuation, and a standard-SGD descent bound with quantified variance inflation. A rate-level analysis linking gap closure to probe dynamics remains open.*

Thank you again for a review focused on exactly the feasibility questions this contribution needed answered.
