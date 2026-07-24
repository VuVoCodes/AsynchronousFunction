# Response to Reviewer tQk1 (Accept, 5)

We thank the reviewer for recognizing the repurposing of linear probes as real-time, stop-gradient meta-controllers as a novel algorithmic concept. All three requested items are addressed with new measurements.

## W1: Wall-clock time and memory consumption

Measured on one RTX 4090, CREMA-D 3-frame, mean seconds/epoch over 7 epochs, identical data pipeline:

| Configuration | s/epoch | overhead | peak GPU memory |
|---|---|---|---|
| Baseline (joint training) | 17.43 | reference | ~12.2 GB |
| PGGB (probes + boost) | 17.71 | **+1.6%** | ~12.2 GB |
| PGGB+OGM-GE | 18.71 | +7.4% | ~12.2 GB |

The probe-attributable overhead is +1.6%, consistent with the ~1% estimate in Section 4.1 (the remaining ~5.8 pp in the composed row is OGM-GE's own per-modality contribution computation, incurred by OGM-GE with or without PGGB). Peak memory is unchanged across the three configurations (differences below 0.2 GB are within allocator noise): each probe is a single linear layer (~3K parameters, or ~0.03% of one ResNet-18 encoder), and the EMA state is a handful of scalars (one smoothed accuracy and one smoothed scale per modality). We will add this table to the appendix.

## W2: Early-training probe trajectories, misdirection, and warm-up

We instrumented all 500 probe evaluations (every K=20 iterations) of a full CREMA-D training run and examined the early window:

- At the **first** probe event (iteration 19), both probes are at chance level (6-class chance = 16.7%) and the EMA-smoothed ordering is inverted by 0.7 pp, which is noise around zero.
- From the **second** probe event (iteration 39) onward, the utilization gap correctly identifies audio as dominant (gap +1.2 pp, growing to +11.8 pp within three epochs) and **never inverts again** during the imbalanced phase.
- The only later inversions (iterations 5434, 5479 of 12,500) occur in late training after the gap has closed, which is the intended terminal state and where self-attenuation holds scales near 1 regardless.

Misdirection exposure is therefore bounded by construction: scales initialize at 1, are refreshed only at probe events, and the EMA (mu=0.3) limits the worst case to a single K-step window with scale at most 1 + mu*alpha (approximately 1.23 in our configuration). No explicit warm-up period is used or needed: the EMA ramp is an implicit warm-up. A diagnostic comparing EMA cold-start initializations (zero versus first measurement) produced identical accuracy by epoch 3. We will add the early-window trajectory figure to Appendix B.6.

## W3: Scope of the theoretical propositions

We agree with the reviewer's reading. The three propositions are **safety properties**: they establish that the intervention is bounded (Prop. 1), that it vanishes when modalities are balanced so it cannot destabilize already-balanced training (Prop. 2), and that training under the intervention retains the standard nonconvex SGD descent guarantee with a quantified worst-case variance inflation of s_max^2 (Prop. 3). They do not establish that PGGB closes the modality gap or improves convergence speed. Gap closure is supported empirically (Section 4.4: the weak-modality probe rises +9.41 pp while the strong falls only 0.81 pp, and the post-hoc utilization gap shrinks 5.4x versus baseline).

We will reword the abstract and introduction in the camera-ready to characterize the results precisely as safety guarantees, for example: "we establish three safety properties (bounded scaling, self-attenuation, and a standard-SGD descent bound with quantified variance inflation); a rate-level analysis linking gap closure to probe dynamics remains open." We believe honest scoping here strengthens rather than weakens the contribution, and we thank the reviewer for pushing on it.
