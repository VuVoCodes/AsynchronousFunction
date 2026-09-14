---
name: review-20260725-r3-rebuttal-soundness
description: Audits of Reviews/responses/R3_miLe.md (2026-07-25, four passes) — final persona pass confirms 2→3 only; gamma=1.0 aux-loss confound, Adam-inert signature in Table 1, and self-attenuation falsified by the paper's own delta table are the load-bearing findings
metadata:
  type: project
---

Audits of the rebuttal to Reviewer miLe (Reject 2, conf 4). Four passes on 2026-07-25.

## Pass 4 (final confirmation persona pass): verdict unchanged, 2 -> 3, conf 4.

Per-ask coverage: FULLY = Q1, Q4, Q5, Q6, Q7 (Q7 contingent on the unverified 61.9 fact).
PARTIALLY = W1, W2, W3, W4, Q2, Q3. NOT ADDRESSED = none outright, but four sub-asks are
missing (SGD-MOSI *accuracy*, the GE-noise implication, BraTS in the W2 regime table, the
OGM-GE-vs-PGGB share of the gap).

New verified findings in pass 4 (all re-checked against main.tex, none previously recorded):

1. **Adam-inert signature is visible in Table 1 and the rebuttal never reports it.** On all four
   Adam datasets PGGB+OGM-GE equals OGM-GE to within +-0.26 pp (MOSI -0.08, Twitter15 +0.26,
   Sarcasm -0.05, MOSEI -0.04), and standalone PGGB is *negative* on both trimodal Adam sets
   (MOSI -0.53, MOSEI -0.62 vs. baseline). This is the cleanest empirical confirmation of W3 and
   it sits in the submitted table.
2. **Self-attenuation cannot explain low-imbalance neutrality.** App. delta table: AVE 0.017,
   KS 0.038, Sarcasm 0.076, Twitter15 0.150. With eps a numerical guard, w_weak = delta/(delta+eps)
   ~ 1 at every one of these, so the boost sits at full 1+alpha all run. Line 471's "consistent with
   self-attenuation as delta -> 0" is falsified by the paper's own appendix, and Q5's (correct)
   answer is what makes this explicit. W2's "near-neutrality is the designed behavior" therefore
   has no mechanism behind it.
3. **W2 omits BraTS entirely** from its regime table (8th dataset, +0.28 pp over OGM-GE).
4. **Twitter15 delta = 0.150 +- 0.096** sits exactly on the high/low boundary with a std of 0.096,
   so the imbalance categorization the W2 answer depends on is itself unstable.
5. **All n=15 statistics re-derive exactly** (Welch t=3.22, p=0.0033, CI [0.58, 2.62], d=1.18) and
   the batch decomposition 2.31 / 1.81 / 0.68 is consistent with the reported 2.31 / 2.06 / 1.60
   aggregates. The arithmetic is sound; only the framing is contestable.
6. **The n=15 work moves the wrong dataset.** CREMA-D was the 1 of 8 the reviewer did *not* call
   minor, and the new seeds shrink it 2.31 -> 1.60. W2's actual complaint (7 datasets under 0.5 pp)
   is untouched. Verified margins: AVE +0.32, KS +0.12, MOSI -0.08, Twitter15 +0.04, Sarcasm +0.04,
   MOSEI -0.04, BraTS +0.28.

## Pass 3 (persona coverage judgment): would move 2 -> 3, not higher.

Coverage: 4 asks FULLY answered (Q1, Q2, Q5, Q7), 5 PARTIALLY (W1, W2, W3, W4, Q3/Q4/Q6),
2 sub-asks NOT ADDRESSED (W3's decisive SGD-MOSI accuracy run, W3's GE-noise implication for the
4 Adam datasets). All reported statistics re-derive correctly. The failures are attribution and
scope, not arithmetic.

**Fixed since pass 2 (do not re-flag):** the MILES/InfoReg protocol-provenance misattribution is
gone (Q7 pt. 2 now says only "3 frames at 3 fps, a single matched pipeline"), and the "7.2 pp
stronger than the original paper" claim is gone.

## New findings (pass 3), highest severity first

1. **gamma=1.0 auxiliary-loss confound in the "OGM-GE" baseline row.** main.tex line 432 sets
   `gamma=1.0` (Eq. 12 unimodal aux losses, full weight). Line 504 states the Table 1 OGM-GE row
   *is* the alpha=0 PGGB arm. So the OGM-GE baseline = OGM-GE + gamma=1.0 aux losses + probes,
   while Baseline (61.59) is plain joint training. No "Baseline + gamma=1.0" control exists
   anywhere. This confounds Q7's single-cause (frames) explanation of 61.59 -> 69.14 and the
   headline "+9.86 pp over joint training". If instead the baselines were run without aux losses,
   line 504's "operationally identical" claim is false. One of the two must be wrong.
2. **Per-batch decomposition, derived from the rebuttal's own three aggregates (2.31 / 2.06 / 1.60):**
   batch1 = +2.31, batch2 = +1.81, batch3 = +0.68. Monotone decay, and the *only* pre-registered
   batch is the third one at +0.68 pp. The rebuttal's "original seeds were favorable draws" framing
   does not survive this: it is a decaying sequence, not a noisy one.
3. **W4 survives at the main operating point.** n=15: baseline 61.59 -> alpha=0 69.40 -> alpha=0.75
   71.00. OGM-GE share 7.81/9.41 = 83%, PGGB share 1.60/9.41 = 17%. At 1-frame it was 2.17/2.39 =
   91% vs 0.22/2.39 = 9%. Moving to 3-frame improves PGGB's share from 9% to 17%; it does not
   refute "PGGB's contribution is minor". The rebuttal quotes +1.60 pp and never states the ratio.
4. **Prop. 2 statement contradicts its own proof.** Line 392 says the EMA scale "converges
   geometrically to unity"; the proof at line 632 gives convergence to within alpha*Delta/(Delta+eps),
   not to unity. Line 734 states the honest version. Line 392 is the overclaim, and the rebuttal's
   Q6 repeats it.
5. **"Best on all four low-imbalance" reduces to two datasets.** Twitter15 and Sarcasm use frozen
   BERT + frozen ResNet-18 with only MLP heads trainable (lines 792, 794) and Adam (line 432).
   App. B.8 line 833 says the probe signal is "only informative once encoder parameters can respond
   to it". So 2 of the 4 wins (both +0.04 pp) are in a regime the paper itself excludes, and the
   rebuttal's own W3 Adam analysis excludes them again. Remainder: AVE +0.32, KS +0.12, both well
   inside std. **This is the sharpest verified counter to the rebuttal's W2 answer, and it is a
   direct W2-vs-W3 cross-answer contradiction.**
6. **The frame change is not effect-neutral across methods.** 1-frame -> 3-frame: baseline +1.29,
   OGM-GE +6.67, PGGB+OGM-GE +8.76. Q7's "richer visual input raises every method" is true but
   raises them by 1.3 to 8.8 pp, so the ranking is protocol-dependent, not merely shifted.
7. **W3 measurement is between-run, so it confounds optimizer attenuation with trajectory divergence.**
   grad-norm/applied-scale = 0.927, 0.932, 1.055 across the three rows: the sign of the divergence
   flips. The 62% SGD transmission uses 1.18/1.29, numbers absent from the table. And the rebuttal
   states the controller boosts *different* modalities under the two optimizers, so the "control"
   changes the optimizer and the treated modality at once. The clean test is within-run:
   ||Adam-update(s*g)|| / ||Adam-update(g)|| from one optimizer state.
8. **Theory point the rebuttal concedes too cheaply:** Adam is exactly invariant to a *constant*
   gradient rescaling, and PGGB deliberately makes the scale piecewise-constant (mu=0.3 EMA,
   refreshed only every K=20 steps, Fig. 1 caption). The predicted asymptotic transmission under
   Adam is ~0, not 35%. So the design maximizes Adam absorption, which makes this a design flaw on
   4/8 benchmarks, not "future work".
9. **Q3's rewrite shifts the mechanism.** It blames the per-modality factor d(y-hat)/dz_m, while
   Section 3.1 line 258 blames the *shared* factor dL/dg. Different claims; the rewrite no longer
   supports the sentence it replaces.
10. **Q4's rewrite over-claims.** "P_m ... independent of how the fusion head weights modality m" is
    true of the probe's readout, false of z_m, which is trained through the fusion head. The
    stop-gradient decouples the measurement, not the measured quantity. Same over-claim as the
    abstract's "unbiased" (lines 81, 279, 222).
11. **Q2's fix is incomplete.** Eq. 2 uses y-hat with no defining equation, and Eq. 3 writes
    dL/dg where g is a *function* (ill-typed; should be dL/dy-hat). Defining g after Eq. 1 fixes
    neither.
12. **Model selection unexplained.** alpha=0.5 for standalone, alpha=0.75 for the composition
    (line 432). If chosen on test accuracy, "best on all four" is selection-contaminated. No reviewer
    has asked this yet; expect it.

## Still open from pass 2 (unchanged)

- **Eq. 7 is bang-bang for M=2 and eps is still unquantified** (line 316 calls it only "a numerical
  guard against division by zero"). For M=2 the weak modality gets the full 1+alpha at any realistic
  delta, so the boost magnitude encodes the *sign* of the gap, not its size. 5 of 8 benchmarks are
  M=2. This makes Prop. 2's self-attenuation vacuous and falsifies line 326, line 471
  ("consistent with self-attenuation as delta -> 0") and line 408. Q5's answer points the reviewer
  straight at the eps.
- **Line 429 tension with Q7.** The paper says 69.14 is "3--6 pp below the 72--75% in follow-up work";
  the rebuttal argues to the reviewer that 69.14 is high because of frames. Coherent only if
  follow-up work uses still more frames, which line 429 never says. The rebuttal promises to fix
  Section 4.1 but not line 429, which is where a reviewer checking Q7 would look.
- **Two configurations impersonating one method.** The "gains where imbalance exists" row is
  PGGB+OGM-GE (alpha=0.75); the "best on all four" row is PGGB alone (alpha=0.5).
- **Unverified (instructed not to open Papers/OGM-GE):** "The OGM-GE paper reports 61.9% ... their
  Tables 1 and 2, concatenation fusion". Reads like OGM-GE's *baseline* row, not its method row;
  the widely cited OGM-GE CREMA-D *method* number is 66.94. If so, Q7 pt. 1 compares their baseline
  against our OGM-GE. **Must be checked before posting** — it corrects a conf-4 reviewer on their
  own source.

**Coordination risk (unchanged):** Q1 re-asserts "CMU-MOSEI" as a trimodal benchmark while the
global/R2 responses disclose it is CH-SIMS. See [[project-mosei-is-chsims]] and
[[project-disclosure-path]].

**Abstract lock (unchanged):** Q6 promises "the abstract and introduction will be reworded" and W2
promises the n=15 headline "throughout the camera-ready", but the abstract is locked this cycle and
names "self-attenuation" explicitly. Needs a camera-ready qualifier.
