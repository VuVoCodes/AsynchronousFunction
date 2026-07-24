# NeurIPS 2026 Rebuttal Battle Plan — Submission 12365 (PGGB)

Reviews received 2026-07-24. Ratings: **tQk1 Accept (5, conf 3)** · **gN93 Borderline Accept (4, conf 4)** · **miLe Reject (2, conf 4)** · **AC wmcn: "lean toward acceptance"** conditioned on resolving empirical attribution + CREMA-D comparison.

Strategy: hold tQk1 and gN93 with complete, generous answers; concentrate firepower on miLe's four weaknesses (all four are answerable, one is factually mistaken); answer the AC's six numbered points explicitly and in order.

---

## AC's six must-answer points

### 1. CREMA-D OGM-GE discrepancy (miLe Q7 "MAJOR") — WE WIN THIS
**Facts (verified against the OGM-GE paper PDF, Peng et al. CVPR 2022):**
- OGM-GE paper reports **61.9%** on CREMA-D (Tables 1 & 2, concatenation + OGM-GE). The figure 61.59 quoted by miLe does not appear in that paper — it is *our* joint-training baseline in our Table 1.
- OGM-GE's CREMA-D protocol is **1 visual frame** (their §4.2: "For CREMA-D, we extract 1 frame from each of the clip"). Their concat baseline is 51.7%.
- Our main protocol is **3 frames @ 3 fps**, following the protocol of MILES / InfoReg follow-up work (disclosed in our §4.1 "Reproduction protocol"). Follow-up works report OGM-GE at 72–75% under richer frame sampling; our 69.14% sits between.
- **Clincher:** our own 1-frame ablation (Table 3) reproduces OGM-GE at **62.47 ± 1.42%**, statistically consistent with the published 61.9%. The discrepancy is entirely the frame protocol, and our reproduction is *verified at the original operating point*.
- Bonus framing: our stronger OGM-GE baseline (69.14 vs 61.9) makes our +2.31 pp claim *harder*, not easier.

### 2. Adam / scale-invariance (miLe W3) — CONCEDE THE MECHANISM, REFRAME THE EVIDENCE
- Honest analysis: for a (quasi-)constant multiplicative scale, Adam's second-moment normalization asymptotically cancels gradient scaling (up to ε and transient effects of the slowly-varying s̄_m, refreshed every K=20 steps with μ=0.3 EMA). PGGB's actuation is therefore muted under Adam. We should say this plainly.
- Reframe: the four Adam benchmarks (MOSI, MOSEI, Twitter15, Sarcasm) are exactly the frozen-feature / text-dominant pipelines where (a) measured imbalance is low or text-sufficiency limits headroom for *any* balancing method, and (b) results are consistent with self-attenuation. All SGD pipelines (CREMA-D, AVE, KS, BraTS) are where actuation is real and where the gains/story live.
- **RUN (cheap):** instrument per-modality effective update norms ‖Δθ_m‖ with and without boost on one Adam dataset (seed 42, few epochs) to show quantitatively how much scaling survives Adam. Optionally: one MOSI-with-SGD control run to show boost engages when the optimizer transmits it.
- Note OGM-GE's own Adam results rely on the GE noise component (miLe concedes this) — orthogonal to gradient scaling; PGGB has no noise term by design.

### 3. Isolate PGGB from OGM-GE with uncertainty (miLe W4 + AC) — NEEDS NEW SEEDS
**Current stats on Table 1 protocol (seeds 42/123/456/789/1024):**
- Welch t = 2.50, p = 0.040; Mann-Whitney p = 0.028; Cohen's d = 1.58
- Paired-by-seed t = 2.04, p = 0.11 (seed 1024 inverts: OGM 70.83 vs PGGB+OGM 68.95) — DO NOT lead with paired test
- ⚠️ INTERNAL: the §4.3 "(i) vs (ii) independent replication" runs are the *same invocation with the same seeds* (verified bit-identical per-seed values 67.88/68.15/69.35/69.49/70.83 in outputs/sweep_3f/3f_ogm_ge_* and outputs/sweep_3way_ablation/monitor_ogm_noboost_*; both configs have continuous_alpha: 0.0, probes active). This is a determinism check, NOT a second seed set. Never cite it as statistical replication in the rebuttal, and fix wording at camera-ready.
- **RUN (P0, ~5.2 GPU-h):** 5 fresh seeds (e.g. 2027/3407/5555/7777/9999) × {α=0, α=0.75} on CREMA-D 3-frame → n=10 per arm. Report mean ± std, Welch CI of the difference, and bootstrap CI. With d≈1.6, n=10 should give p < 0.01 if the effect is real.
- W4's 1-frame ablation point: answer that 1-frame *starves the weak modality of input information* — boosting cannot amplify information that isn't in the input; the +0.22 pp at 1 frame vs +2.31 pp at 3 frames is itself evidence the mechanism operates on representation quality (already in our docs: "10.5× amplification" with richer visual input).

### 4. Early probe trajectories + warm-up (tQk1) — MOSTLY HAVE
- App B.6 instrumentation (500 probe-eval checkpoints, seed 42) already covers training from step 0; extract the early-epoch window into a dedicated figure showing: utilization gap sign is correct from the first probe evaluations; s̄ starts at 1 (no actuation before first probe event at step K=20).
- Cold-start: P̄ init and s̄ init=1; the 2026-04-12 diagnostic showed EMA cold-start (init 0 vs first-measurement) had zero accuracy impact by epoch 3. State this.
- **RUN (optional, cheap):** one fresh instrumented run logging every probe event for epochs 1–5 across 3 seeds for a cleaner early-phase figure.

### 5. δ categorization + 0.15 threshold (gN93) — TEXT + SMALL ANALYSIS
- Move the App B.9 definition into §4.1 main text: δ = final-epoch EMA-smoothed probe-accuracy gap during *baseline* training, averaged over 5 seeds.
- Threshold honesty: per-dataset δ values are 0.017/0.038/0.076/0.150 | 0.178/0.222/0.268. Any threshold in (0.150, 0.178] yields the same split; Twitter15 (0.150 ± 0.096) is the boundary case with large seed variance — acknowledge, and note category-level conclusions are insensitive to thresholds in [0.08, 0.17] except for boundary Twitter15.
- Sensitivity to epoch/seed/probe: can plot δ trajectory over training from existing instrumentation (CREMA-D) to show it stabilizes; seeds already give ± std in Table 9.

### 6. Theory scoping (tQk1 + gN93 + AC) — TEXT ONLY
- Commit to rewording intro/§3.5 (abstract is LOCKED — commit to camera-ready wording instead): the three propositions are *safety properties* (no divergence, bounded variance inflation ≤ s_max², self-attenuation when balanced), NOT gap-closure or rate results. §3.5 already says this; sharpen and echo in intro.
- gN93's high-imbalance question: Prop 3's descent bound is uniform in δ (holds in high imbalance; worst-case variance inflation s_max² = 4 is the price). Prop 2 covers low imbalance (no intervention). What is NOT guaranteed: that boosting closes the gap — that is empirical (§4.4: weak probe +9.41 pp, strong −0.81 pp). Also gives a diagnostic criterion: monitor δ and s̄; harm regime is empirically characterized in §4.3.1 (operating conditions).

---

## Per-reviewer extras (not in AC list)

**tQk1 (protect the 5):**
- Wall-clock + memory table → **RUN (cheap):** baseline vs PGGB vs PGGB+OGM-GE on CREMA-D + one Adam dataset; report s/epoch + peak VRAM (torch.cuda.max_memory_allocated).

**gN93 (pull 4 → 5):**
- MOSI/MOSEI unimodal analysis → **RUN:** post-hoc per-modality linear probes on saved checkpoints (outputs/sweep_mosei, sweep_mosi are local) — does PGGB improve audio/visual representations even when fused accuracy is flat? Mirrors App B.8 protocol.
- α-sensitivity on MOSEI → **RUN:** single-seed sweep α ∈ {0.25, 0.5, 0.75, 1.0, 1.5} (MLP pipeline, fast). Frame around "intrinsically less informative vs undertrained" — connect to §4.1 relevance caveat, expand Limitations.

**miLe (aim: 2 → 4, or at least defang for the AC):**
- W1 (contribution type): brief, deferential — note R1 independently selected Concept & Feasibility; happy to defer to AC. One sentence.
- W2 (<0.5 pp on 7/8): the per-regime story — on the only high-imbalance audio-visual benchmark the gain is +2.31/+9.86 pp; on low-imbalance benchmarks non-regression IS the designed behavior (self-attenuation) while throttling baselines regress (OGM-GE −1.80 KS); PGGB family is the top method on 6 of 8 datasets in Table 1. Don't overclaim; cite the operating-conditions table as the honest scoping.
- Q1–Q6 clarifications: all easy. Q5 in particular is wrong on the math: at balance the numerator of Eq. 7 is 0 for every m, so w_m = 0/(0+ε) = 0 and s_m = 1 — the ε-guard exists precisely for this; code additionally hard-guards (note for camera-ready footnote alignment, Fix #5).

---

## Experiment queue (RTX 4090, priority order)

| # | Experiment | Serves | Cost | Status |
|---|---|---|---|---|
| E1 | 5 new seeds × {α=0, α=0.75}, CREMA-D 3f | AC#3, miLe W4 | ~5.2 h | **DONE 2026-07-24** |
| E2 | Wall-clock + peak-VRAM table (3 configs × 2 datasets, short) | tQk1, AC | ~1 h | **DONE 2026-07-24** |
| E3 | Adam update-norm instrumentation (MOSI + MOSEI + CREMA-D SGD contrast) | AC#2, miLe W3 | ~1 h | **DONE 2026-07-24** |

### P0 RESULTS (2026-07-24, full details in outputs/rebuttal_p0/report.md)

**E1 — n=10 per arm, CREMA-D 3-frame (seeds 42/123/456/789/1024 + 2027/3407/5555/7777/9999):**
- α=0 (OGM-GE-equiv): 69.25 ± 1.34 | α=0.75: 71.30 ± 1.48
- **Δ = +2.06 pp, 95% Welch CI [0.73, 3.38], Welch p=0.0044, Mann-Whitney p=0.0029, d=1.46**
- Seed-matched: paired t p=0.026, Wilcoxon p=0.020, sign 8+/1−/1= (binom p=0.020)
- All tests now agree; effect survives doubling seeds. THIS is the AC#3 answer.

**E3 — update-norm transmission (seed 42, second-half steps):**
- SGD (CREMA-D): visual scale 1.64 → grad ratio 1.52 → update ratio **1.50** (full transmission)
- Adam (MOSI, genuine): audio scale 1.48 → grad 1.38 → update **1.17**; visual 1.31 → 1.31 → **1.11** (~70% attenuated, not eliminated)
- Adam (MOSEI small-gap): scales ≈1.06-1.09, update ratios ≈1.02-1.04 (self-attenuation regime)
- Rebuttal line: concede Adam attenuation, quantify it, note Adam pipelines = low-headroom regime, headline effects all on SGD pipelines.

**E2 — overhead (CREMA-D 3f, span-based, 7 epochs):**
- baseline 17.43 s/epoch → PGGB 17.71 (+1.6%) → PGGB+OGM-GE 18.71 (+7.3%, of which ~5.7% is OGM-GE's own contribution computation)
- Peak VRAM ≈ equal across configs (12.1–12.3 GB incl. ~2.2 GB ambient)
| E4 | α-sweep MOSEI single-seed {0.25,0.5,0.75,1.0,1.5} | gN93 | ~2-3 h | P1 |
| E5 | Post-hoc unimodal probes on MOSI/MOSEI checkpoints | gN93 | ~1-2 h | P1 |
| E6 | Early-probe-trajectory figure (existing B.6 data; optional 3-seed re-instrument) | tQk1, AC#4 | 0-2 h | P1 |
| E7 | MOSI-with-SGD control run (boost engages when optimizer transmits scaling) | AC#2 | ~1 h | P2 |

Total P0 ≈ 7-8 GPU-hours (one overnight). All datasets local (data/ 106 GB intact), env `phd`.

## Response drafts (2026-07-24)

Drafts in `Reviews/responses/`: `00_global.md`, `R1_tQk1.md`, `R2_gN93.md`, `R3_miLe.md`.
- All P0 numbers baked in (n=10 stats, Adam transmission table, overhead table, early-trajectory facts).
- **[PENDING] placeholders remaining:** true-MOSEI Tier 1/2 results (global, gN93), post-hoc unimodal probes on true MOSEI + MOSI (gN93 Q3), α-sweep on true MOSEI (gN93 Q4 / E4).
- **Stress-tested 2026-07-24 by both reviewer agents; all BLOCK/HIGH/MEDIUM findings applied:** Table 2 fix (R3 ×2), overhead relabeled mean +7.4% and report.md regenerated to match, fresh-seeds-only non-significance owned explicitly (global + R3 W2), δ-threshold robustness re-anchored on (0.150, 0.222] independent of the affected column (R2 Q2), disclosure now enumerates all affected App B.1 descriptors + affirms all other benchmarks provenance-re-verified (verified: CREMA-D 7442 wavs, AVE, KS, MOSI YouTube IDs, Twitter15 English tweets, Sarcasm canonical repo, BraTS2021 h5, Food101 UPMC), Adam figure corrected to "two-thirds absorbed (1.48x→1.17x)", MOSI reframed as high-imbalance Adam case, "probes active boost off" labeling unified, tense fixed to "underway" with seed-42 interim numbers.
- Next: fill 4 remaining [PENDING] tokens (1 global, 3 R2) when sweep_mosei_true Tier 1/2 + post-hoc probes + α-sweep complete → John final read → submit to OpenReview. Do NOT submit with [PENDING] tokens present (stress-test BLOCK finding).

## Internal cautions
- Do NOT cite §4.3 (i)↔(ii) replication as statistical evidence (deterministic identity, same seeds).
- Abstract is locked; all abstract-directed wording changes must be promised for camera-ready.
- Std convention (pop vs sample) inconsistency — compute new-seed stats with ddof=1 and say so.
- Keep responses within OpenReview length limits; lead every response with the direct answer, then evidence.
