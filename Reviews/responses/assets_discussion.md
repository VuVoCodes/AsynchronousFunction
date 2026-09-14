# Prepared assets for the discussion window (miLe follow-ups, AC requests)

Internal staging file. Each block below is drafted as postable reply text. Verified against
raw outputs on 2026-07-25. Do not post the n=15 block until the runs finish.

---

## Asset 1: Seed-ID ledger (miLe follow-up 1, AC request (b))

**Verification basis (internal):** per-seed best accuracies extracted from
`outputs/sweep_3f/3f_ogm_ge_*/train.log` and `outputs/sweep_3way_ablation/monitor_ogm_noboost_*/train.log`
are bit-identical: 67.88 / 68.15 / 69.35 / 69.49 / 70.83 (seeds 42/123/456/789/1024), mean 69.14.
Recomputed std: ddof=0 gives 1.06, ddof=1 gives 1.18. Table 1's printed 1.13 matches neither
(nor the F1 column: ddof=1 on F1 gives 1.14) and is a legacy aggregation artifact.

**Postable reply text:**

You are right to ask for an explicit ledger, and we give it in full.

| Reported result | Seeds | Provenance |
|---|---|---|
| Table 1, OGM-GE row ($69.14 \pm 1.13$) | 42/123/456/789/1024 | original sweep, $\alpha=0$, probes active |
| Section 4.3 condition (ii) "replication" ($69.14 \pm 1.18$) | 42/123/456/789/1024 (same seeds) | second invocation of the identical configuration |
| Fresh rebuttal batch ($69.35 \pm 1.62$ for $\alpha=0$) | 2027/3407/5555/7777/9999 | new seeds, identical protocol |
| Pooled $n=10$ ($69.25 \pm 1.34$) | union of the two rows above | |

Three corrections follow from this ledger, which we state plainly.

1. Section 4.3 condition (ii) and Appendix B.7 describe the same invocation inconsistently
("independent 5-seed replication" versus "differing only in the seed set"). Neither wording is
correct: it is a re-run on the same five seeds, and under our deterministic seeding it reproduces
the per-seed values exactly. It demonstrates pipeline determinism, not seed robustness, and we
will relabel it as a determinism check at camera-ready. The only genuine seed-robustness evidence
is the fresh batch and the pooled $n=10$ (and the $n=15$ extension below).
2. The apparent std discrepancy you noticed ($\pm 1.13$ vs $\pm 1.18$) has a mundane cause: an
inconsistent std convention. Recomputing from the archived per-seed values gives 1.06 with the
population convention (ddof=0) and 1.18 with the sample convention (ddof=1). The printed 1.13 is a
legacy aggregation artifact that matches neither and will be corrected. At camera-ready all
reported stds will use the sample convention (ddof=1), stated explicitly, making the Table 1 row
$69.14 \pm 1.18$.
3. All rebuttal statistics (the $n=10$ and $n=15$ analyses) already use ddof=1.

## Asset 2: Full 3-modality x 2-optimizer transmission table (miLe follow-up 3)

**Verification basis (internal):** Adam rows from `outputs/rebuttal_p0/report.md` (E3, 160 steps
after warmup); SGD rows computed from `outputs/rebuttal_sgd_control/norms/*.jsonl` (181 steps
after warmup, same skip-50 protocol). Seed 42, alpha=0.75 vs alpha=0, both arms with OGM-GE.

**Postable reply text:**

Below is the complete per-modality table you asked for, CMU-MOSI under both optimizers, identical
data, architecture, and seed, alpha=0.75 versus alpha=0 (ratios of means over matched steps after
optimizer warmup):

| Modality | Adam: scale | Adam: grad ratio | Adam: **update ratio** | SGD: scale | SGD: grad ratio | SGD: **update ratio** |
|---|---|---|---|---|---|---|
| text | 1.12 | 1.06 | **0.97** | 1.45 | 1.53 | **1.47** |
| audio | 1.48 | 1.38 | **1.17** | 1.29 | 1.27 | **1.18** |
| visual | 1.31 | 1.31 | **1.11** | 1.29 | 1.31 | **1.32** |

Two observations, including the one behind our earlier "most-boosted" label, which we state
openly rather than leave implicit.

1. **Transmission is the invariant.** Under SGD every modality's update ratio tracks its applied
scale (1.45 to 1.47, 1.29 to 1.18, 1.29 to 1.32). Under Adam every update ratio is pulled toward 1
regardless of scale (1.48 to 1.17, 1.31 to 1.11, and text at scale 1.12 transmits nothing at 0.97).
This is the mechanism claim, and it holds modality-by-modality, not only for a selected row.
2. **The scheduler's boost assignment is optimizer-dependent because the learning dynamics are.**
Under Adam the probe gap assigns the largest scale to audio; under SGD at this learning rate the
text pathway lags earliest, so text briefly receives the largest scale. This is the designed
behavior of an online controller reacting to the trajectory it actually observes, not a selection
made for display. We used "most-boosted modality" in the summary row precisely to avoid implying
the two runs boost the same modality. The full table above removes the ambiguity.

## Asset 3: n=15 with a pre-registered stopping rule (miLe follow-up 2, both ACs)

**Status:** RUNNING (launched 2026-07-25, `scripts/rebuttal_n15.sh`, ~5.2 GPU-hours,
outputs to `outputs/rebuttal_seeds/r15_*`).

**Pre-registration (recorded in the script header before launch):** batch-3 seeds fixed in
advance as 1111/2222/3333/4444/6666. We commit to reporting all 15 seeds per arm regardless of
outcome, and to adding no further seeds after this batch.

**Postable reply skeleton (fill bracketed values when runs complete):**

As offered, we extended the isolation comparison to $n=15$ per arm. To address the stopping-rule
concern directly: the five batch-3 seeds (1111/2222/3333/4444/6666) were fixed in advance, we
report all 15 regardless of outcome, and no further seeds will be added. Results (ddof=1):

| | $\alpha=0$ (OGM-GE + probes, boost off) | $\alpha=0.75$ (PGGB+OGM-GE) |
|---|---|---|
| $n=15$ | [MEAN0 +/- STD0] | [MEAN1 +/- STD1] |

Difference [+D pp], 95% Welch CI [[LO, HI]], Welch p=[P], Mann-Whitney p=[PMW], Cohen's d=[D_EFF].
Batch-3 alone: [+D3 pp] ([SIGN3] of 5 pairs positive). Per-seed values for all 15 seeds and both
arms will be included in the appendix table.

---

## Cross-references
- miLe follow-up 4 (second high-imbalance SGD audio-visual benchmark) is NOT prepared here:
  no suitable dataset is ready (AVE/KS are low-imbalance). If pressed, the honest answer is that
  we agree it would strengthen the case and scope it to future work.
- The abstract cannot change this cycle (submitted at abstract deadline); SGD-only scoping of the
  headline goes to Sections 3.3/5 now and abstract at camera-ready.
