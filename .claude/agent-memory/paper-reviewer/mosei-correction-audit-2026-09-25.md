---
name: mosei-correction-audit
description: Audit of the 2026-09-25 CMU-MOSEI data correction in Manuscript_NEURIPS_CAMREADY — what verified clean, what is still broken
metadata:
  type: project
---

# CMU-MOSEI column correction (camera-ready, applied 2026-09-25)

The CH-SIMS-mislabeled "CMU-MOSEI" column was replaced with genuine CMU-MOSEI
(`outputs/sweep_mosei_true`, 5 seeds). Label kept as CMU-MOSEI; CH-SIMS dropped entirely.

## Verified correct (do not re-audit)
- All 8 acc + 8 F1 values, delta 0.218+-0.005, MSLR 62.23+-0.28 trace exactly to
  `outputs/sweep_mosei_true/SUMMARY.md` and to per-seed `train.log` bests. std is ddof=1.
- Deltas: MOSI -0.08 (vs OGM-GE/MMPareto 72.68), MOSEI +0.04 (vs G-Blend 62.20), MSLR +0.01,
  0.98 pp 8-method spread — all arithmetically right.
- App B.1 descriptor matches `data/MOSEI_true/mosei_senti_data.pkl` exactly
  (16265/1869/4643; text 300 / audio 74 / vision 35; labels [-3,3]; |0.5| bucketing in
  `src/datasets/mosei.py` InfoReg branch). `configs/mosei_true.yaml` text/audio/visual_dim
  are stale but inert — loader auto-detects and logs "text=300, audio=74, vision=35".
- No stale old-column numbers survive anywhere (grepped 70.42/72.47/72.43/69.80/0.178/
  60.05/70.99/1.44/"7-class"/BERT/CH-SIMS). Diff = 10 hunks, all MOSEI. references.bib untouched.
- Main text 9 pages, references start p.10 — inside the limit. No MOSEI figures exist.

## Structural problems this column now has
- **Every MOSEI number is max-test-accuracy over 100 epochs, peaking at epoch 2-9**, then the
  model overfits ~4 pp (train acc 99.6%). Within-run top-5-epoch spread 0.24-1.66 pp >= the whole
  0.98 pp between-method spread. ANOVA over the 7 non-CGGM methods: F=1.14, p=0.364.
  PGGB+OGM-GE vs G-Blend: p=0.89. **CGGM alone peaks at epoch 43-69 (converged)**, so its -0.98 pp
  is partly a protocol artifact. The 1,869-sample val split exists and is unused. Same early-peak
  pattern in `outputs/sweep_mosi` — paper-wide for the sentiment columns, not new.
- delta is measured at epoch 100; accuracy at epoch 2-9. Grouping variable and outcome are
  measured at opposite ends of training. +-0.005 is across-seed std of a single-epoch snapshot
  (epoch-to-epoch range within a run is ~0.19-0.25).
- `MLPEncoder.forward` mean-pools the 50-step aligned sequence — App B.1 never says so, nor that
  the loader z-scores per feature dim and zeroes audio -inf/NaN. Mean-pooled GloVe = bag of words.
- 3-class at |0.5| is nonstandard; not comparable to published MOSEI Acc-2/Acc-7. Test majority
  class (neutral) = 41.3%, so 62% is above majority but the paper never anchors it.

## Unapplied rebuttal commitments (Reviews/responses/00_global_disclose.md, "Summary of camera-ready revisions")
1. Promised to **relabel the column CH-SIMS and report BOTH benchmarks**. Camera-ready reports
   only CMU-MOSEI with no provenance note. Contradicts a disclosure made to the AC.
2. Promised CREMA-D headline replaced by n=10 stats (71.30+-1.48, +2.06 pp, 95% CI [0.73,3.38],
   p=0.0044). Paper still has 71.45+-1.71 / +2.31 pp everywhere.
3. Promised measured +1.6% wall-clock overhead to replace "approximately 1%" (main.tex:434). Not done.
4. Promised delta definition moved into 4.1 + threshold-sensitivity ([0.150,0.218) all equivalent)
   + Twitter15 named as boundary case. Absent. The new delta makes this argument *stronger*.
5. Promised per-modality MOSEI probe table (text 60.2 / audio 44.9 / vision 46.5, gap 15.3) and
   multi-seed alpha sweep in the appendix. Absent. #5 is also the only own-measurement evidence
   for the "text-dominant" framing, which the paper currently supports by citation only.
