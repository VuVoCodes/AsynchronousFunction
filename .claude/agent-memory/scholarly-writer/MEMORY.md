# Scholarly Writer Memory

## Paper Identity
- **Title**: "Boost the Weak, Don't Brake the Strong: Probe-Guided Gradient Balancing for Multimodal Learning"
- **Method name in text**: "probe-guided gradient boosting" (lowercase in running text)
- **Abbreviation**: PGGB (informal in docs, full name in paper)
- **Venue**: NeurIPS 2026
- **Style file**: neurips_2025 (used for 2026 submission cycle)

## Style Conventions
- **No em-dashes**: User explicitly forbids `---`. Use commas, periods, or parentheses.
- **No semicolons in prose** (user rule, supersedes older note): split into sentences or use commas.
- **ICLR 2027 port** lives in `Manuscript_ICLR/main.tex` (separate from `Manuscript/main.tex`)
- [Disclosure statements + confirm markers](feedback_disclosure_statements.md): no invented facts, red AUTHORS CONFIRM markers, no dataset names in repro statement
- **Citations**: `\citep{}` for parenthetical, `\citet{}` for narrative
- **pp**: "percentage point" abbreviated as "pp" after first use
- **Tables**: `±` as `$\pm$`, bold best results per column, `\resizebox` from graphicx (NOT adjustbox)
- **Method compound name**: "boost+OGM-GE" (no spaces around +)

## Custom Macros
- `\loss` = `\mathcal{L}`, `\encoder` = `f`, `\probe` = `h`
- `\modality{i}` = `\mathcal{M}_{i}`, `\probeacc{i}` = `a^{\text{probe}}_{i}`, `\boostscale{i}` = `\alpha_{i}`

## Key Terminology
- "throttling" = suppressing dominant modality's gradient (existing methods)
- "boosting" = amplifying weaker modality's gradient (this paper)
- "coupled monitoring" = imbalance detection tied to joint training objective (the problem)
- "decoupled monitoring" = probe-based detection independent of training objective (the solution)
- "utilization gap" = probe accuracy difference between modalities

## Method Notation
- $M$ modalities, encoder $\encoder_m$ with params $\theta_m$
- Features $\mathbf{z}_m \in \mathbb{R}^d$, fusion classifier $g$ with params $\phi$
- Probe $\probe_m$ with params $\psi_m$, probe accuracy $P_m$, EMA $\bar{P}_m$
- Utilization gap $\delta$, weakness score $w_m$, boost scale $s_m$, EMA $\bar{s}_m$
- EMA coefficients: $\beta=0.1$ (probe), $\mu=0.3$ (scale)
- Boost strength $\alpha=0.5$ (boost-only) or $\alpha=0.75$ (with OGM-GE), cap $s_{\max}=2.0$, probe interval $K=20$
- OGM-GE throttle factor $\omega_m^{\text{ogm}}$, unimodal weight $\gamma=1.0$

## Equation Labels
- eq:encoder, eq:joint_loss, eq:gradient_chain, eq:probe_loss, eq:ema_probe
- eq:utilization_gap, eq:weakness, eq:boost_scale, eq:ema_scale
- eq:gradient_mod, eq:combined, eq:total_loss

## Section Labels
- sec:intro, sec:related, sec:method, sec:problem, sec:probes, sec:boost
- sec:compose, sec:algorithm, sec:experiments, sec:datasets, sec:baselines
- sec:main_results, sec:opm_comparison, sec:ablations, sec:analysis, sec:conclusion

## Table/Figure/Algorithm Labels
- fig:architecture (Figure 1 -- TikZ pipeline diagram)
- alg:pgb (Algorithm 1)
- tab:main_results (Table 1 -- 6 datasets), tab:opm (Table 2), tab:ablation (Table 3)

## Structural Decisions
- Introduction: CGGM template, 5 paragraphs + 3 contributions (~1.5 pages)
- Method (Section 3): 5 subsections, 12 equations, Algorithm 1, Figure 1
- Experiments (Section 4): 5 subsections, 3 tables

## Citation Keys (verified in references.bib)
- Core: baltrusaitis2019multimodal, huang2021multimodal, wang2020gblending, peng2022ogmge
- Theory: huang2022modality, du2023suppression, zhang2024unimodal
- Baselines: guo2024cggm, wei2024mmpareto, wei2024opm, li2023agm, kontras2024mlgm, gao2025arm, huang2025inforeg, wei2025arl, fan2023pmr, hua2024reconboost, yang2025iprm, jiang2025aug, guerra2025miles
- Probing: alain2017understanding
- Datasets: cao2014cremad, tian2018ave, arandjelovic2017kinetics, zadeh2016mosi, baid2021brats, yu2020chsims, liu2022chsimsv2 (both CH-SIMS keys now in the NeurIPS cam-ready bib too; zadeh2018mosei retained but uncited)

## Experimental Results (for citing in text)
- CREMA-D: boost+OGM-GE 71.45 ± 1.71% (+9.86pp over baseline 61.59%)
- AVE: boost-only 87.41 ± 0.26% (+0.87pp over baseline 86.54%)
- KS: boost-only 79.17 ± 0.97% (+0.12pp, within noise)
- CH-SIMS (both copies, label-corrected from "CMU-MOSEI"): OGM-GE 72.47% strongest baseline, boost+OGM-GE 72.43% (-0.04 pp), baseline 70.42%
- MOSI: OGM-GE 72.68 ± 0.89%, boost+OGM-GE 72.60% (+0.26pp over baseline)
- BraTS: boost 85.98 ± 1.15% Dice (+0.21pp over baseline)
- Beats CGGM on all 6 datasets

## Dataset Imbalance Classification
- delta > 0.15: HIGH = CREMA-D (0.268), CMU-MOSI (0.222), CH-SIMS (0.178). LOW = Twitter15 (0.150), Sarcasm, KS, AVE. BraTS 2021 separate
- [Sentiment column provenance](project_sentiment_column_provenance.md): both copies now say CH-SIMS with original numbers. Cite liu2022chsimsv2 for text-PREDOMINANCE only, never li2023agm (MOSEI-only)

## Key Framing Decisions
- CGGM underperformance: architectural mismatch caveat (designed for Transformers, tested on CNN/MLP)
- KS: OGM-GE degradation (-1.80pp) is the story, not PGGB improvement (+0.12pp)
- CMU-MOSI/CH-SIMS: "within seed std of strongest baseline" framing, not "helps" (MOSI -0.08 pp, CH-SIMS -0.04 pp)
- BraTS: cross-paper numbers caveat always included
- Variance reduction: "suggestive rather than definitive" hedging

## Packages Loaded
- booktabs, multirow, subcaption, graphicx, amsmath, amssymb, amsfonts
- algorithm, algorithmic, tikz (with arrows.meta, positioning, calc, fit, backgrounds, decorations.pathreplacing)
- NOT loaded: adjustbox
