# Scholarly Writer Memory

## Paper Conventions
- **Title**: "Probe-Guided Gradient Boosting for Balanced Multimodal Learning"
- **Venue**: NeurIPS 2026
- **Style file**: neurips_2025 (used for 2026 submission cycle)
- **Citation commands**: `\citep{}` for parenthetical, `\citet{}` for narrative
- **Custom macros**: `\loss` = `\mathcal{L}`, `\encoder` = `f`, `\probe` = `h`, `\modality{i}` = `\mathcal{M}_{i}`, `\probeacc{i}` = `a^{\text{probe}}_{i}`, `\boostscale{i}` = `\alpha_{i}`
- **No em-dashes**: User explicitly forbids `---`. Use commas, semicolons, or colons instead.

## Key Terminology
- "throttling" = suppressing dominant modality's gradient (what existing methods do)
- "boosting" = amplifying weaker modality's gradient (what this paper proposes)
- "coupled monitoring" = imbalance detection tied to joint training objective (the problem)
- "decoupled monitoring" = probe-based detection independent of training objective (the solution)
- "utilization gap" = probe accuracy difference between modalities
- "OGM*" = Wei et al.'s failed boost variant (key evidence for the gap)

## Citation Keys (verified in references.bib)
- Core: baltrusaitis2019multimodal, huang2021multimodal, wang2020gblending, peng2022ogmge
- Theory: huang2022modality, du2023suppression, zhang2024unimodal
- Baselines: guo2024cggm, wei2024mmpareto, wei2024opm, li2023agm, kontras2024mlgm, gao2025arm, huang2025inforeg, wei2025arl, fan2023pmr, hua2024reconboost, yang2025iprm, jiang2025aug
- Probing: alain2017understanding
- Datasets: cao2014cremad, tian2018ave, arandjelovic2017kinetics, zadeh2018mosei, zadeh2016mosi, baid2021brats

## Experimental Results (for citing in text)
- CREMA-D: boost+OGM-GE 71.45 +/- 1.71% (+9.86pp over baseline)
- AVE: boost-only 87.41 +/- 0.26% (+0.87pp)
- KS: boost-only 79.17 +/- 0.97% (+0.12pp)
- MOSEI: 72.47 +/- 0.70% (+2.05pp)
- MOSI: 72.68 +/- 0.89% (+0.26pp)
- BraTS: 85.98 +/- 1.15% Dice (+0.21pp)
- Beats CGGM on all 6 datasets

## Structural Decisions
- Introduction follows CGGM template: 5 paragraphs + 3 contributions (~1.5 pages)
- Paper sections: intro, related work, method, experiments, discussion, conclusion
- Section labels: sec:intro, sec:related, sec:experiments
