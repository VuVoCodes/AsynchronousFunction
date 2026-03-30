# Introduction Review — v2 (2026-03-30)

## File Reviewed
`/Users/vuvo/Desktop/RMIT-AI/My PhD/Neurips2026-AsyncFunc/Manuscript/main.tex` lines 52-159

## Review Score: 6.5/10

## What Changed Since v1
- Em-dash punctuation issues fixed (only in comments now)
- "Consistently outperforms" overclaim fixed
- MMPareto and Zhang 2024 bib entries corrected
- Paper reframed from ASGML to "Probe-Guided Gradient Boosting"

## Major Issues (Still Present)
1. **KS +0.12pp not significant** — treated as evidence for dataset-adaptive claim
2. **MOSI has 1 seed** — cited alongside 5-seed datasets without qualification
3. **Contribution 5 overclaims** — mentions segmentation with zero evidence
4. **MOSEI misleadingly framed** — OGM-GE alone (72.47) > boost+OGM-GE (72.43), but intro implies method helps
5. **"No existing method exhibits dataset-adaptive property"** — too sweeping; MMPareto, DWA, ReconBoost all adapt
6. **CREMA-D +9.86pp repeated 3 times** — single-result padding

## Strengths Confirmed
- Throttle/boost framing is novel and compelling
- OGM* failure analysis is effective gap identification
- Narrative arc is strong (problem -> theory -> gap -> solution)
- Clean prose, no em-dashes in body text
- Good citation coverage (12+ methods referenced)

## Bib Issues Found
- `wei2024opm` title wrong: "On-the-fly modulation..." should be "Balanced multimodal learning via on-the-fly prediction modulation"
- Multiple entries have "and others" instead of full author lists (jiang2025aug, hua2024reconboost, zhang2024mla, gao2025arm, huang2025inforeg)

## Factual Claims Verified
- +9.86pp on CREMA-D: CORRECT
- -1.80pp OGM-GE on KS: CORRECT
- boost-only preferred on AVE/KS: CORRECT (though KS margin negligible)
- "outperforms either alone" on CREMA-D: CORRECT
- MOSEI generalization: MISLEADING (method doesn't help)
- MOSI generalization: INSUFFICIENT EVIDENCE (1 seed)

## Key Recommendations
1. Reframe KS as "does not hurt" rather than "improves"
2. Either run multi-seed MOSI or drop from introduction
3. Remove segmentation from Contribution 5
4. Be transparent about MOSEI (OGM-GE carries the improvement)
5. Reduce CREMA-D repetition to 1 mention with number
6. Qualify the "no existing method" claims
7. Define "decoupled" precisely to preempt challenges
