# Introduction Review — v3 (2026-03-30)

## File Reviewed
`/Users/vuvo/Desktop/RMIT-AI/My PhD/Neurips2026-AsyncFunc/Manuscript/main.tex` lines 52-159

## Review Score: 7.5/10

## What Changed Since v2
- +9.86pp reduced from 3 to 2 occurrences (removed from P7)
- "Consistently outperforms" replaced with "most effective on high-imbalance benchmarks"
- 11+ bib entries corrected (full author lists, correct titles)
- "and others" removed from all bib entries
- "No existing method" qualified with "explicitly"
- "Decoupled" now precisely defined in P6 (detach(), separate optimizers, split-batch protocol)
- Citation key fixed: huang2022unimodal → huang2022modality in body text
- MMPareto bib: now correct (ICML 2024, PMLR volume 235)
- Zhang 2024 bib: now correct (Zhang, Latham, Saxe; ICML 2024)
- wei2024opm title confirmed correct ("On-the-fly Modulation for Balanced Multimodal Learning")
- Em-dashes: only in comments, none in body text

## Remaining Issues

### Major (still present)
1. **KS improvement (+0.12pp) is not statistically significant** — P7 groups AVE and KS together as "boost-only is preferred" but KS improvement is within noise. The claim is technically correct (boost-only IS preferred over boost+throttle on KS) but misleading by implication that boost-only helps on KS.
2. **MOSI has only 1 seed** — mentioned alongside multi-seed datasets without qualification in P7.
3. **Contribution 5 mentions "segmentation"** — no segmentation experiments exist. Only classification + regression validated.
4. **MOSEI framing in P7** — "generalizes to three-modality settings" is vague. Actually: OGM-GE alone (72.47%) >= boost+OGM-GE (72.43%). The method doesn't demonstrably help on MOSEI.
5. **"dataset-adaptive property" claim** — still somewhat strong. DWA, ReconBoost adapt intervention to dataset characteristics. The qualifier "explicitly" helps but may not survive reviewer scrutiny.

### Moderate
6. **wei2024opm bib title** — Previous review flagged incorrectly; title IS correct per official TPAMI paper.
7. **P6 sentence complexity** — The paragraph is a single block of dense text. The sentence about composability runs long (~60 words).
8. **Abstract is still TODO** — needs to be written.

### Minor
9. No em-dashes in body text (FIXED)
10. Comments still use --- and em-dashes (cosmetic, acceptable)

## Strengths Confirmed
- Throttle/boost framing is novel and compelling
- OGM* failure analysis (P5) is strongest paragraph — effective gap identification
- Narrative arc is strong (problem -> theory -> gap -> solution -> evidence)
- Clean prose, well-structured
- Good citation coverage (15+ methods referenced)
- Decoupling mechanism now precisely defined (detach, separate optimizers, split-batch)
- "Most effective on high-imbalance benchmarks" is accurate

## Factual Claims Status
- +9.86pp on CREMA-D: CORRECT (appears 2x: P7 and Contribution 3)
- -1.80pp OGM-GE on KS: CORRECT
- boost-only preferred on AVE/KS: CORRECT (though KS margin negligible)
- "outperforms either alone" on CREMA-D: CORRECT
- "generalizes to three-modality": MISLEADING (method doesn't add value on MOSEI)
- MOSI generalization: INSUFFICIENT EVIDENCE (1 seed)
- "no existing method explicitly exhibits this": DEBATABLE but defensible with "explicitly"

## Key Recommendations for v4
1. Add "(+0.87 pp)" after AVE mention, drop KS from the "boost-only is preferred" grouping or add qualifier
2. Either run multi-seed MOSI or remove from P7, keep only MOSEI
3. Remove "segmentation" from Contribution 5 — only mention validated tasks
4. Reframe MOSEI: "extends to three-modality settings" is fine if Section 4 is honest about OGM-GE carrying the improvement
5. Consider reducing +9.86pp to 1 occurrence (Contribution 3 only) and use qualitative language in P7
