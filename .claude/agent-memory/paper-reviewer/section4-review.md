---
name: Section 4 experiments review v1
description: Detailed review of Section 4 (Experiments) covering datasets, baselines, main results, ablations, and analysis
type: project
---

# Section 4 (Experiments) Review -- v1 (2026-04-08)

**File reviewed:** `/Users/vuvo/Desktop/RMIT-AI/My PhD/Neurips2026-AsyncFunc/Manuscript/main.tex`, lines 472-631
**Score: 5.5 / 10**

## Key Issues Summary

### HIGH Priority
1. CREMA-D boost-only omitted from Table 1 (hides the method's weakness on the most important dataset)
2. MOSEI/MOSI: OGM-GE alone >= boost+OGM-GE -- paper frames these as successes
3. Missing baselines: MMPareto, AGM, G-Blend are important comparators not included
4. KS improvement (+0.12pp) claimed as "best result" -- within noise
5. Boost-only on CREMA-D can't beat OGM-GE alone -- positioning tension

### MEDIUM Priority
6. BraTS improvement (+0.21pp Dice) with higher variance is not meaningful
7. Std deviation inconsistency (pop vs sample std)
8. No statistical significance tests
9. Ablation on different setting (1-frame) than main results (3-frame)
10. "Never hurts" claim contradicted by MOSEI boost-only (69.80% < 70.42% baseline)

### LOW Priority
11. Computational overhead claim of 3-5% not validated with controlled measurement
12. MOSEI +2.01pp text ambiguity
13. Variance reduction claims lack formal tests
