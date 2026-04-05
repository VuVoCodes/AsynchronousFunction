# Introduction Review History

## v4 (2026-04-05) — Score: 8.0/10

### File Reviewed
`/Users/vuvo/Desktop/RMIT-AI/My PhD/Neurips2026-AsyncFunc/Manuscript/main.tex` lines 52-99

### What Changed Since v3
- Restructured from 7 paragraphs + 5 contributions to 5 paragraphs + 3 contributions
- Followed CGGM (NeurIPS 2024) template
- Added P5 (main results paragraph) — was MISSING in v3
- Removed segmentation from contributions (was in C5 before)
- Compressed SOTA coverage into single P3 paragraph (throttle-only bias framing)
- Method description (P4) is now high-level, no split-batch/implementation details
- "Dataset-adaptive" framing retained in P5 and C3

### Key Improvements
- 5-paragraph structure is much tighter than v3's 7 paragraphs
- 3 contributions are non-overlapping and well-scoped (diagnosis, method, validation)
- P3 throttle/boost gap identification is the best paragraph — sharp, well-cited, technically grounded
- Main result paragraph (P5) immediately signals empirical substance
- No segmentation in contributions (fixed from v3)
- P4 is appropriately abstract, no implementation details leaked

### Remaining Issues

#### Major
1. **"Consistent improvements across all six datasets" (P5)** — OVERCLAIM. On MOSEI: OGM-GE alone >= boost+OGM-GE. On MOSI: OGM-GE > boost+OGM-GE. On KS: +0.12pp is not significant. The method does not "consistently improve" on all six.
2. **"Outperforming all compared baselines including CGGM" (P5)** — CGGM was run on CNN/MLP, not its native Transformer. Needs qualification or removal of explicit CGGM mention.

#### Moderate
3. **"Dataset-adaptive behavior" (P5, C3)** — post-hoc observation, not designed property. C3 uses better language ("probe signal naturally attenuates") but P5 presents it as a deliberate feature.
4. **BraTS listed but no result given** — included in six benchmarks list but no number in P5. BraTS improvement is only +0.21pp Dice.
5. **P3 shared loss derivative notation** — $\partial \mathcal{L} / \partial f(\mathbf{x}_i)$ is ambiguous; each encoder maps to different subspace. Argument is sound but notation imprecise.

#### Minor
6. Missing comma in P1 compound sentence
7. "audio typically dominates visual features" phrasing ambiguity in P2
8. "Over thirty methods" — strong quantitative claim, needs backing
9. Semicolon vs colon style in P3
10. "We trace" in C1 implies formal analysis — should match what Section 3 provides
11. No mention of computational overhead in intro

### Factual Claims Audit (P5)
- +9.86pp on CREMA-D: CORRECT
- "consistent improvements across all six": INCORRECT (MOSEI, MOSI: method <= OGM-GE alone; KS: not significant)
- "outperforming all compared baselines including CGGM": MISLEADING (architecture mismatch)
- "dataset-adaptive behavior": POST-HOC observation, not designed
- BraTS included in scope but no result shown: INCOMPLETE

### Strengths Confirmed
- Throttle/boost framing is novel and compelling (P3 is excellent)
- Narrative arc: promise -> problem -> gap -> solution -> evidence
- 3 contributions are clean and non-overlapping
- Citation density appropriate (10 methods in P3, foundational refs in P1-P2)
- No em-dashes in body text
- P4 method description is appropriately high-level

### Recommendations for v5
1. CRITICAL: Soften P5 "consistent improvements across all six datasets" — replace with honest framing (strongest under high imbalance, competitive elsewhere)
2. Remove "including CGGM" from P5 or add architecture qualifier
3. Make P5 "dataset-adaptive" language match C3's more precise "probe signal naturally attenuates"
4. Either add BraTS result to P5 or acknowledge the modest gain
5. Fix P3 notation: use $\partial \mathcal{L}/\partial z$ for shared fused representation gradient
6. Add one phrase about overhead in P4 (e.g., "negligible overhead")

---

## v3 (2026-03-30) — Score: 7.5/10

### What Changed Since v2
- +9.86pp reduced from 3 to 2 occurrences
- "Consistently outperforms" replaced with "most effective on high-imbalance benchmarks"
- 11+ bib entries corrected
- "No existing method" qualified with "explicitly"
- "Decoupled" precisely defined (detach(), separate optimizers, split-batch)
- Multiple citation key fixes

### Remaining Issues (at v3)
- KS improvement (+0.12pp) not significant
- MOSI only 1 seed
- Contribution 5 mentioned segmentation with zero experiments
- MOSEI: method doesn't demonstrably help
- "Dataset-adaptive" claim was post-hoc

---

## v2 (2026-03-28) — Score: 6.5/10
- Many fixes from v1 but KS/MOSI/MOSEI/segmentation issues remained

## v1 (2026-03-25) — Initial review
- Identified overclaims, bib errors, em-dash issues
