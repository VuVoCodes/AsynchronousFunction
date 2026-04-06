# Paper Reviewer Agent Memory

## Paper Evolution
- Original framing: ASGML (Asynchronous Staleness Guided Multimodal Learning) — async updates, staleness buffers
- Current framing: "Probe-Guided Gradient Boosting for Balanced Multimodal Learning" — decoupled probes + gradient boosting
- The pivot happened because pure async/staleness mechanisms showed no improvement over baseline on CREMA-D
- Key result: boost+OGM-GE works, boost-only ~ baseline on high-imbalance data
- See `introduction-review.md` for detailed review of the Introduction section (v3: score 7.5/10)

## Key Experimental Facts
- CREMA-D (3-frame): Boost+OGM-GE 71.45 +/- 1.71%, baseline 61.59%, OGM-GE 69.14%
- AVE: Boost-only 87.41 +/- 0.26%, baseline 86.54%
- KS: Boost-only 79.17 +/- 0.97%, baseline 79.05 +/- 0.40% (+0.12pp, NOT significant)
- MOSEI: Boost+OGM-GE ~72.47%, baseline ~70.42%
- MOSI: 73.47% vs 73.18% baseline (1 seed only, +0.29pp)
- CGGM comparison: CGGM dramatically underperforms (50.22% on CREMA-D) but this may be architecture mismatch
- KS improvement (+0.12pp) is within noise / not statistically significant
- MOSI has only 1 seed — insufficient for claims

## Recurring Issues to Watch
- Claims of statistical significance need formal tests (not just mean +/- std)
- CGGM comparison may be unfair (designed for Transformers, tested on CNNs/MLPs)
- "Dataset-adaptive" claim is post-hoc pattern, not a designed property
- Contribution 5 (task-agnostic) claims regression generality but only MOSEI regression shown
- Split-batch protocol halves effective batch size — overhead not discussed
- Contribution 5 still mentions "segmentation" with zero segmentation experiments
- MOSEI: OGM-GE alone >= boost+OGM-GE; method doesn't demonstrably help there
- MOSI: 1 seed only — still not addressed as of v3 intro review

## Related Work Review History
- v1: Score 7.5/10 — Good narrative arc (theory -> throttle paradigm -> coupled monitoring -> decoupled probes). Key strengths: OGM* failure argument, honest competitor differentiation. Key weaknesses: missing Data Remixing/DI-MML/TCMax, PMR inconsistency (P2.3 says "all coupled" but P2.4 says PMR is partially decoupled), AGM/MLGM bidirectional claims not acknowledged, split-batch protocol not in text. See `related-work-review.md` for full review.
- v2: Score 8.5/10 — All 5 high/medium v1 issues resolved (Du et al. fixed, PMR qualified, AGM/MLGM acknowledged, Data Remixing+DI-MML added, split-batch in text). Remaining: P2.3 enumeration now 11 items (needs grouping), DI-MML "detached" vs "decoupled" distinction needed in P2.4, P2.1 "shared loss derivative" attribution may overclaim prior work support, TCMax still absent. See `related-work-review-v2.md` for full review.

## Introduction Review History
- v1: Initial review, identified overclaims, bib errors, em-dash issues
- v2: Score 6.5/10 — many fixes but KS/MOSI/MOSEI/segmentation issues remained
- v3: Score 7.5/10 — most bib/formatting issues fixed, remaining: KS significance, MOSI seeds, segmentation claim, MOSEI framing
- v4: Score 8.0/10 — restructured to 5 paragraphs + 3 contributions (CGGM template). Segmentation removed from contributions. Main result paragraph added. Key remaining issue: P5 "consistent improvements across all six datasets" is an overclaim (MOSEI/MOSI: method <= OGM-GE alone; KS not significant). CGGM comparison needs architecture qualifier.
