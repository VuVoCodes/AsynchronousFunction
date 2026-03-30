# Paper Reviewer Agent Memory

## Paper Evolution
- Original framing: ASGML (Asynchronous Staleness Guided Multimodal Learning) — async updates, staleness buffers
- Current framing: "Probe-Guided Gradient Boosting for Balanced Multimodal Learning" — decoupled probes + gradient boosting
- The pivot happened because pure async/staleness mechanisms showed no improvement over baseline on CREMA-D
- Key result: boost+OGM-GE works, boost-only ~ baseline on high-imbalance data
- See `introduction-review.md` for detailed review of the Introduction section

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
