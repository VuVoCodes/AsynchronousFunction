# Paper Reviewer Agent Memory

## Paper Evolution
- Original framing: ASGML (Asynchronous Staleness Guided Multimodal Learning) — async updates, staleness buffers
- Current framing: "Probe-Guided Gradient Boosting for Balanced Multimodal Learning" — decoupled probes + gradient boosting
- The pivot happened because pure async/staleness mechanisms showed no improvement over baseline on CREMA-D
- Key result: boost+OGM-GE works, boost-only ~ baseline on high-imbalance data
- See `introduction-review.md` for detailed review of the Introduction section (v3: score 7.5/10)

## Key Experimental Facts
- CREMA-D (3-frame): Boost+OGM-GE 71.45 +/- 1.71%, baseline 61.59%, OGM-GE 69.14%, Boost-only 60.35% (seed=42, not in Table 1!)
- AVE: Boost-only 87.41 +/- 0.26%, baseline 86.54%
- KS: Boost-only 79.17 +/- 0.97%, baseline 79.05 +/- 0.40% (+0.12pp, NOT significant)
- MOSEI: OGM-GE 72.47 > Boost+OGM-GE 72.43, baseline 70.42%, Boost-only 69.80% (BELOW baseline)
- MOSI: OGM-GE 72.68 > Boost+OGM-GE 72.60, baseline 72.42% (all within noise)
- CGGM comparison: CGGM dramatically underperforms (50.22% on CREMA-D) but this may be architecture mismatch
- BraTS: Boost-only 85.98 +/- 1.15, baseline 85.77 +/- 0.62 (+0.21pp with higher variance)
- KS improvement (+0.12pp) is within noise / not statistically significant
- NEW baselines (seed=42 only): MMPareto 67.07%, AGM 56.85%, G-Blend 58.60%

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

## Section 3 (Method) Review History
- v1: Score 7.0/10 — Clear exposition (9/10), strong chain-rule motivation in 3.1, clean 3-step boost derivation. HIGH: W4 no theoretical analysis (purely heuristic, needs convergence argument). MEDIUM: W1 figure formula mismatch (missing cap+EMA), W5 unimodal regularization tacked on in 3.4, W6 no failure modes, W7 coupled monitoring argument could distinguish magnitude vs direction. LOW-MED: W2 delta defined but unused, W3 EMA convention non-standard. Minor: unused macros, forward ref to nonexistent Sec 4, notation ambiguity after Eq 11. See `section3-review.md` for full review.

## Introduction Review History
- v1: Initial review, identified overclaims, bib errors, em-dash issues
- v2: Score 6.5/10 — many fixes but KS/MOSI/MOSEI/segmentation issues remained
- v3: Score 7.5/10 — most bib/formatting issues fixed, remaining: KS significance, MOSI seeds, segmentation claim, MOSEI framing
- v4: Score 8.0/10 — restructured to 5 paragraphs + 3 contributions (CGGM template). Segmentation removed from contributions. Main result paragraph added. Key remaining issue: P5 "consistent improvements across all six datasets" is an overclaim (MOSEI/MOSI: method <= OGM-GE alone; KS not significant). CGGM comparison needs architecture qualifier.

## Section 4 (Experiments) Review History
- v1: Score 5.5/10 — HIGH: boost-only omitted from Table 1 CREMA-D, MOSEI/MOSI OGM-GE >= method, missing MMPareto/AGM/G-Blend baselines, KS +0.12pp not significant. MEDIUM: BraTS +0.21pp meaningless, no stat tests, ablation on different setting. See `section4-review.md`.

## Full Paper Feasibility Assessment
- [NeurIPS 2026 feasibility review](feasibility_review_2026-04-12.md) — Score 4.5-5.5/10, borderline-to-weak-reject. Key: empirics on CREMA-D only, no theory, overclaims, missing sections. Tier 1 fixes could reach 5,5,6.

## Code Cross-Checks
- [Section 3.2 cross-check](section32-crosscheck.md) — 7 verified correct, 6 discrepancies (BraTS probe divergence, EMA cold start, notation overload, missing probe train steps in paper)

## Audit History
- [Model trace audit 2026-04-08](audit_trace_2026-04-08.md) — All 175+ checkpoints verified, all numbers match, MOSEI +2.01 text ambiguity
- [Std deviation inconsistency](std_inconsistency.md) — Pop vs sample std used inconsistently across datasets
