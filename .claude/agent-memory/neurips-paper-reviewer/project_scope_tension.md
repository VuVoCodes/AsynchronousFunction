---
name: Scope-of-generality tension
description: Headline +2.31 pp is one-dataset-one-combination, needs honest framing
type: project
---

**Why:** The paper claims "boost-guided gradient boosting works across 8 benchmarks" but the quantitative story is concentrated in a single dataset with a single composition. Meta-reviewers will punish overclaiming that is visible in the first-pass read of Table 1.

**How to apply:** The authors should either (a) rewrite claims to accurately reflect that the effect is most pronounced on high-imbalance data, specifically CREMA-D with OGM-GE composition, or (b) add enough positive evidence from other benchmarks to support a plural-dataset claim.

Quantitative summary of Table 1 (best method per dataset):
- CREMA-D: boost+OGM-GE (headline)
- AVE: boost-only (marginal, +0.87 over baseline, within 1σ)
- KS: boost-only (flat, within 0.1 pp of baseline)
- CMU-MOSI: OGM-GE or MMPareto tied (72.68, boost is 72.60)
- Twitter15: boost-only (marginal)
- Sarcasm: boost-only (marginal, within 0.04 pp of baseline)
- CMU-MOSEI: OGM-GE 72.47 best, boost+OGM-GE 72.43 (within noise)
- BraTS: boost+OGM-GE +0.72 pp (small)

Honest summary: on the high-imbalance benchmark, boost+OGM-GE is meaningfully better. Elsewhere, any of (baseline / OGM-GE / boost-only) is within 1 pp of the top, and method choice is noise.

The paper's framing — that boost self-attenuates where it's not needed — is technically correct but implies a stronger positive-everywhere claim than the data supports. The safer claim is "on high-imbalance data boost+OGM-GE wins; on low-imbalance data no method is clearly better, and boost-only is a safer default than throttling alternatives which can regress."
