---
name: Standard deviation inconsistency
description: Population vs sample std used inconsistently across the six datasets
type: project
---

Most experiments (AVE, KS, MOSI, MOSEI, BraTS) use population std (numpy.std default, ddof=0).
CREMA-D 3f results and OPM comparison values fall between pop and sample std, likely due to rounding from intermediate calculations.

Examples:
- CREMA-D 3f Boost+OGM: pop=1.52, sample=1.70, claimed=1.71 (matches sample)
- CREMA-D 3f OGM-GE: pop=1.06, sample=1.18, claimed=1.13 (between)
- AVE Boost-only: pop=0.26, claimed=0.26 (matches pop)
- KS Boost-only: pop=0.97, claimed=0.97 (matches pop)

**Why:** Reviewers may check arithmetic. Inconsistent std computation is a minor but avoidable credibility issue.
**How to apply:** Standardize all std to either population or sample std (recommend sample std with ddof=1 for 5 seeds).
