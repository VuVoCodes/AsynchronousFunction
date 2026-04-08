---
name: Model trace audit 2026-04-08
description: Full audit of trained model checkpoints, manuscript numbers, and credibility disclosures
type: project
---

All 175+ model checkpoints verified present with correct file sizes. All per-seed accuracies traced from training logs to experiment_results.md to manuscript tables -- no fabrication detected.

**Key finding:** Standard deviation inconsistency -- most experiments use population std (numpy default), but CREMA-D 3f results and OPM comparison use values between pop and sample std. See std_inconsistency.md for details.

**MOSEI text ambiguity:** Manuscript says "both substantially above the baseline (+2.01 pp)" but OGM-GE is +2.05 pp, boost+OGM is +2.01 pp. The +2.01 refers to boost+OGM specifically, not "both."

**Credibility audit disclosures:** All major items addressed in manuscript (CGGM caveat, identical conditions statement, encoder initialization per dataset, BraTS split details, KS degradation framing). Discussion section is still TODO.

**Why:** Ensures no numbers are fabricated or miscalculated before submission.
**How to apply:** Reference this if asked about data integrity or number verification.
