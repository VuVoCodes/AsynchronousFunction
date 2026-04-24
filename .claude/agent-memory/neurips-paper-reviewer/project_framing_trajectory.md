---
name: Framing trajectory decision (2026-04-23)
description: Strategic framing debate — boost-first vs monitoring-first vs hybrid; recommended hybrid with three-way ablation
type: project
---

Authors considered pivoting novelty framing from "probe-guided gradient boosting" to "decoupled probe-based utilization monitoring as composable diagnostic/control." Three paths evaluated 2026-04-23: (i) boost-first, (ii) monitoring-first, (iii) hybrid.

Recommendation: **hybrid (iii)** — co-equal monitor+boost contribution, keep title, add three-way ablation (Baseline / OGM-GE / Monitor+OGM-GE-no-boost / Boost+OGM-GE) to isolate monitor's marginal value from boost's.

**Why:** Monitoring-only pivot trades ~+1 novelty for ~-1.5 rigor because B.7 composability evidence is n=3 seeds with one negative composition (MILES) that lacks mechanistic explanation. Boost-only leaves W1 novelty critique unanswered (boost-only ablation within 1σ of baseline on Table 3 1-frame).

**How to apply:**
- In future review passes, check whether a three-way ablation isolating monitor-without-boost has been added. Absence = framing still unresolved.
- If monitoring contribution is elevated, check that MILES negative-composition anomaly is mechanistically explained, not hand-waved.
- Novelty 6/10 is realistic ceiling for decoupled-probe-monitor framing vs. 5/10 for boost-first; hybrid targets 6–7.
- Prior art the monitor must be distinguished from: Alain & Bengio 2017 (linear probes as diagnostics), CGGM auxiliary classifiers (NeurIPS 2024), AGM Shapley attribution, G-Blend O/G ratio. The delta is `.detach()` decoupling + control-signal use, not the probe idea itself.
