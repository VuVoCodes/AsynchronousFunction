---
name: Section 3.2 Code Cross-Check
description: Detailed cross-check of Section 3.2 (Decoupled Probe Monitoring) claims vs actual implementation code
type: project
---

## Section 3.2 Cross-Check Summary (2026-04-12, re-evaluated after diagnostics)

### Verified Correct
1. LinearProbe = nn.Linear(feature_dim, num_classes) -- matches h_m: R^d -> R^C
2. .detach() enforced in both train_probes() and evaluate_probes()
3. Separate Adam optimizers (lr=1e-3) per probe, never touch encoder params
4. EMA formula matches Eq. 5: ema_alpha * accuracy + (1 - ema_alpha) * old_ema with beta=0.1
5. Utilization gap = max(ema) - min(ema), matches Eq. 6
6. Split-batch protocol: first half trains, second half evaluates, at every K=20 iterations
7. Scale EMA mu=0.3 hardcoded in ASGMLScheduler, matches paper's Eq. 8
8. BraTSProbeAdapter now wraps shared ProbeManager with split-batch, EMA, K=20, mu=0.3

### Resolved by Diagnostics (2026-04-12)
- EMA cold-start (init=0.0): Diagnostic A/B test showed identical accuracy at epoch 3 (38.71%) for both old (init=0.0) and new (init=first measurement). Decision: leave as-is, zero impact.
- BraTS probe divergence: BraTSProbeAdapter refactored to use shared ProbeManager. Now consistent with Section 3.2 claims.
- Beta notation collision: Paper uses beta only for probe EMA (beta=0.1). LearningDynamicsTracker.beta is internal to code, not in paper. No reader-facing collision.
- Algorithm "Split (x,y)" vs splitting features: Algorithm is internally consistent (step 6 splits batch, step 8 uses z_m^tr). No ambiguity.

### Still Open (minor)
1. PROBE TRAIN STEPS OMITTED: Paper/algorithm don't specify number of SGD steps per probe training call (code uses 10 for continuous mode). Reproducibility gap. Fix: add "10 inner steps" to hyperparameters paragraph..

**Why:** Documents paper-code alignment for NeurIPS review preparation.
**How to apply:** Reference when reviewing Section 3 claims, experimental reproducibility, or when verifying fixes.
