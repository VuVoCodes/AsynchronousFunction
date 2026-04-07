# Section 3 Review — v1 (2026-04-07)
## Score: 7.0/10

## Strengths
- Chain-rule decomposition (Eq 3) directly reveals why coupled monitoring fails (shared ∂L/∂g)
- Clean 3-step boost: weakness score → raw scale → EMA-smoothed scale
- Principled probe isolation: .detach(), separate Adam, split-batch eval
- Self-attenuating: all s_m → 1 as δ → 0
- Composability argument (Sec 3.4): multiplicative stacking of boost (≥1) × throttle (≤1)
- Algorithm 1 well-structured for reproduction

## Issues

### HIGH
| # | Issue | Fix |
|---|-------|-----|
| W4 | No theoretical analysis — purely heuristic, no convergence argument | Add subsection or paragraph: bounded s̄_m ∈ [1,s_max] ≡ modality-specific LR in [η, s_max·η], converges under standard smoothness if s_max·η < 2/L |

### MEDIUM
| # | Issue | Fix |
|---|-------|-----|
| W1 | Figure box shows s̄_m = 1 + αw_m but omits min(·,s_max) cap and EMA | Simplify to just "$\bar{s}_m$" output or add note in caption |
| W5 | Unimodal regularization (Eq 11) in Sec 3.4 unrelated to composability | Move to own paragraph or remove from Sec 3, present in experimental setup |
| W6 | No failure modes discussed | Add paragraph: warm-up period, M>2 behavior, when method has minimal effect |
| W7 | Coupled monitoring argument conflates magnitude vs direction | Strengthen: Jacobian ∂g/∂z_m shaped by dominant modality's learned weights |

### LOW-MEDIUM
| # | Issue | Fix |
|---|-------|-----|
| W2 | δ defined formally (Eq 5) but never used algorithmically | Make inline or add gating δ_min |
| W3 | EMA β=0.1 means 90% on history — non-standard convention | Add parenthetical clarification |
| W8 | Throttle-then-boost ordering is commutative — not noted | Brief note |

### MINOR
- Unused macros: \modality, \probeacc, \boostscale
- Forward reference to nonexistent Section 4 (line 438)
- Notation: unsubscripted L in Eq 9-10 ambiguous after Eq 11 redefines total L
- Algorithm 1 missing ∂L/∂φ for fusion params
- Figure caption: "weaker encoder's" should be plural possessive for M>2

## Equations Check
- Eq 1-3: correct (encoder def, fusion loss, chain rule)
- Eq 4: probe loss with detach (correct)
- Eq 5: EMA for P̄_m, β=0.1 (correct, convention non-standard)
- Eq 6: utilization gap δ (correct but unused)
- Eq 7: weakness score w_m (correct normalization to [0,1])
- Eq 8: boost scale s_m = min(1+αw_m, s_max) (correct)
- Eq 9: EMA for s̄_m, μ=0.3 (correct)
- Eq 10: gradient modification (correct)
- Eq 11: combined boost×throttle (correct)
- Eq 12: total loss with unimodal reg (correct)

## Key Questions for Authors
1. Why not use δ as gating mechanism?
2. First K iterations: probes meaningful after warm-up?
3. Sensitivity to β=0.1 and μ=0.3?
4. Comparison to modality-specific LR (MSLR)?
5. Why g_m classifiers needed in addition to h_m probes?
6. Split-batch + batch norm interaction?
