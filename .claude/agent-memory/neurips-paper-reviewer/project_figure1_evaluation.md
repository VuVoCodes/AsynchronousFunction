---
name: Figure 1 evaluation history
description: Meta-reviewer assessments of Figure 1 (architecture overview) across draft iterations, current finalized state and open polish items
type: project
---

Figure 1 (architecture overview) trajectory and current state.

**Why:** Figure 1 is the first artifact reviewers see. Tracking iterations lets future review passes catch regressions (e.g., if a late edit re-introduces collisions or removes the K-step annotation).

**How to apply:** Before approving any new Figure 1 edit, compare against this baseline.

## Current state (finalized 2026-04-27)

- Externally rendered: `Manuscript/figures/architecture.tex` (standalone document) → `architecture.pdf` → imported via `\includegraphics[width=0.95\textwidth]` at `main.tex:238`
- Visual render preview: `Manuscript/figures/architecture-1.png` (200 DPI)
- Layout: two-row (audio top, visual bottom), boost block right, probe row 2.4cm below forward path, gradient-tag column on far left, actuation rail rerouted via top
- OGM-GE composition inset REMOVED (matches prior reviewer recommendation)
- EMA K-step granularity annotation present: `s̄_m ← EMA, every K steps  1 + α w_m, s̄_m ≤ s_max`
- `\definecolor` calls in standalone preamble (no scope fragility)
- Caption mentions: stop-grad, EMA cadence, s_max ceiling, decoupled monitoring

## Meta-reviewer scores (2026-04-27, final)

- First-glance comprehension: 9/10 (prior projection 8.5)
- Mechanism coverage: all 5 elements (forward, stop-grad, probe→P̄, boost compute, actuation) present and visually distinguishable
- Style fit for NeurIPS 2026: strong (muted palette, dashed=monitor / solid=trainable visual grammar)
- Clarity-score impact: upholds 8.5/10, mild lift to 8.7

## Open polish items (not blockers)

1. Legend term "weak-modality amplification" vs §3.3 body terminology "probe-guided gradient boosting" — reconcile (use "boost" in legend, or add "amplification" as anchored synonym in §3.3)
2. EMA cadence on s̄_m vs P̄_m — caption or §3.3 should explicitly state s̄_m is recomputed only at probe-evaluation events (every K steps) and held constant between events. Verify `src/asgml.py` matches this before submission

## Resolved risks (no longer active)

- Layout collisions: resolved
- definecolor scope fragility: resolved (moved to standalone preamble)
- Left-side U-turn on actuation rail: resolved (top-routed)
- Diagonal z_a→Probe h_a path crossing z_v→Probe h_v: not an actual path crossing — TikZ uses a clean L-dogleg under the audio row (lines 83-87 of architecture.tex). Visual ambiguity is from the stop-grad badge bbox, not the arrow path
- OGM-GE composition inset: removed

## Things that worked and should be preserved in future iterations

- Right-side boost block (eliminates left-side U-turn)
- Two prominent `stop-grad` badges
- Mechanism-based legend (forward / monitor / boost actuation / amplification token)
- Muted RGB palette (audblue 70,110,180 / visora 220,135,55 / probegr 60,135,75)
- Explicit gradient-tag boxes `∇_θ_m L × s̄_m` adjacent to encoders
- External standalone TikZ source (decouples figure rebuild from main.tex compilation)

## Things to avoid in future redesigns

- Embedding multi-line formulas inside Figure 1 (current boost block has one but is justified; do not add more)
- Sub-insets that introduce notation undefined elsewhere
- Forward-referencing color definitions in tikzpicture styles

## Verdict

CONDITIONAL APPROVE for submission. Two minor polishes recommended; neither is a blocker.
