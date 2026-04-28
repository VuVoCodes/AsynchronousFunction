---
name: Figure 1 component-to-code mapping
description: Which TikZ node in fig:architecture corresponds to which code element / equation
type: reference
---

| TikZ node | Symbol  | Maps to                                           |
|-----------|---------|---------------------------------------------------|
| `xa`,`xv` | `x_m`   | Modality input batches (CREMA-D audio/visual)     |
| `fa`,`fv` | `f_m(θ_m)` | Encoders in `src/models/multimodal.py`         |
| `za`,`zv` | `z_m`   | Encoder feature outputs                           |
| `concat`  | `[·,·]` | Late-fusion concatenation                         |
| `g`       | `g`     | Classification head                               |
| `loss`    | `L`     | `loss_fusion` in `src/asgml.py`                   |
| `pa`,`pv` | `h_m`   | `LinearProbe` in `src/probes.py`                  |
| `Pa`,`Pv` | `P̄_m`   | EMA-smoothed probe accuracy (`probe_acc_ema`)     |
| `boost`   | `w_m`/`s̄_m` | Relative-gap weight + EMA scale (asgml loss)  |
| `bfa`,`bfv` | `∇θ_m L × s̄_m` | Boost actuation = scaled gradient applied at optimizer step |
| dashed arrows | (no grad) | `.detach()` boundary in train loop          |
| thick green rail | actuation | Per-modality grad scaling before opt.step() |

**Forbidden:** any arrow indicating gradient flow from `pa`/`pv` back to `fa`/`fv` — probes must be visually decoupled.

**EMA cadence:** `s̄_m` updates every `K` steps (default K=20). Annotate this on the EMA arrow inside the boost block (per neurips-reviewer feedback).
