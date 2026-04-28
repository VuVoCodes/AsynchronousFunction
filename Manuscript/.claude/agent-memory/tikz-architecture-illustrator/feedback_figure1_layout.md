---
name: Figure 1 layout rules (hard-won from compile-then-render iterations)
description: TikZ layout rules for ASGML Figure 1 architecture overview, derived from 7 revision rounds catching visual collisions invisible at compile time
type: feedback
---

The ASGML Figure 1 must depict a 2-modality late-fusion forward path on top, decoupled probes on bottom, and a boost actuation feedback loop. The user has caught multiple visual collision bugs that compile cleanly but render badly. Always render to PNG and visually inspect before delivering.

**Why:** `compile-exit=0` is misleading — the figure compiles even when nodes overlap, arrows cross encoders, or labels collide with other elements. The user has explicitly demanded post-render visual inspection.

**How to apply:**

1. **Always render via `pdftoppm -png -r 250` (or higher) and use Read tool on the PNG** before declaring a figure done. Crop the figure region with `-x -y -W -H` for high-detail inspection.

2. **Vertical row separation: ≥ 1.7cm** between audio and visual encoder rows. 0.85cm gives no room for boost tags. Below 1.5cm risks collision with probe row anchor or top-of-figure rail.

3. **Probe row anchor: ≥ 2.4cm below `fv.south`** via an explicit `\coordinate (probeRow)`. Anchoring to `below=of zv` puts probes inside the forward path. The probe row needs its own Y coordinate.

4. **Stagger vs. align decision:** When audio/visual rows are vertically aligned (xa/xv same X), za.south's detach line passes to the RIGHT of fv (because za is right of fa, and za extends past fv's right edge). So a vertical drop from za.south does NOT collide with fv if za is far enough right. The audio detach can then dogleg to a left-shifted pa via a horizontal channel BELOW fv (`[yshift=-0.55cm]fv.south -| pa`).

5. **Probe horizontal layout: `pa under xv column, pv under zv column`** gives clean splay with `right=0.25cm of pa` for Pa (which fits between pa and pv when pv is under zv = far right of forward path).

6. **Boost tags (`∇θ_m L × s̄_m`):** place in a DEDICATED LEFT COLUMN (`xshift=-2.0cm of xa.west`), NOT under encoder bodies. Putting them under fv or beside xv causes overlap with input nodes or detach paths. Connect boost tag to encoder via curved arrow entering from above/below (semantic: parameter update, not forward flow).

7. **Stop-grad badges:** use inline `node[sgbadge, pos=...]` ON the detach arrow itself, not as a separate floating node. Floating nodes get clipped or hidden behind other elements.

8. **Boost actuation rail:** route UP from boost.north to a `railTopY` ABOVE the figure (`yshift=0.7cm of fa.north`), then LEFT to boost-tag column, DOWN to bfa, continuing DOWN to bfv. Place "boost actuation" italic label ABOVE the rail at the midpoint — far from the loss node.

9. **Pa→boost arc:** use `to[out=60,in=180]` to arc OVER pv/Pv. Trying `-- ++(0.1,0) |- (boost.170)` causes the line to cross through pv visually.

10. **Forbidden patterns reconfirmed:**
    - No probe-to-encoder gradient arrow (probes must be visually decoupled)
    - No OGM-GE composition inset at figure level (per neurips-reviewer feedback — promotes composition to figure-level visibility, conflicting with standalone-mechanism framing). Move that information to caption or §3.4 body text.
    - No `;` semicolons in caption (project-wide style rule).

11. **EMA K-step granularity** must be annotated either inline on the EMA arrow (`\xleftarrow{\text{EMA, every } K \text{ steps}}`) or in caption — neurips-reviewer flagged its absence.

12. **`\definecolor` calls** must come BEFORE `\begin{tikzpicture}` (paper-reviewer feedback: forward references are fragile).

13. **Drop-in target:** lines 235-end-of-figure-block in `Manuscript/main.tex`. Must fit `\resizebox{0.95\textwidth}{!}` for NeurIPS single-column. Keep body at 9 pages.
