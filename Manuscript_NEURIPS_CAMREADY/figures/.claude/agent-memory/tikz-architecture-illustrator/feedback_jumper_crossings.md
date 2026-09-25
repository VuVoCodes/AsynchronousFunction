---
name: Jumper crossings via preaction white halo
description: Standard TikZ technique for "blue over green" wire-crossing visual in architecture.tex
type: feedback
---

When two solid lines must cross visually with one passing OVER the other (e.g., the blue audio probe descent crossing the green boost actuation rail near `(xa.west, [yshift=0.4cm]fa.north)` in `Manuscript/figures/architecture.tex`), apply `preaction={draw=white, line width=3.5pt, -}` to the line that should appear ON TOP. The white halo masks the underlying line at the crossing, producing a clean visible break in the lower line.

**Why:** User explicitly preferred this technique over alternatives (rectangle erasers, semicircle hops). 3.5pt width was confirmed visually correct — clear gap without being excessive. Cleaner than coordinate-based eraser rectangles because TikZ handles masking automatically.

**How to apply:** Add `preaction={draw=white, line width=3.5pt, -}` as the first style after the line's main style (e.g., `\draw[parraud, preaction={draw=white, line width=3.5pt, -}] ...`). For thicker underlying lines (e.g., `very thick` style like `barr`), bump halo to 4pt or 4.5pt. For thinner lines, 3pt may suffice. Always verify by rendering at 200 DPI and inspecting the crossing.
