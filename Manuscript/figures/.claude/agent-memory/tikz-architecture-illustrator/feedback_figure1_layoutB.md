---
name: Figure 1 Layout B (probes-flank-Boost) gotchas
description: Geometry traps when arranging audio-left + visual-right + Boost-centered in architecture.tex bottom panel
type: feedback
---

Layout B for the bottom panel of `Manuscript/figures/architecture.tex`:
audio chain `[h_a][P_a]` on LEFT, visual chain `[P_v][h_v]` (mirrored) on RIGHT,
Boost block CENTERED below both. Several non-obvious geometry traps to remember:

**Why:** Round 1 + Round 2 of the convergence loop (2026-05-06) hit each of these
traps and had to iterate. A future round redoing Layout B should not re-derive the
constraints from scratch.

**How to apply:**

1. **GEOMETRY: za.x == zv.x is a hard constraint.** The top panel stacks audio over
   visual rows (xv `below=1.7cm of xa`, etc.), so all three columns share x — in
   particular `za.x = zv.x`. A LITERAL straight vertical drop from each z_m to a
   non-overlapping probe is therefore impossible. The cleanest compromise: anchor
   each probe's INNER top corner OUTSIDE the centerline (pa.NE 1.5cm LEFT of
   za.south for audio, pv.NW 1.5cm RIGHT of zv.south for visual). The sg-wire is
   then a SHORT slanted single-segment line, not an L-bend. With a 1.5cm offset
   over a 2.6cm vertical drop, the slant is ~30 deg from vertical -- visually
   "near-vertical", not "long horizontal traverse".

2. **Probe y-alignment must use probeRow, not z_m.south offset.** If you write
   `at ($(za.south)+(0,-2.4)$)` for pa, pa lands at za.south.y - 2.4 -- but
   zv.south.y is 1.7cm BELOW za.south.y, so pv lands 1.7cm BELOW pa. Both probes
   end up at different y. Fix: define `\coordinate (probeRow) at ([yshift=-1.6cm]fv.south)`
   ONCE, then anchor both probes via `($(probeRow -| za)+(-1.5,0)$)` and
   `($(probeRow -| zv)+(1.5,0)$)`. Both share probeRow.y.

3. **Chain-crossover trap.** With pa.NE pinned ON the centerline (offset 0), and
   Pa to the right of pa, Pa extends PAST za.x into the right (visual) half --
   crossing where Pv ought to be. To keep Pa in the LEFT half AND Pv in the RIGHT
   half, the offset must be at least `Pa-width + 0.25 + Pa-inner-gap = 0.7 + 0.25 + 0.55 = 1.5cm`.
   Smaller offsets cause Pa to overlap Pv.

4. **sg-wire passes through the visual encoder row.** A slanted line from za.south
   (audio row, above visual row) to pa.NE (below probeRow) MUST cross the visual
   row at some y. With 1.5cm horizontal offset and probeRow at fv.south - 1.6cm,
   the wire at y=zv.center sits at x = za.x - 0.62cm, which is in the GAP between
   fv.east (za.x - 0.9) and zv.west (za.x - 0.35) -- clean. Tighter offsets (e.g.,
   1.0cm) push the wire INTO zv. Stay at >= 1.5cm.

5. **sg badge placement.** Naively placing the badge at the slant-midpoint puts it
   AT the visual row (since the wire crosses that row near the midpoint). The
   badge then overlaps fv or zv. Place the badge near the OUTER corner of the
   probe instead: `at ($(pa.north east)+(-0.08,0.10)$)` with `anchor=south east`
   for audio (mirrored for visual). The badge then sits just above the probe box,
   below the visual row.

6. **"Boost block" header above wrapper blocks centered top entries.** If header
   is anchored above the wrapper, probe rails entering at boost.north center will
   cross it. Round 2 fix: **embed the header as the FIRST line of the boost node**
   (`\textbf{\textcolor{probegr}{Boost block}}\\[1pt]` followed by equations). The
   top edge of the boost wrapper is then fully clear for two SHORT vertical rails
   from Pa.south and Pv.south.

7. **Boost rail entries use `Pa.south |- boost.north`.** With Pa and Pv positioned
   inward at offsets ~1.5cm from za.x, their .south x-coords (~za.x +/- 0.9) land
   well within boost.north's span [za.x - 2.2, za.x + 2.2] (boost width 4.4cm).
   Use `\coordinate (boostInNW) at (Pa.south |- boost.north)` for a STRAIGHT
   vertical rail Pa.south -> boostInNW (no L-bend).

8. **Vertical compression for page-9 anchor.** Default `probeRow at fv.south - 2.4cm`
   plus `boost below=1.3cm of probes` plus inner_sep 5pt makes the figure 363pt
   tall, pushing the manuscript to 30 pages (lost page-9 anchor). Tightening to
   `probeRow at fv.south - 1.6cm`, `below=0.85cm of probes`, and `inner_sep=4pt`
   on boost yields 324pt tall -- restores 29 pages. Watch this dimension.

9. **Boost actuation rail and audio probe rail share the WEST edge.** Both naturally
   exit/enter the west of the centered Boost. To avoid them overlapping, exit
   actuation from the LOWER west (`boostSW = [yshift=-0.18cm]boostWrap.west`) and
   route it up the FAR LEFT lane. In Round 2 the lane was tightened from
   `bfa.west - 0.65cm` to `bfa.west - 0.35cm` (slimmer green column).

10. **Legend anchor must move with Boost.** Default `[yshift=-1.1cm]pa.south -| concat`
    puts legend right where Boost now sits. Use `[yshift=-0.6cm]boostWrap.south -| concat`.

11. **Visual chain mirror:** `Pv` is `left=0.25cm of pv` (NOT right) so P_v faces
    inward toward Boost. Correspondingly the h_v -> P_v arrow is `(pv.west) -- (Pv.east)`.

Layout B Round 2 (final): probes UNDER z_m columns with 1.5cm outward offset on
the inner-corner anchor, sg-wires as short slanted single segments, Boost header
embedded inside the box, both Pa.south->Boost and Pv.south->Boost as straight
vertical drops. Figure: 575 x 324 pt. Manuscript: 29 pages, page-9 anchor intact.
