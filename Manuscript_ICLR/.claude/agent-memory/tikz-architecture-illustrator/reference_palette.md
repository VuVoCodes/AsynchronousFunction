---
name: ASGML figure palette
description: Standard colors and TikZ styles used across paper figures
type: reference
---

```latex
\definecolor{audblue}{RGB}{70,110,180}   % audio modality (replaces blue!8/blue!40)
\definecolor{visora}{RGB}{220,135,55}    % visual modality (replaces orange!8/orange!40)
\definecolor{probegr}{RGB}{60,135,75}    % probe / monitoring / actuation (decoupled)
```

**Conventions:**
- Audio: `audblue` (modality 1, typically dominant on CREMA-D/AVE)
- Visual: `visora` (modality 2, typically weaker)
- Probe / monitoring / actuation: `probegr` — single color for the entire decoupled monitoring+boost path. Dashed for "no grad" (monitoring), solid thick for "boost actuation".
- Loss node: `red!6` fill, `red!45` border
- Fusion blocks ([;], g): `gray!8` fill

**Why one color (probegr) for both probe and actuation paths:**
The decoupled monitoring system (probes + boost computation + actuation rail) is conceptually one unit — its color identity reinforces "this is the auxiliary controller, not part of the main forward graph". Using two different greens or adding orange to actuation would fragment the visual story.

**Line-style semantics (must match across figures):**
- solid thin: forward flow
- dashed thin (probegr): monitor / no-grad path
- solid thick (probegr, `very thick`): boost actuation
