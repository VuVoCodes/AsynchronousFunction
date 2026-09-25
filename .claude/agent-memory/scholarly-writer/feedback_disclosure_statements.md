---
name: disclosure-statements-confirm-markers
description: How to draft AI-use / ethics / reproducibility statements (ICLR 2027 port) without inventing facts; red AUTHORS CONFIRM marker convention; no dataset names while provenance is unresolved
metadata:
  type: feedback
---

When drafting disclosure-type text (AI use, ethics, reproducibility) for the paper, assert only what has verified evidence. Put every unverified fact inside a compact red marker of the form
`\textcolor{red}{[AUTHORS CONFIRM: ...]}`, grouping related items (about 2 to 4 markers per statement).

**Why:** The user explicitly forbade invented facts in these statements (2026-09-14, Manuscript_ICLR/main.tex). Evidence of AI use was limited to coding (Claude Code + CLAUDE.md), drafting/editing (scholarly-writer agent), reviewer-agent critique (paper-reviewer, neurips-paper-reviewer), and the TikZ Figure 1 (tikz-architecture-illustrator; source at Manuscript_ICLR/figures/architecture.tex). Proofs, propositions, literature search, framing, dataset cleaning, interpretation, and the verification procedure had no evidence either way.

**How to apply:**
- AI use statement: list the evidenced uses, state what is verifiably not applicable (LLMs not part of PGGB, no synthetic data, no qualitative analysis), keep "authors reviewed AI-assisted work" generic with specifics in a marker, and end with the responsibility sentence. No author names or institutions (double-blind).
- Ethics: do not claim the work "does not enable surveillance", and do not claim dataset provenance was audited. The −4.56 pp (AVE from scratch) and −1.64 pp (Food101 frozen) regressions are PGGB+OGM-GE vs. baseline, not PGGB alone.
- Reproducibility: point to sections, do not repeat details, do not name datasets (see [[mosei-column-is-actually-chsims]] in user auto-memory), and do not promise a public license.
- ICLR port gotcha: `sec:baselines` sits inside subsection 4.1, so `\ref{sec:baselines}` prints the same number as `\ref{sec:datasets}`. Reference only one of them.
- Open tension (not yet fixed by the user): Section 4.1 says "identical hyperparameters across all datasets", but the Conclusion says "Per-setting tuning is required". Avoid repeating either claim in new text.
