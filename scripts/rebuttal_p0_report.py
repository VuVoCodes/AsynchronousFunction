#!/usr/bin/env python
"""Aggregate P0 rebuttal experiment results into outputs/rebuttal_p0/report.md.

Covers:
  E2 timing  -- seconds/epoch (median inter-epoch delta from train.log) + peak VRAM
  E3 norms   -- per-modality boost scales, gradient norms, effective update norms
                (Adam on MOSEI vs SGD on CREMA-D)
  E1 seeds   -- n=10 per arm statistics for alpha=0 vs alpha=0.75 on CREMA-D 3-frame
"""
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/main/AsynchronousFunction")
OUT = ROOT / "outputs/rebuttal_p0"
SEEDDIR = ROOT / "outputs/rebuttal_seeds"

# Existing 5-seed results (seeds 42/123/456/789/1024, Table 1 protocol)
EXISTING = {
    "a0": [67.88, 68.15, 69.35, 69.49, 70.83],
    "a075": [71.37, 72.04, 73.66, 71.24, 68.95],
}

lines = ["# P0 rebuttal experiment report", ""]
lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
lines.append("")


# ---------------- E2: timing ----------------
def epoch_seconds(train_log: Path):
    ts = []
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| INFO \| Epoch (\d+):")
    for line in train_log.read_text().splitlines():
        m = pat.match(line)
        if m:
            ts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    if len(ts) < 3:
        return None
    # Mean over the full span: integer-second log timestamps make the median
    # too coarse to resolve sub-second per-epoch differences.
    return (ts[-1] - ts[0]).total_seconds() / (len(ts) - 1)


lines.append("## E2: wall-clock and peak VRAM")
lines.append("")
lines.append("| Run | s/epoch (mean over 7 epochs) | overhead vs baseline | peak VRAM (MiB) |")
lines.append("|---|---|---|---|")
base_sec = {}
for name in ["cremad_baseline", "cremad_boost", "cremad_boost_ogm",
             "mosei_baseline", "mosei_boost", "mosei_boost_ogm"]:
    tl = OUT / "timing" / f"timing_{name}" / "train.log"
    sec = epoch_seconds(tl) if tl.exists() else None
    vfile = OUT / "timing" / f"{name}.vram"
    vmax = None
    if vfile.exists():
        vals = [int(v) for v in vfile.read_text().split() if v.strip().isdigit()]
        vmax = max(vals) if vals else None
    ds = name.split("_")[0]
    if name.endswith("baseline") and sec:
        base_sec[ds] = sec
    ovh = ""
    if sec and ds in base_sec and not name.endswith("baseline"):
        ovh = f"{100.0 * (sec - base_sec[ds]) / base_sec[ds]:+.1f}%"
    lines.append(f"| {name} | {sec:.1f}s | {ovh} | {vmax} |" if sec
                 else f"| {name} | MISSING | | {vmax} |")
lines.append("")

# ---------------- E3: update norms ----------------
lines.append("## E3: effective update norms (Adam vs SGD)")
lines.append("")


def load_norms(fname):
    f = OUT / "norms" / fname
    if not f.exists():
        return []
    recs = []
    for line in f.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return recs


for ds, optname in [("mosi", "Adam"), ("mosei", "Adam, CH-SIMS pipeline"), ("cremad", "SGD")]:
    boosted = load_norms(f"{ds}_a075.jsonl")
    control = load_norms(f"{ds}_a0.jsonl")
    if not boosted or not control:
        lines.append(f"**{ds} ({optname}): MISSING DATA**")
        lines.append("")
        continue
    mods = list(boosted[0]["scales"].keys())
    # Skip the first 50 steps (optimizer state warmup)
    b = boosted[50:] if len(boosted) > 100 else boosted
    c = control[50:] if len(control) > 100 else control
    n = min(len(b), len(c))
    b, c = b[:n], c[:n]
    lines.append(f"### {ds} ({optname}, {n} steps after warmup)")
    lines.append("")
    lines.append("| Modality | mean scale (a=0.75) | grad-norm ratio (0.75/0) | update-norm ratio (0.75/0) |")
    lines.append("|---|---|---|---|")
    for m in mods:
        ms = float(np.mean([r["scales"][m] for r in b]))
        gr = float(np.mean([r["grad_norm_scaled"][m] for r in b]) /
                   max(np.mean([r["grad_norm_scaled"][m] for r in c]), 1e-12))
        ur = float(np.mean([r["update_norm"][m] for r in b]) /
                   max(np.mean([r["update_norm"][m] for r in c]), 1e-12))
        lines.append(f"| {m} | {ms:.3f} | {gr:.3f} | {ur:.3f} |")
    lines.append("")
    lines.append("Interpretation: if the boost is transmitted to parameters, the update-norm "
                 "ratio for the boosted (weak) modality should approach its grad-norm ratio; "
                 "under a scale-adaptive optimizer it should stay near 1.0.")
    lines.append("")

# ---------------- E1: n=10 statistics ----------------
lines.append("## E1: CREMA-D 3-frame, n=10 per arm")
lines.append("")
new = {"a0": [], "a075": []}
missing = []
for arm in ["a0", "a075"]:
    for seed in [2027, 3407, 5555, 7777, 9999]:
        tl = SEEDDIR / f"r10_{arm}_seed{seed}" / "train.log"
        acc = None
        if tl.exists():
            m = re.search(r"Training complete.*Best accuracy:\s*([\d.]+)", tl.read_text())
            if m:
                acc = float(m.group(1)) * 100.0
        if acc is None:
            missing.append(f"r10_{arm}_seed{seed}")
        else:
            new[arm].append(acc)

if missing:
    lines.append(f"MISSING RUNS: {', '.join(missing)}")
    lines.append("")

for label, arms in [("new seeds only (n=5)", new),
                    ("pooled old+new (n=10)", {k: EXISTING[k] + new[k] for k in new})]:
    a0, a075 = np.array(arms["a0"]), np.array(arms["a075"])
    if len(a0) < 2 or len(a075) < 2:
        lines.append(f"**{label}: insufficient data**")
        continue
    t, p = stats.ttest_ind(a075, a0, equal_var=False)
    u, pu = stats.mannwhitneyu(a075, a0, alternative="greater")
    d = (a075.mean() - a0.mean()) / np.sqrt((a075.std(ddof=1) ** 2 + a0.std(ddof=1) ** 2) / 2)
    # Welch CI of the difference
    se = np.sqrt(a075.var(ddof=1) / len(a075) + a0.var(ddof=1) / len(a0))
    dfw = se ** 4 / ((a075.var(ddof=1) / len(a075)) ** 2 / (len(a075) - 1)
                     + (a0.var(ddof=1) / len(a0)) ** 2 / (len(a0) - 1))
    tcrit = stats.t.ppf(0.975, dfw)
    diff = a075.mean() - a0.mean()
    lines.append(f"### {label}")
    lines.append("")
    lines.append(f"- alpha=0   : {a0.mean():.2f} +/- {a0.std(ddof=1):.2f}  (n={len(a0)}) {list(np.round(a0, 2))}")
    lines.append(f"- alpha=0.75: {a075.mean():.2f} +/- {a075.std(ddof=1):.2f}  (n={len(a075)}) {list(np.round(a075, 2))}")
    lines.append(f"- diff = {diff:.2f} pp, 95% Welch CI [{diff - tcrit * se:.2f}, {diff + tcrit * se:.2f}]")
    lines.append(f"- Welch t={t:.3f}, p={p:.5f} | Mann-Whitney p={pu:.5f} | Cohen's d={d:.2f}")
    if len(a0) == len(a075):
        tp, pp_ = stats.ttest_rel(a075, a0)
        try:
            _, pw = stats.wilcoxon(a075, a0, zero_method="wilcox", alternative="greater")
        except ValueError:
            pw = float("nan")
        dd = a075 - a0
        lines.append(f"- Seed-matched: paired t={tp:.3f} p={pp_:.5f} | Wilcoxon p={pw:.5f} | "
                     f"sign {int((dd > 0).sum())}+/{int((dd < 0).sum())}-/{int((dd == 0).sum())}=")
    lines.append("")

report = "\n".join(lines)
(OUT / "report.md").write_text(report)
print(report)
