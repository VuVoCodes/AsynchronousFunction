"""Figure 2 (v2): Per-modality probe accuracy across 9 methods on CREMA-D 3f.

Source: post-hoc linear-probe evaluation on frozen encoder features from
each method's best_model.pt (50 epochs probe training, fixed seed=2026,
cudnn.deterministic, 5 seeds per method).

Output files:
  - probe_accuracy_cremad_posthoc.pdf (the figure)
  - probe_accuracy_cremad_posthoc.csv (summary numbers)
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/main/AsynchronousFunction")
JSON_PATH = ROOT / "outputs" / "post_hoc_probe_cremad.json"
OUT_DIR = ROOT / "Manuscript" / "figures"

# Display order: throttle baselines, then SOTA comparators, then ours
METHOD_ORDER = [
    ("3f_baseline", "Baseline"),
    ("3f_ogm_ge", "OGM-GE"),
    ("3f_mmpareto", "MMPareto"),
    ("3f_agm", "AGM"),
    ("3f_gblend", "G-Blend"),
    ("3f_miles_t02", "MILES"),
    ("3f_inforeg_100ep", "InfoReg"),
    ("3f_boost_only", "Boost only"),
    ("3f_boost_ogm_a075", "Boost+OGM-GE"),
]


def load_summary():
    with open(JSON_PATH) as f:
        data = json.load(f)
    rows = []
    for key, name in METHOD_ORDER:
        seeds = data.get(key, {})
        if not seeds:
            continue
        aud = np.array([v["audio"] for v in seeds.values()]) * 100
        vis = np.array([v["visual"] for v in seeds.values()]) * 100
        rows.append({
            "key": key, "name": name,
            "aud_mean": aud.mean(), "aud_std": aud.std(),
            "vis_mean": vis.mean(), "vis_std": vis.std(),
            "gap_mean": (aud - vis).mean(),
            "gap_std": (aud - vis).std(),
        })
    return rows


rows = load_summary()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6),
                               gridspec_kw={"width_ratios": [1.55, 1.0]})

# ==================== Panel (a): per-modality grouped bars ====================
x = np.arange(len(rows))
width = 0.36
aud_means = np.array([r["aud_mean"] for r in rows])
aud_std = np.array([r["aud_std"] for r in rows])
vis_means = np.array([r["vis_mean"] for r in rows])
vis_std = np.array([r["vis_std"] for r in rows])

b1 = ax1.bar(x - width/2, aud_means, width, yerr=aud_std, capsize=2.5,
             label="Audio (strong)", color="#5a7bb8",
             edgecolor="black", linewidth=0.4)
b2 = ax1.bar(x + width/2, vis_means, width, yerr=vis_std, capsize=2.5,
             label="Visual (weak)", color="#e3a56b",
             edgecolor="black", linewidth=0.4)

for bar, val in zip(b1, aud_means):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.4, f"{val:.1f}",
             ha="center", va="bottom", fontsize=6.8)
for bar, val in zip(b2, vis_means):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.4, f"{val:.1f}",
             ha="center", va="bottom", fontsize=6.8)

# Highlight our method with a bold label
labels = [r["name"] for r in rows]
tick_labels = []
for lab in labels:
    if lab in ("Boost only", "Boost+OGM-GE"):
        tick_labels.append(r"$\bf{" + lab.replace(" ", r"\ ").replace("+", r"\!+\!") + r"}$")
    else:
        tick_labels.append(lab)
ax1.set_xticks(x)
ax1.set_xticklabels(tick_labels, fontsize=8, rotation=22, ha="right")
ax1.set_ylabel("Linear-probe accuracy (\\%)", fontsize=9)
ax1.set_ylim(10, 72)
ax1.set_title("(a) Per-modality linear-probe accuracy (CREMA-D, 5 seeds)", fontsize=9.5)
ax1.legend(loc="lower center", fontsize=8, ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.45))
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.set_axisbelow(True)
# Chance line (~1/6 = 16.67%) to contextualize AGM / MILES collapse
ax1.axhline(100/6, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7)
ax1.text(len(rows) - 0.5, 100/6 + 0.6, "chance", fontsize=6.8,
         color="#666666", ha="right", va="bottom")

# ==================== Panel (b): utilization gap ====================
gap_means = np.array([r["gap_mean"] for r in rows])
gap_std = np.array([r["gap_std"] for r in rows])

# Color bars: red for methods that WIDEN gap vs baseline, blue for those that CLOSE it
baseline_gap = next(r["gap_mean"] for r in rows if r["key"] == "3f_baseline")
colors = []
for r in rows:
    g = r["gap_mean"]
    if r["key"] == "3f_baseline":
        colors.append("#888888")
    elif g < baseline_gap * 0.4:  # strong gap closure
        colors.append("#1f77b4")
    elif g < baseline_gap:  # mild gap closure
        colors.append("#83b5dc")
    elif g < baseline_gap * 1.8:  # mild gap widening
        colors.append("#f0a080")
    else:  # catastrophic widening
        colors.append("#c94c3b")

bars = ax2.bar(x, gap_means, yerr=gap_std, capsize=2.5,
               color=colors, edgecolor="black", linewidth=0.4)

for bar, val in zip(bars, gap_means):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}",
             ha="center", va="bottom", fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels(tick_labels, fontsize=8, rotation=22, ha="right")
ax2.set_ylabel("Utilization gap $|a_\\mathrm{audio}-a_\\mathrm{visual}|$ (pp)", fontsize=9)
ax2.set_title("(b) Utilization gap (lower = more balanced)", fontsize=9.5)
ax2.axhline(baseline_gap, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.text(len(rows) - 0.5, baseline_gap + 0.6, "baseline",
         fontsize=6.8, color="#666666", ha="right", va="bottom")
ax2.set_ylim(0, 55)
ax2.grid(axis="y", linestyle=":", alpha=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
out_pdf = OUT_DIR / "probe_accuracy_cremad_posthoc.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_pdf}")

# CSV
csv_path = OUT_DIR / "probe_accuracy_cremad_posthoc.csv"
with open(csv_path, "w") as f:
    f.write("method,audio_mean,audio_std,visual_mean,visual_std,gap_mean,gap_std\n")
    for r in rows:
        f.write(f"{r['name']},{r['aud_mean']:.2f},{r['aud_std']:.2f},"
                f"{r['vis_mean']:.2f},{r['vis_std']:.2f},"
                f"{r['gap_mean']:.2f},{r['gap_std']:.2f}\n")
print(f"Saved {csv_path}")
