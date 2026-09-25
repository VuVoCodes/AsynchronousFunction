"""Figure 2: Per-modality probe accuracy + utilization gap trajectory on CREMA-D 3f.

Panels:
  (a) Grouped bar chart: audio (strong) vs visual (weak) probe accuracy for 4 methods.
  (b) Utilization gap (|audio - visual|) trajectory across training epochs for 4 methods.

Data source: tensorboard logs under outputs/sweep_3f/3f_<method>_seed{42,123,456,789,1024}/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

METHODS = [
    ("Baseline", "3f_baseline"),
    ("OGM-GE", "3f_ogm_ge"),
    ("Boost only", "3f_boost_only"),
    ("Boost+OGM-GE", "3f_boost_ogm_a075"),
]
SEEDS = [42, 123, 456, 789, 1024]
ROOT = Path("/home/main/AsynchronousFunction/outputs/sweep_3f")

def load_final_probes():
    rows = []
    for name, m in METHODS:
        aud, vis = [], []
        for s in SEEDS:
            ea = EventAccumulator(str(ROOT / f"{m}_seed{s}" / "tensorboard"))
            ea.Reload()
            aud.append(ea.Scalars("test/probe_acc_audio")[-1].value * 100)
            vis.append(ea.Scalars("test/probe_acc_visual")[-1].value * 100)
        rows.append((name, np.array(aud), np.array(vis)))
    return rows

def load_gap_trajectories():
    # Compute gap = |audio - visual| per epoch; mean over seeds; return (epochs, mean).
    trajectories = []
    for name, m in METHODS:
        gaps = []
        for s in SEEDS:
            ea = EventAccumulator(str(ROOT / f"{m}_seed{s}" / "tensorboard"))
            ea.Reload()
            aud = np.array([x.value for x in ea.Scalars("test/probe_acc_audio")])
            vis = np.array([x.value for x in ea.Scalars("test/probe_acc_visual")])
            gaps.append(np.abs(aud - vis) * 100)
        gaps = np.stack(gaps, axis=0)  # (5 seeds, 100 epochs)
        epochs = np.arange(1, gaps.shape[1] + 1)
        trajectories.append((name, epochs, gaps.mean(axis=0), gaps.std(axis=0)))
    return trajectories


rows = load_final_probes()
trajs = load_gap_trajectories()

# Colors: distinctive, print-friendly
COLORS = {
    "Baseline": "#888888",
    "OGM-GE": "#e07a5f",
    "Boost only": "#3d9970",
    "Boost+OGM-GE": "#1f77b4",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.3))

# ==================== Panel (a): grouped bar chart ====================
x = np.arange(len(rows))
width = 0.36
aud_means = np.array([r[1].mean() for r in rows])
aud_std = np.array([r[1].std() for r in rows])
vis_means = np.array([r[2].mean() for r in rows])
vis_std = np.array([r[2].std() for r in rows])

b1 = ax1.bar(x - width/2, aud_means, width, yerr=aud_std, capsize=3,
             label="Audio (strong)", color="#5a7bb8", edgecolor="black", linewidth=0.4)
b2 = ax1.bar(x + width/2, vis_means, width, yerr=vis_std, capsize=3,
             label="Visual (weak)", color="#e3a56b", edgecolor="black", linewidth=0.4)

for bar, val in zip(b1, aud_means):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.6, f"{val:.1f}",
             ha="center", va="bottom", fontsize=7.5)
for bar, val in zip(b2, vis_means):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.6, f"{val:.1f}",
             ha="center", va="bottom", fontsize=7.5)

ax1.set_xticks(x)
ax1.set_xticklabels([r[0] for r in rows], fontsize=8.5)
ax1.set_ylabel("Probe accuracy (\\%)", fontsize=9)
ax1.set_ylim(40, 66)
ax1.set_title("(a) Per-modality probe accuracy (CREMA-D, 5 seeds)", fontsize=9)
ax1.legend(loc="lower center", fontsize=8, ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.32))
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.set_axisbelow(True)

# ==================== Panel (b): gap trajectory ====================
for name, epochs, mean, std in trajs:
    c = COLORS[name]
    ax2.plot(epochs, mean, label=name, color=c, linewidth=1.6)
    ax2.fill_between(epochs, mean - std, mean + std, color=c, alpha=0.12, linewidth=0)

ax2.set_xlabel("Training epoch", fontsize=9)
ax2.set_ylabel("Utilization gap $|a_{\\mathrm{audio}} - a_{\\mathrm{visual}}|$ (\\%)", fontsize=9)
ax2.set_title("(b) Utilization gap closing during training", fontsize=9)
ax2.set_xlim(1, 100)
ax2.set_ylim(bottom=0)
ax2.legend(loc="upper right", fontsize=8, frameon=False)
ax2.grid(linestyle=":", alpha=0.5)
ax2.set_axisbelow(True)

plt.rcParams["text.usetex"] = False  # keep portable; use mathtext for math
plt.tight_layout()
out_pdf = Path(__file__).parent / "probe_accuracy_cremad.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out_pdf}")

# Emit CSV of the final-epoch numbers for the LaTeX table
csv_path = Path(__file__).parent / "probe_accuracy_cremad.csv"
with open(csv_path, "w") as f:
    f.write("method,audio_mean,audio_std,visual_mean,visual_std,gap_mean,gap_std\n")
    for name, aud, vis in rows:
        gap = aud - vis
        f.write(f"{name},{aud.mean():.2f},{aud.std():.2f},{vis.mean():.2f},{vis.std():.2f},{gap.mean():.2f},{gap.std():.2f}\n")
print(f"Saved {csv_path}")
