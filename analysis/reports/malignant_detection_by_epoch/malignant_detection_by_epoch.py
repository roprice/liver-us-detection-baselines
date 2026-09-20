"""Malignant-mass detection vs. epoch, five criteria on one chart.

Plots malignant case-level recall across five detection criteria, so the
permissiveness ladder (triage -> overlap -> centroid -> IoU 0.2 -> IoU 0.5) can
be read off against epoch in one view:

  - triage:       any retained predicted mass anywhere (no overlap required)
  - iou_0:        overlap-based (any overlap, IoU > 0)
  - centroid:     predicted centroid within 0.5x GT equivalent diameter
  - iou_0.2:      IoU >= 0.2
  - iou_0.5:      IoU >= 0.5

The values are read from the per-criterion JSON artifacts already produced by
the individual analysis scripts (recomputation would duplicate their logic).
Malignant detection is the case-level recall (fraction of malignant cases
detected), the headline signal for this dataset.

Single-seed run (seed 42, 625 images, joint-selected checkpoints); no error bars.

Run from project root:
    python analysis/reports/malignant_detection_by_epoch/malignant_detection_by_epoch.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUT_DIR = SCRIPT_DIR

# Root holding the individual per-criterion analysis outputs.
ANALYSIS_ROOT = PROJECT_ROOT / "analysis/epoch_convergence"

# Criterion -> (source JSON path, legend label, color, line style). Ordered from
# most to least permissive so the chart reads as a ladder.
CRITERIA = [
    ("triage",   "epoch_convergence_by_triage_detection/epoch_convergence_by_triage_detection.json",
     "Triage",                 "#9cc4f2", "--"),
    ("iou_0",    "epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_00.json",
     "Overlap (IoU > 0)",      "#2a78d6", "-"),
    ("centroid", "epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection.json",
     "Centroid",               "#3b6d11", "-"),
    ("iou_02",   "epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_02.json",
     "IoU >= 0.2",             "#eb6834", "-"),
    ("iou_05",   "epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_05.json",
     "IoU >= 0.5",             "#c0c0c0", "-"),
]

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 10,
    'font.weight': '400',
    'axes.labelsize': 10,
    'axes.labelcolor': '#999999',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#e0e0e0',
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': True,
    'axes.spines.right': True,
})


def load_recall(rel_path):
    """Return (epochs, malignant case_recall) from a criterion JSON."""
    path = ANALYSIS_ROOT / rel_path
    with open(path) as f:
        data = json.load(f)
    return data["epochs"], data["malignant"]["Detection"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    epochs = None
    series = []  # (label, recall_list, color, linestyle)
    for key, rel_path, label, color, linestyle in CRITERIA:
        p = ANALYSIS_ROOT / rel_path
        if not p.exists():
            print(f"WARNING: {p} not found, skipping {key}")
            continue
        ep, recall = load_recall(rel_path)
        if epochs is None:
            epochs = ep
        elif ep != epochs:
            print(f"WARNING: epoch mismatch for {key}, expected {epochs}, got {ep}")
        series.append((label, recall, color, linestyle))

    if epochs is None:
        print("No criterion JSONs found, exiting")
        return

    # --- Terminal table ---
    header = f"{'Epoch':>5} | " + " | ".join(f"{l:>18}" for l, *_ in series)
    print(header)
    print("-" * len(header))
    for i, ep in enumerate(epochs):
        print(f"{ep:>5} | " + " | ".join(f"{s[i]:>18.4f}" for _, s, *_ in series))

    # --- Markdown ---
    md_lines = [
        "# Malignant detection vs. epoch",
        "",
        "Single preliminary milestones run: seed 42, 625 images, joint-selected "
        "checkpoints. No error bars (single seed).",
        "",
        "Values are malignant case-level recall (fraction of malignant cases "
        "detected) under five detection criteria, ordered from most to least "
        "permissive.",
        "",
        "| Epoch |" + "".join(f" {l} |" for l, *_ in series),
    ]
    md_lines.append("|------:|" + "".join("--------------:|" for _ in series))
    for i, ep in enumerate(epochs):
        md_lines.append(f"| {ep} |" +
                        "".join(f" {s[i]:.4f} |" for _, s, *_ in series))
    md_lines.append("")
    md_path = OUT_DIR / "malignant_detection_by_epoch.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, recall, color, linestyle in series:
        ax.plot(epochs, recall, linestyle, color=color, label=label,
                marker='o', markersize=7, linewidth=1.5, markeredgecolor='white',
                markeredgewidth=1.5, zorder=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Malignant detection (case recall)')
    ax.set_xlim(left=0, right=epochs[-1])
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.grid(False, axis='x')

    for sp in ax.spines.values():
        sp.set_color('#c0c0c0')
    ax.tick_params(axis='both', which='both', color='#c0c0c0',
                   labelcolor='#999999')

    ax.legend(loc='lower right', frameon=True, fancybox=False, borderpad=0.8,
              handlelength=2.5)

    plt.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'malignant_detection_by_epoch.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
