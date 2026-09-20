"""False positives vs. epoch for the preliminary milestones run (seed 42, 625 images).

Reports, per milestone checkpoint, how many Normal (mass-free) test cases the
model flags as containing a mass. On this dataset every image is one patient and
holds at most one mass, so a "false positive" collapses to a single case-level
signal: a Normal case where the model predicts any retained mass.

Two views per epoch:
  - Raw false-alarm count over the Normal test cases.
  - False-alarm rate (count / number of Normal cases).

Reads the canonical predictions dataset (analysis/predictions_dataset): a false
positive is simply a normal case whose normal_false_positive column is True.

Single-seed run (seed 42, 625 images, joint-selected checkpoints); no error bars.

Run from project root:
    python analysis/epoch_convergence_by_false_positives/epoch_convergence_by_false_positives.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# Make the project root importable so the predictions-dataset loader can be
# used regardless of the current working directory.
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

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

OUT_DIR = SCRIPT_DIR

# Folder name -> epoch number.
EPOCH_DIRS = [
    ("predictions_milestones_625images_seed42_epoch50", 50),
    ("predictions_milestones_625images_seed42_epoch100", 100),
    ("predictions_milestones_625images_seed42_epoch150", 150),
    ("predictions_milestones_625images_seed42_epoch300", 300),
    ("predictions_milestones_625images_seed42_epoch500", 500),
    ("predictions_milestones_625images_seed42_epoch750", 750),
]

FP_COUNT_COLOR = "#eb6834"
FP_RATE_COLOR = "#2a78d6"


def count_false_positives(rows):
    """Return (fp_count, n_normal) from a checkpoint's rows."""
    normal = [r for r in rows if r.pathology == "normal"]
    fp_count = sum(1 for r in normal if r.normal_false_positive)
    return fp_count, len(normal)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(data.rows)} rows")

    epochs = []
    fp_counts = []
    fp_rates = []
    n_normal = None
    for folder_name, epoch in EPOCH_DIRS:
        rows = [r for r in data.rows if r.configuration_id == folder_name]
        if not rows:
            print(f"WARNING: no rows for {folder_name}, skipping epoch {epoch}")
            continue
        fp_count, n = count_false_positives(rows)
        if n_normal is None:
            n_normal = n
        elif n != n_normal:
            print(f"WARNING: normal count {n} != {n_normal} for {folder_name}")
        rate = fp_count / n if n > 0 else 0.0
        epochs.append(epoch)
        fp_counts.append(fp_count)
        fp_rates.append(rate)

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal table ---
    print("\nEpoch | FP count | FP rate")
    print("------|----------|---------")
    for i, ep in enumerate(epochs):
        print(f"{ep:>5} | {fp_counts[i]:>8} | {fp_rates[i]:.4f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_false_positives.json"
    payload = {
        "seed": 42,
        "images": 625,
        "normal_cases": n_normal,
        "dataset_snapshot_id": data.snapshot_id,
        "epochs": epochs,
        "fp_count": fp_counts,
        "fp_rate": fp_rates,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# False positives vs. epoch",
        "",
        "Single preliminary milestones run: seed 42, 625 images, joint-selected "
        "checkpoints.",
        "",
        "A false positive is a Normal (mass-free) case the model flags as "
        "containing a mass. Every image is one patient with at most one mass, so "
        "this is a single case-level signal.",
        "",
        "No error bars (single seed).",
        "",
        f"Normal test cases: {n_normal}.",
        "",
        "| Epoch | False positives | FP rate |",
        "|------:|----------------:|--------:|",
    ]
    for i, ep in enumerate(epochs):
        md_lines.append(f"| {ep} | {fp_counts[i]} | {fp_rates[i]:.4f} |")
    md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_false_positives.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot: FP count (left) and rate (right) on twin axes ---
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, fp_counts, '-o', color=FP_COUNT_COLOR, label='False positives',
            markersize=7, linewidth=1.5, markeredgecolor='white',
            markeredgewidth=1.5, zorder=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('False positives (count)', color=FP_COUNT_COLOR)
    ax.set_xlim(left=0, right=epochs[-1])
    ax.set_xticks(epochs)
    ax.tick_params(axis='y', labelcolor=FP_COUNT_COLOR)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.grid(False, axis='x')

    ax2 = ax.twinx()
    ax2.plot(epochs, fp_rates, '-o', color=FP_RATE_COLOR, label='FP rate',
             markersize=7, linewidth=1.5, markeredgecolor='white',
             markeredgewidth=1.5, zorder=3)
    ax2.set_ylabel('False-positive rate', color=FP_RATE_COLOR)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor=FP_RATE_COLOR)
    ax2.grid(False)

    for sp in list(ax.spines.values()) + list(ax2.spines.values()):
        sp.set_color('#c0c0c0')
    ax.tick_params(axis='x', color='#c0c0c0', labelcolor='#999999')

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc='best', frameon=True,
              fancybox=False, borderpad=0.8, handlelength=2.5)

    plt.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'epoch_convergence_by_false_positives.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
