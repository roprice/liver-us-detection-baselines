"""False positives vs. epoch for the preliminary milestones run (3 seeds, 625 images each).

Reports, per milestone checkpoint, how many Normal (mass-free) test cases the
model flags as containing a mass. On this dataset every image is one patient and
holds at most one mass, so a "false positive" collapses to a single case-level
signal: a Normal case where the model predicts any retained mass.

Two views per epoch, averaged across the three seeds with standard deviation:
  - Raw false-alarm count over the Normal test cases.
  - False-alarm rate (count / number of Normal cases).

Reads the canonical predictions dataset (analysis/predictions_dataset): a false
positive is simply a normal case whose normal_false_positive column is True.

Three seeds (42/43/44), 625 images each; error bars are the across-seed std.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_false_positives/epoch_convergence_by_false_positives.py
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

SEEDS = [42, 43, 44]
EPOCHS = [50, 100, 150, 300, 500, 750]

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
    fp_count_means = []
    fp_count_stds = []
    fp_rate_means = []
    fp_rate_stds = []
    fp_count_per_seed = []  # per epoch, per seed: raw fp count integers
    n_normal = None
    for epoch in EPOCHS:
        counts = []
        rates = []
        seed_counts = []
        for seed in SEEDS:
            folder = f"predictions_milestones_625images_seed{seed}_epoch{epoch}"
            rows = [r for r in data.rows if r.configuration_id == folder]
            if not rows:
                print(f"WARNING: no rows for {folder}")
                continue
            fp_count, n = count_false_positives(rows)
            if n_normal is None:
                n_normal = n
            elif n != n_normal:
                print(f"WARNING: normal count {n} != {n_normal} for {folder}")
            counts.append(fp_count)
            rates.append(fp_count / n if n > 0 else 0.0)
            seed_counts.append(fp_count)
        epochs.append(epoch)
        fp_count_means.append(float(np.nanmean(counts)) if counts else float('nan'))
        fp_count_stds.append(float(np.nanstd(counts, ddof=1)) if counts else float('nan'))
        fp_rate_means.append(float(np.nanmean(rates)) if rates else float('nan'))
        fp_rate_stds.append(float(np.nanstd(rates, ddof=1)) if rates else float('nan'))
        fp_count_per_seed.append(seed_counts)

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal table ---
    print("\nEpoch | Seed 42 | Seed 43 | Seed 44 | FP rate | FP count")
    print("------|---------|---------|---------|---------|----------")
    for i, ep in enumerate(epochs):
        sc = fp_count_per_seed[i]
        print(f"{ep:>5} | {sc[0]}/{n_normal} | {sc[1]}/{n_normal} | {sc[2]}/{n_normal} | "
              f"{fp_rate_means[i]:.4f} ± {fp_rate_stds[i]:.4f} | "
              f"{fp_count_means[i]:.2f} ± {fp_count_stds[i]:.2f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_false_positives.json"
    payload = {
        "seeds": SEEDS,
        "images": 625,
        "normal_cases": n_normal,
        "dataset_snapshot_id": data.snapshot_id,
        "epochs": epochs,
        "fp_count_per_seed": fp_count_per_seed,
        "fp_count": {"mean": fp_count_means, "std": fp_count_stds},
        "fp_rate": {"mean": fp_rate_means, "std": fp_rate_stds},
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# False positives vs. epoch",
        "",
        "Preliminary milestones run: seeds 42/43/44, 625 images each.",
        "",
        "A false positive is a Normal (mass-free) case the model flags as "
        "containing a mass. Every image is one patient with at most one mass, so "
        "this is a single case-level signal.",
        "",
        "Values are mean ± standard deviation across the three seeds.",
        "",
        f"Normal test cases: {n_normal}.",
        "",
        "| Epoch | Seed 42 | Seed 43 | Seed 44 | Mean FP rate | Mean FP count |",
        "|------:|:-------:|:-------:|:-------:|-------------:|--------------:|",
    ]
    for i, ep in enumerate(epochs):
        sc = fp_count_per_seed[i]
        md_lines.append(
            f"| {ep} | {sc[0]}/{n_normal} | {sc[1]}/{n_normal} | {sc[2]}/{n_normal} | "
            f"{fp_rate_means[i]:.4f} ± {fp_rate_stds[i]:.4f} | "
            f"{fp_count_means[i]:.2f} ± {fp_count_stds[i]:.2f} |"
        )
    md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_false_positives.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot: FP count (left) and rate (right) on twin axes ---
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.errorbar(epochs, fp_count_means, yerr=fp_count_stds, fmt='-o',
                color=FP_COUNT_COLOR, label='False positives', markersize=7,
                linewidth=1.5, capsize=3, markeredgecolor='white',
                markeredgewidth=1.5, zorder=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('False positives (count)', color=FP_COUNT_COLOR)
    ax.set_xlim(left=0, right=epochs[-1])
    ax.set_xticks(epochs)
    ax.tick_params(axis='y', labelcolor=FP_COUNT_COLOR)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.grid(False, axis='x')

    ax2 = ax.twinx()
    ax2.errorbar(epochs, fp_rate_means, yerr=fp_rate_stds, fmt='-o',
                 color=FP_RATE_COLOR, label='FP rate', markersize=7,
                 linewidth=1.5, capsize=3, markeredgecolor='white',
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
