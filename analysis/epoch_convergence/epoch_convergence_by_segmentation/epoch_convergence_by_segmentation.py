"""Check whether the preliminary milestones run (Dataset 001 AUL, seed 42) has
converged using real segmentation Dice on hard predictions, rather than the
training-time pseudo Dice alone.

For each milestone epoch (50/100/150/300/500/750) the script reports
segmentation Dice (liver, malignant mass, benign mass, combined mass) from the
canonical predictions dataset (analysis/predictions_dataset), then plots Dice
vs epoch and reports the tail-epoch slope. The final/best/best mass snapshot
folders are evaluated as additional rows in the report.

The training-time EMA mass pseudo-Dice trace (from the training log) is overlaid
in gray, dotted, on its own legend, to show the continuous climb that the sparse
checkpoint predictions cannot.

This is a single-seed script: only seed 42 prediction data exists for this
preliminary run, so there is no seed averaging or multi-folder ablation loop.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_segmentation/epoch_convergence_by_segmentation.py
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.legend as mlegend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

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

# Metric key -> (color, legend label, line style) for the test-set predictions.
SERIES = {
    'liver':     ('#2a78d6', 'Liver',          '-'),
    'malignant': ('#eb6834', 'Malignant mass', '-'),
    'benign':    ('#3b6d11', 'Benign mass',    '-'),
}

# Style for the overlaid training-time mass pseudo-Dice trace.
PSEUDO_STYLE = dict(color='#8c8c8c', linestyle=':', linewidth=1.2)

# Script lives at analysis/epoch_convergence_by_segmentation/, so this folder is
# the output dir and the project root is two parents up.

# Keep all outputs (Markdown/pdf/png) in the same folder as this script.
OUT_DIR = SCRIPT_DIR

# Training log (nnU-Net training_log_*.txt) for the single seed-42 run.
LOG_DIR = PROJECT_ROOT / "nnUNet_results/Dataset001_AUL" / (
    "nnUNetTrainerMilestones_seed42__nnUNetPlans__2d/fold_0")
LOG_DIR = str(LOG_DIR)

# Folder name -> milestone epoch whose convergence we want to assess. Matches
# nnUNetTrainerMilestones.MILESTONE_EPOCHS = {50, 100, 150, 300, 500, 750}.
EPOCH_DIRS = [
    ("predictions_milestones_625images_seed42_epoch50", 50),
    ("predictions_milestones_625images_seed42_epoch100", 100),
    ("predictions_milestones_625images_seed42_epoch150", 150),
    ("predictions_milestones_625images_seed42_epoch300", 300),
    ("predictions_milestones_625images_seed42_epoch500", 500),
    ("predictions_milestones_625images_seed42_epoch750", 750),
]

# Snapshot checkpoints reported as extra rows, by configuration_id suffix.
SNAPSHOT_DIRS = [
    ("best", "best"),
    ("best_mass", "best mass"),
    ("final", "final"),
]


def evaluate_rows(rows):
    """Dice across a configuration's rows, with mass Dice split by pathology."""
    liver_scores = []
    malignant_scores = []
    benign_scores = []
    combined_mass_scores = []

    for row in rows:
        liver_scores.append(row.liver_dice)
        if row.pathology in ("malignant", "benign"):
            combined_mass_scores.append(row.mass_dice)
            if row.pathology == "benign":
                benign_scores.append(row.mass_dice)
            else:
                malignant_scores.append(row.mass_dice)

    return {
        "liver": np.mean(liver_scores) if liver_scores else 0.0,
        "malignant": np.mean(malignant_scores) if malignant_scores else 0.0,
        "benign": np.mean(benign_scores) if benign_scores else 0.0,
        "combined_mass": (np.mean(combined_mass_scores)
                          if combined_mass_scores else 0.0),
    }


def find_training_log(log_dir):
    """Return the training_log_*.txt path, or None if absent."""
    matches = glob.glob(os.path.join(log_dir, "training_log_*.txt"))
    return matches[0] if matches else None


def parse_raw_mass_pseudo_dice(log_path):
    """Return (epochs, EMA mass pseudo Dice) from the training log.

    The custom trainer emits exactly one "Mass pseudo Dice: X, EMA: Y" line per
    epoch (from on_epoch_end), in epoch order, with no separate "Epoch N" header
    lines in the native nnU-Net log. So the first such line is epoch 1, the
    second is epoch 2, and so on. The EMA trace is used for the overlay because
    it is smooth; the raw value is noisy.
    """
    epochs = []
    ema = []
    line_idx = 0
    for line in open(log_path):
        m = re.search(r'Mass pseudo Dice: ([\d.]+), EMA: ([\d.]+)', line)
        if m:
            line_idx += 1
            epochs.append(line_idx)  # human epoch number, matches checkpoints
            ema.append(float(m.group(2)))
    return epochs, ema


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(data.rows)} rows")

    epochs = []
    dice = {"liver": [], "malignant": [], "benign": [], "combined_mass": []}
    for folder_name, epoch in EPOCH_DIRS:
        rows = [r for r in data.rows if r.configuration_id == folder_name]
        if not rows:
            print(f"WARNING: no rows for {folder_name}, skipping epoch {epoch}")
            continue
        r = evaluate_rows(rows)
        epochs.append(epoch)
        for key in dice:
            dice[key].append(r[key])

    # Snapshots reported as extra rows alongside the milestones.
    snapshots = []
    for folder_label, display_label in SNAPSHOT_DIRS:
        config_id = f"predictions_milestones_625images_seed42_{folder_label}"
        rows = [r for r in data.rows if r.configuration_id == config_id]
        if not rows:
            print(f"WARNING: no rows for {config_id}, skipping {folder_label}")
            continue
        snapshots.append((display_label, evaluate_rows(rows)))

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # Tail-epoch slope over the last two milestone checkpoints, to check whether
    # segmentation Dice is still climbing at the final milestone. Computed on the
    # malignant (mass) class, the label class of interest for the convergence claim.
    tail_n = min(2, len(epochs))
    tail_metric = dice["malignant"]
    slope = ((tail_metric[-1] - tail_metric[-tail_n]) / (epochs[-1] - epochs[-tail_n])
             if epochs[-1] != epochs[-tail_n] else 0.0)

    print("Epoch | Liver | Malignant | Benign | Combined mass")
    print("------|-------|-----------|--------|--------------")
    for i, ep in enumerate(epochs):
        print(f"{ep:>5} | {dice['liver'][i]:.4f} | {dice['malignant'][i]:.4f} "
              f"| {dice['benign'][i]:.4f} | {dice['combined_mass'][i]:.4f}")
    for label, r in snapshots:
        print(f"{label:>5} | {r['liver']:.4f} | {r['malignant']:.4f} "
              f"| {r['benign']:.4f} | {r['combined_mass']:.4f}")
    print(f"\nFinal malignant {tail_metric[-1]:.4f}, "
          f"tail-{tail_n}-epoch slope {slope:+.5f}/epoch")

    # --- Markdown ---
    md_lines = [
        "# Segmentation quality vs. epoch",
        "",
        "Preliminary milestones run: seed 42, 625 images, joint-selected checkpoints.",
        "",
        "No error bars (single seed).",
        "",
        "| Epoch | Liver Dice | Malignant mass Dice | Benign mass Dice | Combined mass Dice |",
        "|------:|-----------:|--------------------:|----------------:|-------------------:|",
    ]
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {dice['liver'][i]:.4f} | {dice['malignant'][i]:.4f} | "
            f"{dice['benign'][i]:.4f} | {dice['combined_mass'][i]:.4f} |"
        )
    for label, r in snapshots:
        md_lines.append(
            f"| {label} | {r['liver']:.4f} | {r['malignant']:.4f} | "
            f"{r['benign']:.4f} | {r['combined_mass']:.4f} |"
        )
    md_lines.extend([
        "",
        f"Final malignant Dice: {tail_metric[-1]:.4f}. Tail-{tail_n}-epoch slope: "
        f"{slope:+.5f}/epoch.",
        "",
    ])
    md_path = os.path.join(OUT_DIR, "epoch_convergence_by_segmentation.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # EMA mass pseudo-Dice from the training log, providing a continuous real
    # estimate of the early climb that the sparse checkpoint predictions cannot.
    log_path = find_training_log(LOG_DIR)
    pseudo_epochs, pseudo_ema = [], []
    if log_path:
        pseudo_epochs, pseudo_ema = parse_raw_mass_pseudo_dice(log_path)
    else:
        print(f"WARNING: no training_log_*.txt found in {LOG_DIR}; "
              "pseudo-Dice trace omitted")

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, (color, label, linestyle) in SERIES.items():
        ax.plot(epochs, dice[key], linestyle, color=color, marker='o',
                markersize=7, linewidth=1.5, label=label, zorder=3,
                markeredgecolor='white', markeredgewidth=1.5)

    if pseudo_epochs:
        ax.plot(pseudo_epochs, pseudo_ema, color=PSEUDO_STYLE['color'],
                linestyle=PSEUDO_STYLE['linestyle'], linewidth=PSEUDO_STYLE['linewidth'],
                label='Mass pseudo-Dice', zorder=1)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')

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

    plt.tight_layout(pad=1.2)

    handles_labels = ax.get_legend_handles_labels()
    all_handles, all_labels = handles_labels
    pred_handles = [h for h, l in zip(all_handles, all_labels) if l != 'Mass pseudo-Dice']
    pred_labels = ['Liver', 'Malignant mass', 'Benign mass']
    pseudo_handles = [h for h, l in zip(all_handles, all_labels) if l == 'Mass pseudo-Dice']

    common = dict(frameon=True, fancybox=False, borderpad=0.8, handlelength=2.5)
    legend_pred = mlegend.Legend(ax, pred_handles, pred_labels,
                                 title='Segmentation Dice', loc='lower right', **common)
    ax.add_artist(legend_pred)

    legend_pseudo = None
    if pseudo_handles:
        invisible = Line2D([0, 0], [0, 0], color='none', linewidth=0, marker='')
        padded_handles = pseudo_handles + [invisible] * 2
        padded_labels = ['Mass pseudo-Dice', '', '']
        legend_pseudo = mlegend.Legend(ax, padded_handles, padded_labels,
                                       title='Pseudo-Dice trace', loc='lower right',
                                       **common)
        ax.add_artist(legend_pseudo)

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bound = lambda lg: inv.transform(lg.get_window_extent())
    if legend_pseudo is not None:
        pred0, pred1 = bound(legend_pred)
        pseudo0, pseudo1 = bound(legend_pseudo)
        width = max(pred1[0] - pred0[0], pseudo1[0] - pseudo0[0])
        height = max(pred1[1] - pred0[1], pseudo1[1] - pseudo0[1])
        bottom = min(pred0[1], pseudo0[1])
        right = max(pred1[0], pseudo1[0])
        gap = 0.006
        new_pred = (right - width, bottom, width, height)
        new_pseudo = (right - 2 * width - gap, bottom, width, height)
        legend_pred.set_bbox_to_anchor(new_pred, transform=fig.transFigure)
        legend_pseudo.set_bbox_to_anchor(new_pseudo, transform=fig.transFigure)
        legend_pred._mode = "expand"
        legend_pseudo._mode = "expand"
    else:
        legend_pred._mode = "expand"

    for ext in ('pdf', 'png'):
        out_path = os.path.join(OUT_DIR, f'epoch_convergence_by_segmentation.{ext}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Plot saved to: {out_path}')
    plt.close(fig)


if __name__ == "__main__":
    main()
