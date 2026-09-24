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

EXPERIMENT_NAME = "milestones_pilot"
EVALUATION_SPLIT = "test"

SEEDS = [42, 43, 44]
EPOCHS = [50, 100, 150, 300, 500, 750]

# Snapshot checkpoints reported as extra rows, by configuration_id suffix.
# ``best``/``best_mass`` have per-seed selection epochs (see BEST_EPOCHS), so
# their epoch label is a placeholder reset below.
SNAPSHOT_DIRS = {
    "best": ("best", "Best"),
    "best_mass": ("best mass", "Best mass"),
    "final": ("1000", "Final"),
}

# Epoch at which ``best`` / ``best_mass`` were last selected, per seed. Recovered
# from each seed's training log (the epoch of the final EMA-improving update).
BEST_EPOCHS = {
    ("best", 42): 838,
    ("best", 43): 695,
    ("best", 44): 999,
    ("best_mass", 42): 838,
    ("best_mass", 43): 708,
    ("best_mass", 44): 999,
}

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


def summarize_checkpoint(prediction_rows, suffix):
    """Return (seed_counts, fp_count_mean, fp_count_std, fp_rate_mean, fp_rate_std, n_normal)."""
    seed_counts = []
    n_normal = None
    for seed in SEEDS:
        config_id = f"predictions_milestones_625images_seed{seed}_{suffix}"
        rows = [r for r in prediction_rows if r.configuration_id == config_id]
        if not rows:
            print(f"WARNING: no rows for {config_id}")
            continue
        fp_count, n = count_false_positives(rows)
        if n_normal is None:
            n_normal = n
        seed_counts.append(fp_count)

    rates = [c / n_normal if n_normal else 0.0 for c in seed_counts]
    count_mean = float(np.nanmean(seed_counts)) if seed_counts else float('nan')
    count_std = float(np.nanstd(seed_counts, ddof=1)) if seed_counts else float('nan')
    rate_mean = float(np.nanmean(rates)) if rates else float('nan')
    rate_std = float(np.nanstd(rates, ddof=1)) if rates else float('nan')
    return seed_counts, count_mean, count_std, rate_mean, rate_std, n_normal


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    prediction_rows = [
        row for row in data.rows
        if row.experiment_name == EXPERIMENT_NAME
        and row.evaluation_split == EVALUATION_SPLIT
    ]
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(prediction_rows)} rows")

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
            rows = [r for r in prediction_rows if r.configuration_id == folder]
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

    # Selected and final checkpoints reported as extra rows.
    snapshots = []
    for suffix, (epoch_label, display_name) in SNAPSHOT_DIRS.items():
        seed_counts, count_m, count_s, rate_m, rate_s, _ = summarize_checkpoint(prediction_rows, suffix)
        if not seed_counts:
            print(f"WARNING: no rows for {suffix}, skipping")
            continue
        snapshots.append((suffix, epoch_label, display_name, seed_counts, rate_m, rate_s, count_m, count_s))

    # --- Terminal table ---
    print("\nEpoch | Seed 42 | Seed 43 | Seed 44 | FP rate | FP count")
    print("------|---------|---------|---------|---------|----------")
    for i, ep in enumerate(epochs):
        sc = fp_count_per_seed[i]
        print(f"{ep:>5} | {sc[0]}/{n_normal} | {sc[1]}/{n_normal} | {sc[2]}/{n_normal} | "
              f"{fp_rate_means[i]:.4f} ± {fp_rate_stds[i]:.4f} | "
              f"{fp_count_means[i]:.2f} ± {fp_count_stds[i]:.2f}")

    print("\nSelected and final checkpoints")
    print("Epoch | Seed 42 | Seed 43 | Seed 44 | FP rate | FP count")
    print("------|---------|---------|---------|---------|----------")
    for suffix, epoch_label, display_name, sc, rate_m, rate_s, count_m, count_s in snapshots:
        print(f"{epoch_label:<17} | {sc[0]}/{n_normal} | {sc[1]}/{n_normal} | {sc[2]}/{n_normal} | "
              f"{rate_m:.4f} ± {rate_s:.4f} | "
              f"{count_m:.2f} ± {count_s:.2f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_false_positives.json"
    payload = {
        "seeds": SEEDS,
        "images": 625,
        "normal_cases": n_normal,
        "dataset_snapshot_id": data.snapshot_id,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_split": EVALUATION_SPLIT,
        "best_epochs": {
            str(seed): BEST_EPOCHS[("best", seed)] for seed in SEEDS
        },
        "best_mass_epochs": {
            str(seed): BEST_EPOCHS[("best_mass", seed)] for seed in SEEDS
        },
        "epochs": epochs,
        "fp_count_per_seed": fp_count_per_seed,
        "fp_count": {"mean": fp_count_means, "std": fp_count_stds},
        "fp_rate": {"mean": fp_rate_means, "std": fp_rate_stds},
        "snapshots": [
            {
                "suffix": suffix,
                "epoch_label": epoch_label,
                "display_name": display_name,
                "fp_count_per_seed": sc,
                "fp_count": {"mean": count_m, "std": count_s},
                "fp_rate": {"mean": rate_m, "std": rate_s},
            }
            for suffix, epoch_label, display_name, sc, rate_m, rate_s, count_m, count_s in snapshots
        ],
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
        "**Predetermined saved checkpoints**",
        "| Checkpoint | Seed 42 | Seed 43 | Seed 44 | Mean FP rate | Mean FP count |",
        "|------:|:-------:|:-------:|:-------:|-------------:|--------------:|",
    ]
    for i, ep in enumerate(epochs):
        sc = fp_count_per_seed[i]
        md_lines.append(
            f"| {ep} | {sc[0]}/{n_normal} | {sc[1]}/{n_normal} | {sc[2]}/{n_normal} | "
            f"{fp_rate_means[i]:.4f} ± {fp_rate_stds[i]:.4f} | "
            f"{fp_count_means[i]:.2f} ± {fp_count_stds[i]:.2f} |"
        )
    md_lines.extend([
        "",
        "**Selected and final checkpoints**",
        "| Checkpoint | Seed 42 | Seed 43 | Seed 44 | Mean FP rate | Mean FP count |",
        "|------:|:-------:|:-------:|:-------:|-------------:|--------------:|",
    ])
    for suffix, epoch_label, display_name, sc, rate_m, rate_s, count_m, count_s in snapshots:
        label = display_name if suffix in ("best", "best_mass") else f"{epoch_label} ({display_name})"
        md_lines.append(
            f"| {label} | {sc[0]}/{n_normal} | {sc[1]}/{n_normal} | {sc[2]}/{n_normal} | "
            f"{rate_m:.4f} ± {rate_s:.4f} | "
            f"{count_m:.2f} ± {count_s:.2f} |"
        )
    best_s = f"`Best` epoch was {BEST_EPOCHS[('best', 42)]} for seed 42, {BEST_EPOCHS[('best', 43)]} for seed 43, and {BEST_EPOCHS[('best', 44)]} for seed 44."
    best_mass_s = f"`Best mass` epoch was {BEST_EPOCHS[('best_mass', 42)]} for seed 42, {BEST_EPOCHS[('best_mass', 43)]} for seed 43, and {BEST_EPOCHS[('best_mass', 44)]} for seed 44."
    md_lines.extend(["", best_s, best_mass_s, ""])
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
