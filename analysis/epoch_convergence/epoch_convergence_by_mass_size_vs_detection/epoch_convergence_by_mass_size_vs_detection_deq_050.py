"""Centroid detection (deq 0.5) vs. epoch, broken down by mass size, for the
preliminary milestones run (3 seeds, 625 images each).

Centroid detection uses the LUNA16-style 2D criterion: for each ground-truth
mass component, find the closest retained predicted component's centroid. The
component is detected when that predicted centroid lies within 0.5x the
ground-truth component's equivalent circular diameter ``D = 2 * sqrt(area / pi)``.
A case is detected when any ground-truth component is detected.

This is scale-relative (the tolerance grows with the mass's own size), not a
fixed physical distance, because this dataset carries no physical spacings.

On this dataset every image is one patient with at most one mass, so detection
is effectively case-level. A Normal case with any retained prediction is a
false alarm.

Mass-present cases are split into 4 area bins on ground_truth_mass_area_px. For
each bin, and per mass grouping (combined / malignant / benign), this reports
case-level recall and the false-positive rate over Normal cases, averaged across
the three seeds with standard deviation. Snapshot (best / best mass / final)
checkpoints are reported as extra rows.

The false-positive rate is computed on the *full* normal slice (it does not
depend on mass size, since Normal cases have no mass), so it is identical across
bins within a grouping/checkpoint.

Reads the canonical predictions dataset (analysis/predictions_dataset) rather
than the raw masks; centroid_detection_deq_050_flag, normal_false_positive,
pathology, and ground_truth_mass_area_px come from there.

Three seeds (42/43/44), 625 images each; error bars are the across-seed std.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_mass_size_vs_detection/epoch_convergence_by_mass_size_vs_detection_deq_050.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# Make the project root importable so the predictions-dataset loader can be
# used regardless of the current working directory.
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

NOISE_FLOOR = "0.03% of image area"

DEQ_FACTOR = 0.5
FLAG_FIELD = "centroid_detection_deq_050_flag"
OUT_BASE = "epoch_convergence_by_mass_size_vs_detection_deq_050"

SEEDS = [42, 43, 44]
EPOCHS = [50, 100, 150, 300, 500, 750]

TARGET_EXPERIMENT_NAME = "milestones_pilot"
TARGET_EVALUATION_SPLIT = "test"

# Mass-area bins on ground_truth_mass_area_px. (label, inclusive_low,
# exclusive_high); high=None means "no upper bound".
MASS_BINS = [
    ("0\u20133,000 px\u00b2", 0, 3001),
    (">3,000\u201310,000 px\u00b2", 3001, 10001),
    (">10,000\u201330,000 px\u00b2", 10001, 30001),
    (">30,000 px\u00b2", 30001, None),
]

# Snapshot checkpoints (selected / final), by configuration_id suffix.
SNAPSHOTS = [
    ("best", "Best"),
    ("best_mass", "Best mass"),
    ("final", "Final"),
]

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

# Metric key -> label (not used for styling; kept for readability).
METRICS = {
    "case_recall": "Case recall",
    "case_fp_rate": "Case FP rate",
}


def in_bin(area, low, high):
    """Return True when ``area`` falls in [low, high) (high None -> open-ended)."""
    if area < low:
        return False
    return high is None or area < high


def count_mass_in_bin(rows, positive_class, low, high):
    """Count mass-present rows (for positive_class) whose GT area falls in the bin."""
    n = 0
    for row in rows:
        if positive_class == "combined":
            is_positive = row.pathology in ("malignant", "benign")
        else:
            is_positive = row.pathology == positive_class
        if is_positive and in_bin(row.ground_truth_mass_area_px, low, high):
            n += 1
    return n


def bin_counts_per_group(all_rows):
    """Return {group: {bin_label: n_images}}.

    Counts are computed on the first seed's ``best`` checkpoint rows (the test
    set is identical across checkpoints for this run), and should be the same
    across seeds and checkpoints.
    """
    counts = {}
    for group in ("combined", "malignant", "benign"):
        rows = [r for r in all_rows
                if r.configuration_id == f"predictions_milestones_625images_seed{SEEDS[0]}_best"]
        counts[group] = {
            label: count_mass_in_bin(rows, group, low, high)
            for label, low, high in MASS_BINS
        }
    return counts


def evaluate_centroid_bin(rows, positive_class, low, high=None):
    """Return (tp, fp, fn, n_normal) at case level for one mass-size bin.

    Only mass-present rows whose ground-truth area falls in the bin count toward
    tp/fn. The FP denominator is the *full* normal slice (Normal cases carry no
    mass, so they do not belong to any size bin); a Normal case with any
    retained prediction is a false alarm.
    """
    tp = fn = 0
    fp = 0
    n_normal = 0

    for row in rows:
        flagged = getattr(row, FLAG_FIELD)
        if positive_class == "combined":
            is_positive = row.pathology in ("malignant", "benign")
        else:
            is_positive = row.pathology == positive_class

        if is_positive:
            if in_bin(row.ground_truth_mass_area_px, low, high):
                if flagged:
                    tp += 1
                else:
                    fn += 1
        elif row.pathology == "normal":
            n_normal += 1
            if row.normal_false_positive:
                fp += 1

    return tp, fp, fn, n_normal


def evaluate_epoch_bin(rows, positive_class, low, high=None):
    """Run centroid eval for one checkpoint and bin; return recall and FP rate."""
    ctp, cfp, cfn, n_normal = evaluate_centroid_bin(rows, positive_class, low, high)

    crec = ctp / (ctp + cfn) if (ctp + cfn) > 0 else float('nan')
    cfp_rate = cfp / n_normal if n_normal > 0 else float('nan')

    return crec, cfp_rate


def summarize_bin_across_seeds(all_rows, positive_class, low, high, suffix=None):
    """Return (recalls, fp_rates, rec_mean, rec_std, fp_mean, fp_std) for one bin.

    ``suffix`` selects a snapshot checkpoint (``best``/``best_mass``/``final``);
    when None it is ignored by callers that instead pass a per-epoch filter via
    summarize_epochs. Kept separate so snapshot and epoch summaries stay explicit.
    """
    recalls = []
    fp_rates = []
    for seed in SEEDS:
        folder = f"predictions_milestones_625images_seed{seed}_{suffix}"
        rows = [r for r in all_rows if r.configuration_id == folder]
        if not rows:
            print(f"WARNING: no rows for {folder}")
            recalls.append(float('nan'))
            fp_rates.append(float('nan'))
            continue
        rec, fp = evaluate_epoch_bin(rows, positive_class, low, high)
        recalls.append(rec)
        fp_rates.append(fp)
    rec_mean = float(np.nanmean(recalls)) if recalls else float('nan')
    rec_std = float(np.nanstd(recalls, ddof=1)) if recalls else float('nan')
    fp_mean = float(np.nanmean(fp_rates)) if fp_rates else float('nan')
    fp_std = float(np.nanstd(fp_rates, ddof=1)) if fp_rates else float('nan')
    return recalls, fp_rates, rec_mean, rec_std, fp_mean, fp_std


def summarize_epoch_bin_across_seeds(all_rows, positive_class, low, high, epoch):
    """Return (rec_mean, rec_std, fp_mean, fp_std) for one bin at one epoch."""
    recalls = []
    fp_rates = []
    for seed in SEEDS:
        folder = f"predictions_milestones_625images_seed{seed}_epoch{epoch}"
        rows = [r for r in all_rows if r.configuration_id == folder]
        if not rows:
            print(f"WARNING: no rows for {folder}")
            continue
        rec, fp = evaluate_epoch_bin(rows, positive_class, low, high)
        recalls.append(rec)
        fp_rates.append(fp)
    rec_mean = float(np.nanmean(recalls)) if recalls else float('nan')
    rec_std = float(np.nanstd(recalls, ddof=1)) if recalls else float('nan')
    fp_mean = float(np.nanmean(fp_rates)) if fp_rates else float('nan')
    fp_std = float(np.nanstd(fp_rates, ddof=1)) if fp_rates else float('nan')
    return rec_mean, rec_std, fp_mean, fp_std


def summarize_epochs_by_bin(all_rows, positive_class):
    """Return {bin_label: {"case_recall": (means, stds), "case_fp_rate": (means, stds)}}.

    Bin keys follow MASS_BINS order; means/stds are lists aligned with EPOCHS.
    """
    result = {}
    for label, low, high in MASS_BINS:
        rec_means = []
        rec_stds = []
        fp_means = []
        fp_stds = []
        for epoch in EPOCHS:
            rm, rs, fm, fs = summarize_epoch_bin_across_seeds(
                all_rows, positive_class, low, high, epoch
            )
            rec_means.append(rm)
            rec_stds.append(rs)
            fp_means.append(fm)
            fp_stds.append(fs)
        result[label] = {
            "case_recall": (rec_means, rec_stds),
            "case_fp_rate": (fp_means, fp_stds),
        }
    return result


def summarize_snapshot_by_bin(all_rows, positive_class, suffix):
    """Return list of per-bin entries for one snapshot checkpoint."""
    entries = []
    for label, low, high in MASS_BINS:
        recalls, fp_rates, rm, rs, fm, fs = summarize_bin_across_seeds(
            all_rows, positive_class, low, high, suffix=suffix
        )
        entries.append({
            "label": label,
            "low": low,
            "high": high,
            "per_seed_recall": recalls,
            "detection": {"mean": rm, "std": rs},
            "fp_rate": {"mean": fm, "std": fs},
        })
    return entries


def bin_payload(stats_by_bin):
    """Convert {bin_label: {metric: (means, stds)}} into a JSON-ready structure."""
    return {
        label: {
            "Detection": {"mean": m["case_recall"][0], "std": m["case_recall"][1]},
            "False positive rate": {
                "mean": m["case_fp_rate"][0],
                "std": m["case_fp_rate"][1],
            },
        }
        for label, m in stats_by_bin.items()
    }


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def label_with_count(label, count):
    """Render a bin label with its image count, e.g. '0–3,000 px² (n=12)'."""
    return f"{label} (n={count})"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    rows = [
        row for row in data.rows
        if row.experiment_name == TARGET_EXPERIMENT_NAME
        and row.evaluation_split == TARGET_EVALUATION_SPLIT
    ]
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(rows)} selected rows")

    stats = {
        "combined": summarize_epochs_by_bin(rows, "combined"),
        "malignant": summarize_epochs_by_bin(rows, "malignant"),
        "benign": summarize_epochs_by_bin(rows, "benign"),
    }

    # Snapshot detection per grouping and bin.
    snapshot_stats = {}
    for group in ("combined", "malignant", "benign"):
        per_snapshot = []
        for suffix, display in SNAPSHOTS:
            per_snapshot.append({
                "suffix": suffix,
                "display": display,
                "bins": summarize_snapshot_by_bin(rows, group, suffix),
            })
        snapshot_stats[group] = per_snapshot

    epochs = EPOCHS

    bin_counts = bin_counts_per_group(rows)

    # --- Terminal tables ---
    for title, group in [("All masses", "combined"),
                         ("Malignant", "malignant"),
                         ("Benign", "benign")]:
        print(f"\n{title} — per mass-size bin")
        stat = stats[group]
        for label, low, high in MASS_BINS:
            rec_means, rec_stds = stat[label]["case_recall"]
            fp_means, fp_stds = stat[label]["case_fp_rate"]
            print(f"\n  {label_with_count(label, bin_counts[group][label])}")
            print(f"  {'':>5}  ------|------|-----")
            for i, ep in enumerate(epochs):
                print(f"  {ep:>5} | {fmt(rec_means[i], rec_stds[i])} | "
                      f"{fmt(fp_means[i], fp_stds[i])}")
            print(f"  {'':>5}  --- selected/final ---")
            for snap in snapshot_stats[group]:
                entry = snap["bins"][MASS_BINS.index((label, low, high))]
                print(f"  {snap['display']:<5} | "
                      f"{fmt(entry['detection']['mean'], entry['detection']['std'])} | "
                      f"{fmt(entry['fp_rate']['mean'], entry['fp_rate']['std'])}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / f"{OUT_BASE}.json"
    payload = {
        "title": "Case-level centroid-based detection (deq=0.5) by saved milestone epoch and mass size",
        "description": ("Three preliminary milestones test runs: seeds 42/43/44, "
                        "625 images each, centroid detection = predicted centroid "
                        "within 0.5x GT equivalent diameter, noise floor "
                        f"{NOISE_FLOOR}. Mass-present cases split by ground-truth "
                        "mass area into four bins. Values are mean ± std across "
                        "seeds."),
        "noise_floor": NOISE_FLOOR,
        "centroid_definition": "predicted centroid within 0.5x GT equivalent diameter",
        "deq_factor": DEQ_FACTOR,
        "mass_bins": [
            {"label": l, "low": lo, "high": hi} for l, lo, hi in MASS_BINS
        ],
        "bin_counts": bin_counts,
        "dataset_snapshot_id": data.snapshot_id,
        "experiment_name": TARGET_EXPERIMENT_NAME,
        "evaluation_split": TARGET_EVALUATION_SPLIT,
        "seeds": SEEDS,
        "images": 625,
        "epochs": epochs,
        "best_epochs": {str(s): BEST_EPOCHS[("best", s)] for s in SEEDS},
        "best_mass_epochs": {str(s): BEST_EPOCHS[("best_mass", s)] for s in SEEDS},
        "combined": bin_payload(stats["combined"]),
        "malignant": bin_payload(stats["malignant"]),
        "benign": bin_payload(stats["benign"]),
        "snapshots": snapshot_stats,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# Case-level centroid-based detection (deq=0.5) by saved milestone epoch and mass size",
        "",
        f"Three preliminary milestones test runs: seeds 42/43/44, 625 images each, "
        f"centroid detection = predicted centroid within 0.5x GT equivalent "
        f"diameter, noise floor {NOISE_FLOOR}.",
        "",
        "A mass case is detected when the closest retained predicted centroid "
        "lies within 0.5x the ground-truth mass's equivalent circular diameter.",
        "",
        "Mass into 4 bins according to size and centroid detection is measured "
        "against each by epoch",
    ]
    for label, low, high in MASS_BINS:
        md_lines.append(f"- {label_with_count(label, bin_counts['combined'][label])}")
    md_lines.extend([
        "",
        "Each table reports case-level false positives computed on normal cases "
        "only. A normal case with any prediction is a false alarm.",
        "",
        "Values are mean ± standard deviation across the three seeds.",
        "",
    ])

    for title, group in [("Both pathologies, 4 log-sized area bins", "combined"),
                         ("Malignant, 4 log-sized area bins", "malignant"),
                         ("Benign, 4 log-sized area bins", "benign")]:
        md_lines.append(f"## {title}")
        md_lines.append("")
        stat = stats[group]
        for label, low, high in MASS_BINS:
            md_lines.append(f"### {label_with_count(label, bin_counts[group][label])}")
            rec_means, rec_stds = stat[label]["case_recall"]
            fp_means, fp_stds = stat[label]["case_fp_rate"]
            md_lines.append("**Predetermined saved checkpoints**")
            md_lines.append("| Checkpoint | Centroid Detection | Normal cases FP rate |")
            md_lines.append("|------:|----------:|-------------------:|")
            for i, ep in enumerate(epochs):
                md_lines.append(
                    f"| {ep} | {fmt(rec_means[i], rec_stds[i])} | "
                    f"{fmt(fp_means[i], fp_stds[i])} |"
                )
            md_lines.append("")
            md_lines.append("**Selected and final checkpoints**")
            md_lines.append("| Checkpoint | Detection | Normal cases FP rate |")
            md_lines.append("|------:|----------:|-------------------:|")
            for snap in snapshot_stats[group]:
                entry = snap["bins"][MASS_BINS.index((label, low, high))]
                disp = snap["display"]
                snap_label = disp if snap["suffix"] in ("best", "best_mass") else f"1000 ({disp})"
                md_lines.append(
                    f"| {snap_label} | "
                    f"{fmt(entry['detection']['mean'], entry['detection']['std'])} | "
                    f"{fmt(entry['fp_rate']['mean'], entry['fp_rate']['std'])} |"
                )
            md_lines.append("")

    best_s = (f"`Best` epoch was {BEST_EPOCHS[('best', 42)]} for seed 42, "
              f"{BEST_EPOCHS[('best', 43)]} for seed 43, and {BEST_EPOCHS[('best', 44)]} for seed 44.")
    best_mass_s = (f"`Best mass` epoch was {BEST_EPOCHS[('best_mass', 42)]} for seed 42, "
                   f"{BEST_EPOCHS[('best_mass', 43)]} for seed 43, and {BEST_EPOCHS[('best_mass', 44)]} for seed 44.")
    md_lines.extend(["", best_s, best_mass_s, ""])

    md_path = OUT_DIR / f"{OUT_BASE}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot: single chart, combined centroid recall vs epoch, one line per bin ---
    fig, ax = plt.subplots(figsize=(8, 4.5))

    bin_colors = {
        "0\u20133,000 px\u00b2": "#2a78d6",
        ">3,000\u201310,000 px\u00b2": "#eb6834",
        ">10,000\u201330,000 px\u00b2": "#3b6d11",
        ">30,000 px\u00b2": "#7B5BD6",
    }

    for label, low, high in MASS_BINS:
        rec_means, rec_stds = stats["combined"][label]["case_recall"]
        ax.errorbar(epochs, rec_means, yerr=rec_stds, fmt='-o',
                    color=bin_colors[label],
                    label=label_with_count(label, bin_counts["combined"][label]),
                    markersize=7, linewidth=1.5, capsize=3,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Combined centroid recall')
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
    ax.legend(loc='best', frameon=True, fancybox=False, borderpad=0.8,
              handlelength=2.5)

    plt.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'{OUT_BASE}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
