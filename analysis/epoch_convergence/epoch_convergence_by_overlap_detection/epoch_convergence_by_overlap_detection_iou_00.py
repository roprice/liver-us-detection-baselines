"""Overlap detection quality vs. epoch for the preliminary milestones run
(3 seeds, 625 images each) at IoU > 0.

Evaluates the six milestone checkpoints (50/100/150/300/500/750) on mass
*detection* metrics using overlap-based matching with IoU > 0 (any overlap),
reported for all masses combined and separately for malignant and benign masses,
averaged across the three seeds with standard deviation.

Per mass grouping, reports case-level recall and false-positive rate over Normal
cases. A second table reports the selected (best / best mass) and final
checkpoints.

Reads the canonical predictions dataset (analysis/predictions_dataset) rather
than the raw masks; overlap_detection_iou_00_flag and pathology come from there.

Three seeds (42/43/44), 625 images each; error bars are the across-seed std.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_00.py
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

# Script lives at analysis/epoch_convergence/epoch_convergence_by_overlap_detection/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# Make the project root importable so the predictions-dataset loader can be
# used regardless of the current working directory.
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

NOISE_FLOOR = "0.03% of image area"

SEEDS = [42, 43, 44]
EPOCHS = [50, 100, 150, 300, 500, 750]

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

# Metric key -> (legend label, color).
METRICS = {
    "case_recall":      ("Case recall",      "#f5c0aa"),
    "case_fp_rate":     ("Case FP rate",     "#3b6d11"),
}

# Case-level overlap-based detection across all mass-present cases and by
# pathology. Combined is plotted first so the legend follows the table order.
PLOT_SERIES = [
    ("combined",  "case_recall", "Combined detection",  "#2a78d6"),
    ("malignant", "case_recall", "Malignant detection", "#eb6834"),
    ("benign",    "case_recall", "Benign detection",    "#3b6d11"),
]


def evaluate_detection(rows, positive_class):
    """Return (tp, fp, fn, n_normal) at case level for overlap (IoU > 0) detection."""
    tp = fn = 0
    fp = 0
    n_normal = 0

    for row in rows:
        flagged = row.overlap_detection_iou_00_flag
        if positive_class == "combined":
            is_positive = row.pathology in ("malignant", "benign")
        else:
            is_positive = row.pathology == positive_class

        if is_positive:
            if flagged:
                tp += 1
            else:
                fn += 1
        elif row.pathology == "normal":
            n_normal += 1
            if row.normal_false_positive:
                fp += 1

    return tp, fp, fn, n_normal


def evaluate_epoch(rows, positive_class="malignant"):
    """Run overlap (IoU > 0) eval for one checkpoint, return recall and FP rate."""
    ctp, cfp, cfn, n_normal = evaluate_detection(rows, positive_class)

    crec = ctp / (ctp + cfn) if (ctp + cfn) > 0 else float('nan')
    cfp_rate = cfp / n_normal if n_normal > 0 else float('nan')

    return crec, cfp_rate


def summarize_snapshot(data, positive_class, suffix):
    """Return (per_seed_recall, recall_mean, recall_std, fp_mean, fp_std) for one
    snapshot checkpoint across seeds."""
    recalls = []
    fp_rates = []
    for seed in SEEDS:
        folder = f"predictions_milestones_625images_seed{seed}_{suffix}"
        rows = [r for r in data.rows if r.configuration_id == folder]
        if not rows:
            print(f"WARNING: no rows for {folder}")
            recalls.append(float('nan'))
            fp_rates.append(float('nan'))
            continue
        rec, fp = evaluate_epoch(rows, positive_class)
        recalls.append(rec)
        fp_rates.append(fp)
    recall_mean = float(np.nanmean(recalls)) if recalls else float('nan')
    recall_std = float(np.nanstd(recalls, ddof=1)) if recalls else float('nan')
    fp_mean = float(np.nanmean(fp_rates)) if fp_rates else float('nan')
    fp_std = float(np.nanstd(fp_rates, ddof=1)) if fp_rates else float('nan')
    return recalls, recall_mean, recall_std, fp_mean, fp_std


def summarize_epochs(data, positive_class):
    """Return {metric: (means, stds)} across seeds, per epoch."""
    means = {key: [] for key in METRICS}
    stds = {key: [] for key in METRICS}
    for epoch in EPOCHS:
        recalls = []
        fp_rates = []
        for seed in SEEDS:
            folder = f"predictions_milestones_625images_seed{seed}_epoch{epoch}"
            rows = [r for r in data.rows if r.configuration_id == folder]
            if not rows:
                print(f"WARNING: no rows for {folder}")
                continue
            rec, fp = evaluate_epoch(rows, positive_class)
            recalls.append(rec)
            fp_rates.append(fp)
        means["case_recall"].append(float(np.nanmean(recalls)) if recalls else float('nan'))
        stds["case_recall"].append(float(np.nanstd(recalls, ddof=1)) if recalls else float('nan'))
        means["case_fp_rate"].append(float(np.nanmean(fp_rates)) if fp_rates else float('nan'))
        stds["case_fp_rate"].append(float(np.nanstd(fp_rates, ddof=1)) if fp_rates else float('nan'))
    return means, stds


def metric_payload(means, stds):
    return {
        "Detection": {"mean": means["case_recall"], "std": stds["case_recall"]},
        "False positive rate": {"mean": means["case_fp_rate"], "std": stds["case_fp_rate"]},
    }


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def main():
    definition = "IoU > 0"

    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(data.rows)} rows")

    stats = {
        "combined": summarize_epochs(data, "combined"),
        "malignant": summarize_epochs(data, "malignant"),
        "benign": summarize_epochs(data, "benign"),
    }

    # Snapshot detection per grouping.
    snapshot_stats = {}
    for group in ("combined", "malignant", "benign"):
        entries = []
        for suffix, display in SNAPSHOTS:
            recalls, rec_m, rec_s, fp_m, fp_s = summarize_snapshot(data, group, suffix)
            entries.append({
                "suffix": suffix,
                "display": display,
                "per_seed_recall": recalls,
                "detection": {"mean": rec_m, "std": rec_s},
                "fp_rate": {"mean": fp_m, "std": fp_s},
            })
        snapshot_stats[group] = entries

    epochs = EPOCHS

    # --- Terminal tables ---
    for title, group in [("All masses", "combined"),
                         ("Malignant", "malignant"),
                         ("Benign", "benign")]:
        means, stds = stats[group]
        print(f"\n{title} — Checkpoint | CRec | CFP")
        print(f"{'':>7}  ------|------|-----")
        for i, ep in enumerate(epochs):
            print(f"{ep:>5} | {fmt(means['case_recall'][i], stds['case_recall'][i])} | "
                  f"{fmt(means['case_fp_rate'][i], stds['case_fp_rate'][i])}")
        print(f"{'':>7}  --- selected/final ---")
        for e in snapshot_stats[group]:
            print(f"{e['display']:<5} | {fmt(e['detection']['mean'], e['detection']['std'])} | "
                  f"{fmt(e['fp_rate']['mean'], e['fp_rate']['std'])}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_overlap_detection_iou_00.json"
    payload = {
        "title": "Case-level overlap-based detection (IoU>0) by saved milestone epoch",
        "description": ("Three preliminary milestones test runs: seeds 42/43/44, "
                        "625 images each, detection = IoU > 0, "
                        f"noise floor {NOISE_FLOOR}. Values are mean ± std across seeds."),
        "noise_floor": NOISE_FLOOR,
        "detection_definition": definition,
        "dataset_snapshot_id": data.snapshot_id,
        "seeds": SEEDS,
        "images": 625,
        "epochs": epochs,
        "best_epochs": {str(s): BEST_EPOCHS[("best", s)] for s in SEEDS},
        "best_mass_epochs": {str(s): BEST_EPOCHS[("best_mass", s)] for s in SEEDS},
        "combined": metric_payload(stats["combined"][0], stats["combined"][1]),
        "malignant": metric_payload(stats["malignant"][0], stats["malignant"][1]),
        "benign": metric_payload(stats["benign"][0], stats["benign"][1]),
        "snapshots": snapshot_stats,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# Case-level overlap-based detection (IoU>0) by saved milestone epoch",
        "",
        f"Three preliminary milestones test runs: seeds 42/43/44, 625 images each, "
        f"detection = {definition}, noise floor {NOISE_FLOOR}.",
        "",
        "A mass-present case is detected when overlap_detection_iou_00_flag is True.",
        "",
        "Each table reports case-level false positives computed on normal cases "
        "only. A normal case with any prediction is a false alarm.",
        "",
        "Values are mean ± standard deviation across the three seeds.",
        "",
    ]

    for title, group in [("All masses", "combined"),
                         ("Malignant masses", "malignant"),
                         ("Benign masses", "benign")]:
        means, stds = stats[group]
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append("**Predetermined saved checkpoints**")
        md_lines.append("| Checkpoint | Detection | Normal cases FP rate |")
        md_lines.append("|------:|----------:|-------------------:|")
        for i, ep in enumerate(epochs):
            md_lines.append(
                f"| {ep} | {fmt(means['case_recall'][i], stds['case_recall'][i])} | "
                f"{fmt(means['case_fp_rate'][i], stds['case_fp_rate'][i])} |"
            )
        md_lines.append("")
        md_lines.append("**Selected and final checkpoints**")
        md_lines.append("| Checkpoint | Detection | Normal cases FP rate |")
        md_lines.append("|------:|----------:|-------------------:|")
        for e in snapshot_stats[group]:
            label = e["display"] if e["suffix"] in ("best", "best_mass") else f"1000 ({e['display']})"
            md_lines.append(
                f"| {label} | {fmt(e['detection']['mean'], e['detection']['std'])} | "
                f"{fmt(e['fp_rate']['mean'], e['fp_rate']['std'])} |"
            )
        best_s = (f"`Best` epoch was {BEST_EPOCHS[('best', 42)]} for seed 42, "
                  f"{BEST_EPOCHS[('best', 43)]} for seed 43, and {BEST_EPOCHS[('best', 44)]} for seed 44.")
        best_mass_s = (f"`Best mass` epoch was {BEST_EPOCHS[('best_mass', 42)]} for seed 42, "
                       f"{BEST_EPOCHS[('best_mass', 43)]} for seed 43, and {BEST_EPOCHS[('best_mass', 44)]} for seed 44.")
        md_lines.extend(["", best_s, best_mass_s, ""])

    md_path = OUT_DIR / "epoch_convergence_by_overlap_detection_iou_00.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for src, key, label, color in PLOT_SERIES:
        means, stds = stats[src]
        ax.errorbar(epochs, means[key], yerr=stds[key], fmt='-o', color=color,
                    label=label, markersize=7, linewidth=1.5, capsize=3,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Detection rate')
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
        out = OUT_DIR / f'epoch_convergence_by_overlap_detection_iou_00.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
