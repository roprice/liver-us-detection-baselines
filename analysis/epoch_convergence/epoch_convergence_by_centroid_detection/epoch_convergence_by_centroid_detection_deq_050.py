"""Centroid detection vs. epoch for the preliminary milestones run (3 seeds, 625 images each).

Centroid detection uses the LUNA16-style 2D criterion: for each ground-truth
mass component, find the closest retained predicted component's centroid. The
component is detected when that predicted centroid lies within 0.5x the
ground-truth component's equivalent circular diameter ``D = 2 * sqrt(area / pi)``.
A case is detected when any ground-truth component is detected.

This is scale-relative (the tolerance grows with the mass's own size), not a
fixed physical distance, because this dataset carries no physical spacings.

On this dataset every image is one patient with at most one mass, so detection
is effectively case-level. Reports, per milestone checkpoint and per mass
grouping (combined / malignant / benign): case-level recall and false-positive
rate over Normal cases, averaged across the three seeds with standard deviation.

Reads the canonical predictions dataset (analysis/predictions_dataset) rather
than the raw masks; centroid_detection_deq_050_flag and pathology come from there.

Three seeds (42/43/44), 625 images each; error bars are the across-seed std.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection_deq_050.py
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
OUT_BASE = "epoch_convergence_by_centroid_detection_deq_050"

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

# Metric key -> (legend label, color).
METRICS = {
    "case_recall":      ("Case recall",      "#f5c0aa"),
    "case_fp_rate":     ("Case FP rate",     "#3b6d11"),
}

PLOT_SERIES = [
    ("combined",  "case_recall", "Combined centroid",  "#2a78d6"),
    ("malignant", "case_recall", "Malignant centroid", "#eb6834"),
    ("benign",    "case_recall", "Benign centroid",    "#3b6d11"),
]


def evaluate_centroid(rows, positive_class):
    """Return (tp, fp, fn, n_normal) at case level for centroid detection.

    A mass-present case is detected when the flag field is True; a Normal case
    with any retained prediction is a false alarm. All facts come from the
    loaded predictions dataset, not the raw masks.
    """
    tp = fn = 0
    fp = 0
    n_normal = 0

    for row in rows:
        if positive_class == "combined":
            is_positive = row.pathology in ("malignant", "benign")
        else:
            is_positive = row.pathology == positive_class

        if is_positive:
            if getattr(row, FLAG_FIELD):
                tp += 1
            else:
                fn += 1
        elif row.pathology == "normal":
            n_normal += 1
            if row.normal_false_positive:
                fp += 1

    return tp, fp, fn, n_normal


def evaluate_epoch(rows, positive_class="malignant"):
    """Run centroid eval for one checkpoint, return case recall and FP rate."""
    ctp, cfp, cfn, n_normal = evaluate_centroid(rows, positive_class)

    crec = ctp / (ctp + cfn) if (ctp + cfn) > 0 else float('nan')
    cfp_rate = cfp / n_normal if n_normal > 0 else float('nan')

    return crec, cfp_rate


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
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(data.rows)} rows")

    combined = summarize_epochs(data, "combined")
    malignant = summarize_epochs(data, "malignant")
    benign = summarize_epochs(data, "benign")

    stats = {"combined": combined, "malignant": malignant, "benign": benign}
    epochs = EPOCHS

    # --- Terminal tables ---
    for title, group in [("All masses", "combined"),
                         ("Malignant", "malignant"),
                         ("Benign", "benign")]:
        means, stds = stats[group]
        print(f"\n{title} — Epoch | CRec | CFP")
        print(f"{'':>7}  ------|------|-----")
        for i, ep in enumerate(epochs):
            print(f"{ep:>5} | {fmt(means['case_recall'][i], stds['case_recall'][i])} | "
                  f"{fmt(means['case_fp_rate'][i], stds['case_fp_rate'][i])}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / f"{OUT_BASE}.json"
    payload = {
        "title": f"Case-level centroid-based detection (deq={DEQ_FACTOR:g}) by saved milestone epoch",
        "description": ("Three preliminary milestones test runs: seeds 42/43/44, "
                        "625 images each, centroid detection = predicted centroid "
                        f"within {DEQ_FACTOR:g}x GT equivalent diameter, noise floor "
                        f"{NOISE_FLOOR}. Values are mean ± std across seeds."),
        "noise_floor": NOISE_FLOOR,
        "centroid_definition": f"predicted centroid within {DEQ_FACTOR:g}x GT equivalent diameter",
        "deq_factor": DEQ_FACTOR,
        "dataset_snapshot_id": data.snapshot_id,
        "seeds": SEEDS,
        "images": 625,
        "epochs": epochs,
        "combined": metric_payload(combined[0], combined[1]),
        "malignant": metric_payload(malignant[0], malignant[1]),
        "benign": metric_payload(benign[0], benign[1]),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        f"# Case-level centroid-based detection (deq={DEQ_FACTOR:g}) by saved milestone epoch",
        "",
        f"Three preliminary milestones test runs: seeds 42/43/44, 625 images each, "
        f"centroid detection = predicted centroid within {DEQ_FACTOR:g}x GT "
        f"equivalent diameter, noise floor {NOISE_FLOOR}.",
        "",
        "A mass case is detected when the closest retained predicted centroid "
        f"lies within {DEQ_FACTOR:g}x the ground-truth mass's equivalent circular "
        "diameter.",
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
        md_lines.append("| Epoch | Detection | Normal cases FP rate |")
        md_lines.append("|------:|----------:|-------------------:|")
        for i, ep in enumerate(epochs):
            md_lines.append(
                f"| {ep} | {fmt(means['case_recall'][i], stds['case_recall'][i])} | "
                f"{fmt(means['case_fp_rate'][i], stds['case_fp_rate'][i])} |"
            )
        md_lines.append("")
    md_path = OUT_DIR / f"{OUT_BASE}.md"
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
    ax.set_ylabel('Centroid recall')
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
