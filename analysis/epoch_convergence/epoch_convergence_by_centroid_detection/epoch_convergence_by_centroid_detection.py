"""Centroid detection vs. epoch for the preliminary milestones run (seed 42, 625 images).

Centroid detection uses the LUNA16-style 2D criterion: for each ground-truth
mass component, find the closest retained predicted component's centroid. The
component is detected when that predicted centroid lies within half the
ground-truth component's equivalent circular diameter ``D = 2 * sqrt(area / pi)``.
A case is detected when any ground-truth component is detected.

This is scale-relative (the tolerance grows with the mass's own size), not a
fixed physical distance, because this dataset carries no physical spacings.

On this dataset every image is one patient with at most one mass, so detection
is effectively case-level. Reports, per milestone checkpoint and per mass
grouping (combined / malignant / benign): case-level recall and false-positive
rate over Normal cases.

Reads the canonical predictions dataset (analysis/predictions_dataset) rather
than the raw masks; centroid_detection_flag and pathology come from there.

Single-seed run (seed 42, 625 images); no error bars.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection.py
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

MIN_PRED_AREA = 100  # noise floor (px^2); already applied when the dataset was built

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

# Milestone checkpoints to report (epoch50..epoch750), excluding the snapshot
# best/best_mass/final checkpoints which have no fixed epoch.
EPOCH_DIRS = [
    ("predictions_milestones_625images_seed42_epoch50", 50),
    ("predictions_milestones_625images_seed42_epoch100", 100),
    ("predictions_milestones_625images_seed42_epoch150", 150),
    ("predictions_milestones_625images_seed42_epoch300", 300),
    ("predictions_milestones_625images_seed42_epoch500", 500),
    ("predictions_milestones_625images_seed42_epoch750", 750),
]

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

    A mass-present case is detected when centroid_detection_flag is True; a
    Normal case with any retained prediction is a false alarm. All facts come
    from the loaded predictions dataset, not the raw masks.
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
            if row.centroid_detection_flag:
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_predictions_dataset()
    print(f"Loaded predictions dataset: {data.snapshot_id}, "
          f"{len(data.rows)} rows")

    epochs = []
    metrics = {key: [] for key in METRICS}
    benign_metrics = {key: [] for key in METRICS}
    combined_metrics = {key: [] for key in METRICS}
    for folder_name, epoch in EPOCH_DIRS:
        rows = [r for r in data.rows if r.configuration_id == folder_name]
        if not rows:
            print(f"WARNING: no rows for {folder_name}, skipping epoch {epoch}")
            continue
        crec, cfp = evaluate_epoch(rows, positive_class="malignant")
        rb_rec, rb_fp = evaluate_epoch(rows, positive_class="benign")
        rc_rec, rc_fp = evaluate_epoch(rows, positive_class="combined")
        epochs.append(epoch)
        metrics["case_recall"].append(crec)
        metrics["case_fp_rate"].append(cfp)
        benign_metrics["case_recall"].append(rb_rec)
        benign_metrics["case_fp_rate"].append(rb_fp)
        combined_metrics["case_recall"].append(rc_rec)
        combined_metrics["case_fp_rate"].append(rc_fp)

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal tables ---
    header = "Epoch | CRec | CFP"
    sep = "------|------|-----"
    for title, table in [("All masses", combined_metrics),
                         ("Malignant", metrics),
                         ("Benign", benign_metrics)]:
        print(f"\n{title} — {header}")
        print(f"{'':>7}  {sep}")
        for i, ep in enumerate(epochs):
            print(f"{ep:>5} | {table['case_recall'][i]:.3f} | "
                  f"{table['case_fp_rate'][i]:.3f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_centroid_detection.json"
    payload = {
        "title": "Case-level centroid-based detection by saved milestone epoch",
        "description": ("Single preliminary milestones test run: seed 42, 625 images, "
                        "centroid detection = predicted centroid within 0.5x GT "
                        f"equivalent diameter, noise floor {MIN_PRED_AREA} px."),
        "noise_floor_px": MIN_PRED_AREA,
        "centroid_definition": "predicted centroid within 0.5x GT equivalent diameter",
        "dataset_snapshot_id": data.snapshot_id,
        "seed": 42,
        "images": 625,
        "epochs": epochs,
        "combined": {
            "Detection": combined_metrics["case_recall"],
            "False positive rate": combined_metrics["case_fp_rate"],
        },
        "malignant": {
            "Detection": metrics["case_recall"],
            "False positive rate": metrics["case_fp_rate"],
        },
        "benign": {
            "Detection": benign_metrics["case_recall"],
            "False positive rate": benign_metrics["case_fp_rate"],
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# Case-level centroid-based detection by saved milestone epoch",
        "",
        f"Single preliminary milestones test run: seed 42, 625 images, "
        f"centroid detection = predicted centroid within 0.5x GT equivalent "
        f"diameter, noise floor {MIN_PRED_AREA} px.",
        "",
        "A mass case is detected when the closest retained predicted centroid "
        "lies within half the ground-truth mass's equivalent circular diameter.",
        "",
        "Each table reports case-level false positives computed on normal cases "
        "only. A normal case with any prediction is a false alarm.",
        "",
        "No error bars (single seed).",
        "",
    ]

    for title, table in [("All masses", combined_metrics),
                         ("Malignant masses", metrics),
                         ("Benign masses", benign_metrics)]:
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append("| Epoch | Detection | Normal cases FP rate |")
        md_lines.append("|------:|----------:|-------------------:|")
        for i, ep in enumerate(epochs):
            md_lines.append(
                f"| {ep} | {table['case_recall'][i]:.3f} | "
                f"{table['case_fp_rate'][i]:.3f} |"
            )
        md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_centroid_detection.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for src, key, label, color in PLOT_SERIES:
        data = {
            "combined": combined_metrics,
            "malignant": metrics,
            "benign": benign_metrics,
        }[src][key]
        ax.plot(epochs, data, '-o', color=color, label=label,
                markersize=7, linewidth=1.5, markeredgecolor='white',
                markeredgewidth=1.5, zorder=3)

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
        out = OUT_DIR / f'epoch_convergence_by_centroid_detection.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
