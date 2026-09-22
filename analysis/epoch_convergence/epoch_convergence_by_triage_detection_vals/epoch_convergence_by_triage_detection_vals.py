"""Triage detection vs. epoch on fold-0 validation data (3 seeds, 125 images each).

Triage is the most permissive detection view: a case is "detected" if the model
predicts any retained mass anywhere on the image, whether or not it overlaps the
true mass. On this dataset every image is one patient with at most one mass, so
triage is a single case-level binary:

  - Mass-present case (malignant/benign): flagged -> detected, not flagged -> miss.
  - Normal case (no mass): flagged -> false alarm, not flagged -> correct.

Because there is no overlap requirement, an IoU threshold does not apply. The
only filter is the relative noise floor, already baked into the
triage_detection_flag column of the predictions dataset.

Reports, per milestone checkpoint and per mass grouping (combined / malignant /
benign): case-level recall and false-positive rate over Normal cases, averaged
across the three seeds with standard deviation.

Reads the canonical predictions dataset (analysis/predictions_dataset) rather
than the raw masks; triage_detection_flag and pathology come from there.

Three seeds (42/43/44), 125 fold-0 validation images each; error bars are the
across-seed std.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_triage_detection_vals/epoch_convergence_by_triage_detection_vals.py
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

EXPERIMENT_NAME = "milestones_pilot_vals"
EVALUATION_SPLIT = "validation_fold_0"
SEEDS = [42, 43, 44]
EPOCHS = [150, 300, 750]


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
    ("combined",  "case_recall", "Combined triage",  "#2a78d6"),
    ("malignant", "case_recall", "Malignant triage", "#eb6834"),
    ("benign",    "case_recall", "Benign triage",    "#3b6d11"),
]


def evaluate_triage(rows, positive_class):
    """Return (tp, fp, fn, n_normal) at case level for triage detection."""
    tp = fn = 0
    fp = 0
    n_normal = 0

    for row in rows:
        flagged = row.triage_detection_flag
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
            if flagged:
                fp += 1

    return tp, fp, fn, n_normal


def evaluate_epoch(rows, positive_class="malignant"):
    """Run triage eval for one checkpoint, return case recall and FP rate."""
    ctp, cfp, cfn, n_normal = evaluate_triage(rows, positive_class)

    crec = ctp / (ctp + cfn) if (ctp + cfn) > 0 else float('nan')
    cfp_rate = cfp / n_normal if n_normal > 0 else float('nan')

    return crec, cfp_rate



def summarize_epochs(rows, positive_class):
    """Return {metric: (means, stds)} across seeds, per epoch."""
    means = {key: [] for key in METRICS}
    stds = {key: [] for key in METRICS}
    for epoch in EPOCHS:
        recalls = []
        fp_rates = []
        for seed in SEEDS:
            folder = f"predictions_milestones_val_625images_seed{seed}_epoch{epoch}"
            configuration_rows = [r for r in rows if r.configuration_id == folder]
            if not configuration_rows:
                print(f"WARNING: no rows for {folder}")
                continue
            rec, fp = evaluate_epoch(configuration_rows, positive_class)
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
    rows = [
        row for row in data.rows
        if row.experiment_name == EXPERIMENT_NAME
        and row.evaluation_split == EVALUATION_SPLIT
    ]
    print(
        f"Loaded predictions dataset: {data.snapshot_id}, {len(rows)} rows "
        f"for {EXPERIMENT_NAME}/{EVALUATION_SPLIT}"
    )

    stats = {
        "combined": summarize_epochs(rows, "combined"),
        "malignant": summarize_epochs(rows, "malignant"),
        "benign": summarize_epochs(rows, "benign"),
    }


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


    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_triage_detection_vals.json"
    payload = {
        "title": "Case-level triage-based detection by saved milestone epoch on fold-0 validation data",
        "description": ("Three preliminary milestones validation runs: seeds 42/43/44, "
                        "125 fold-0 validation images each, triage = any retained mass "
                        f"(no overlap required), noise floor {NOISE_FLOOR}. Values are "
                        "mean ± std across seeds."),
        "noise_floor": NOISE_FLOOR,
        "triage_definition": "any retained mass (no overlap required)",
        "dataset_snapshot_id": data.snapshot_id,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_split": EVALUATION_SPLIT,
        "seeds": SEEDS,
        "images": 125,
        "epochs": epochs,
        "combined": metric_payload(stats["combined"][0], stats["combined"][1]),
        "malignant": metric_payload(stats["malignant"][0], stats["malignant"][1]),
        "benign": metric_payload(stats["benign"][0], stats["benign"][1]),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# Case-level triage-based detection by saved milestone epoch on fold-0 validation data",
        "",
        f"Three preliminary milestones validation runs: seeds 42/43/44, 125 fold-0 "
        f"validation images each, triage = any retained mass (no overlap required), "
        f"noise floor {NOISE_FLOOR}.",
        "",
        "A mass-present case is detected if the model predicts any retained mass "
        "anywhere on the image.",
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
        md_lines.append("**Saved checkpoints**")
        md_lines.append("| Checkpoint | Detection | Normal cases FP rate |")
        md_lines.append("|------:|----------:|-------------------:|")
        for i, ep in enumerate(epochs):
            md_lines.append(
                f"| {ep} | {fmt(means['case_recall'][i], stds['case_recall'][i])} | "
                f"{fmt(means['case_fp_rate'][i], stds['case_fp_rate'][i])} |"
            )
        md_lines.append("")

    md_path = OUT_DIR / "epoch_convergence_by_triage_detection_vals.md"
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
    ax.set_ylabel('Triage recall')
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
        out = OUT_DIR / f'epoch_convergence_by_triage_detection_vals.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
