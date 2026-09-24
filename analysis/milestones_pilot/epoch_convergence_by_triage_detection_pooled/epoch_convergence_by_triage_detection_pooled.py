"""Pooled malignant triage detection by epoch across test and fold-0 validation.

For each seed and saved epoch, this analysis pools malignant cases from the
milestones pilot test cohort and the milestones pilot validation cohort before
calculating recall. It then reports the mean and sample standard deviation across
seeds.

Run from the project root:
    python3 analysis/epoch_convergence/epoch_convergence_by_triage_detection_pooled/epoch_convergence_by_triage_detection_pooled.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

NOISE_FLOOR = "0.03% of image area"
SEEDS = [42, 43, 44]
EPOCHS = [150, 300, 750]
COHORTS = (
    ("milestones_pilot", "test", "predictions_milestones_625images"),
    (
        "milestones_pilot_vals",
        "validation_fold_0",
        "predictions_milestones_val_625images",
    ),
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size": 10,
    "font.weight": "400",
    "axes.labelsize": 10,
    "axes.labelcolor": "#999999",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    "legend.fontsize": 10,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#e0e0e0",
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": True,
    "axes.spines.right": True,
})


def pooled_rows(data, seed, epoch):
    """Return malignant rows pooled across the approved cohorts for one seed/epoch."""
    rows = []
    for experiment_name, evaluation_split, configuration_prefix in COHORTS:
        configuration_id = f"{configuration_prefix}_seed{seed}_epoch{epoch}"
        cohort_rows = [
            row for row in data.rows
            if row.experiment_name == experiment_name
            and row.evaluation_split == evaluation_split
            and row.configuration_id == configuration_id
            and row.pathology == "malignant"
        ]
        if not cohort_rows:
            raise RuntimeError(
                f"No malignant rows for {experiment_name}/{evaluation_split}, "
                f"seed {seed}, epoch {epoch}"
            )
        rows.extend(cohort_rows)
    return rows


def summarize_epochs(data):
    """Calculate pooled malignant triage recall and counts for each epoch."""
    summaries = []
    for epoch in EPOCHS:
        per_seed = []
        recalls = []
        for seed in SEEDS:
            rows = pooled_rows(data, seed, epoch)
            detected = sum(row.triage_detection_flag for row in rows)
            total = len(rows)
            recall = detected / total
            per_seed.append({
                "seed": seed,
                "detected": detected,
                "total": total,
                "recall": recall,
            })
            recalls.append(recall)
        summaries.append({
            "epoch": epoch,
            "per_seed": per_seed,
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls, ddof=1)),
        })
    return summaries


def main():
    data = load_predictions_dataset()
    summaries = summarize_epochs(data)

    print(
        f"Loaded predictions dataset: {data.snapshot_id}; pooled malignant triage "
        "recall across test and fold-0 validation"
    )
    print("\nEpoch | Seed 42 | Seed 43 | Seed 44 | Mean ± SD")
    print("------|---------|---------|---------|----------")
    for summary in summaries:
        counts = " | ".join(
            f"{entry['detected']}/{entry['total']}"
            for entry in summary["per_seed"]
        )
        print(
            f"{summary['epoch']:>5} | {counts} | "
            f"{summary['mean']:.3f} ± {summary['std']:.3f}"
        )

    payload = {
        "title": "Pooled malignant triage detection by saved milestone epoch",
        "description": (
            "Malignant triage recall pooled within each seed across the milestones "
            "pilot test and fold-0 validation cohorts. Values are mean ± sample "
            f"standard deviation across seeds; noise floor {NOISE_FLOOR}."
        ),
        "dataset_snapshot_id": data.snapshot_id,
        "cohorts": [
            {
                "experiment_name": experiment_name,
                "evaluation_split": evaluation_split,
            }
            for experiment_name, evaluation_split, _ in COHORTS
        ],
        "triage_definition": "any retained mass (no overlap required)",
        "noise_floor": NOISE_FLOOR,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "results": summaries,
    }
    json_path = SCRIPT_DIR / "epoch_convergence_by_triage_detection_pooled.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {json_path}")

    markdown_lines = [
        "# Pooled malignant triage detection by saved milestone epoch",
        "",
        "Malignant triage recall is pooled within each seed across the "
        "`milestones_pilot` test cohort and the `milestones_pilot_vals` fold-0 "
        "validation cohort. A malignant case is detected if the model predicts "
        "any retained mass. Values are mean ± sample standard deviation across "
        f"the three seeds; noise floor {NOISE_FLOOR}.",
        "",
        "| Epoch | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |",
        "|------:|--------:|--------:|--------:|----------:|",
    ]
    for summary in summaries:
        counts = " | ".join(
            f"{entry['detected']}/{entry['total']}"
            for entry in summary["per_seed"]
        )
        markdown_lines.append(
            f"| {summary['epoch']} | {counts} | "
            f"{summary['mean']:.3f} ± {summary['std']:.3f} |"
        )
    markdown_lines.extend([
        "",
        "*Each seed pools 65 malignant test cases and 78 malignant fold-0 "
        "validation cases (143 total).*",
        "",
    ])
    markdown_path = SCRIPT_DIR / "epoch_convergence_by_triage_detection_pooled.md"
    markdown_path.write_text("\n".join(markdown_lines))
    print(f"Wrote {markdown_path}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = [summary["mean"] for summary in summaries]
    stds = [summary["std"] for summary in summaries]
    ax.errorbar(
        EPOCHS,
        means,
        yerr=stds,
        fmt="-o",
        color="#eb6834",
        label="Pooled malignant triage",
        markersize=7,
        linewidth=1.5,
        capsize=3,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triage recall")
    ax.set_xlim(left=0, right=EPOCHS[-1])
    ax.set_xticks(EPOCHS)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#c0c0c0")
    ax.tick_params(axis="both", which="both", color="#c0c0c0", labelcolor="#999999")
    ax.legend(loc="best", frameon=True, fancybox=False, borderpad=0.8)
    plt.tight_layout(pad=1.2)
    for extension in ("pdf", "png"):
        output_path = SCRIPT_DIR / (
            f"epoch_convergence_by_triage_detection_pooled.{extension}"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight", metadata={"CreationDate": None})
        print(f"Wrote {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
