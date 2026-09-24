"""Centroid detection vs. training size for the data-scaling experiment.

Uses the canonical prediction dataset's retained-component centroid flags and
normal false-positive flags. Outputs terminal tables, Markdown, JSON, PNG, and
PDF with mean ± sample SD across seeds 42/43/44 at final checkpoints.

Run from the project root:
    python analysis/data_scaling/evaluate_centroid_detection/evaluate_centroid_detection_deq_050.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import (
    DatasetValidationError,
    load_predictions_dataset,
)

DEQ_FACTOR = 0.5
FLAG_FIELD = "centroid_detection_deq_050_flag"
OUT_BASE = "evaluate_centroid_detection_deq_050"
NOISE_FLOOR = f"{MIN_PRED_AREA_FRACTION * 100:g}% of image area"
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"
CHECKPOINT = "final"
SEEDS = [42, 43, 44]
SIZES = [5, 10, 20, 40, 80, 160, 320, 625]
GROUPS = [("All masses", "combined"), ("Malignant masses", "malignant"), ("Benign masses", "benign")]
PLOT_SERIES = [
    ("combined", "Combined centroid", "#2a78d6"),
    ("malignant", "Malignant centroid", "#eb6834"),
    ("benign", "Benign centroid", "#3b6d11"),
]

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


def select_runs(rows):
    runs = defaultdict(list)
    for row in rows:
        if (
            row.experiment_name == EXPERIMENT_NAME
            and row.evaluation_split == EVALUATION_SPLIT
            and row.checkpoint == CHECKPOINT
            and row.training_set_size in SIZES
            and row.seed in SEEDS
        ):
            runs[row.training_set_size, row.seed].append(row)

    reference_cases = None
    for size in SIZES:
        for seed in SEEDS:
            run = runs[size, seed]
            if not run:
                raise DatasetValidationError(f"Missing scaling run: size={size}, seed={seed}")
            if len({row.configuration_id for row in run}) != 1:
                raise DatasetValidationError(f"Multiple configurations: size={size}, seed={seed}")
            cases = {row.image_id: (row.pathology, row.mass_present) for row in run}
            if reference_cases is None:
                reference_cases = cases
            if cases != reference_cases:
                raise DatasetValidationError(f"Test cases differ: size={size}, seed={seed}")
            for row in run:
                if row.pathology in ("malignant", "benign"):
                    if not row.mass_present or getattr(row, FLAG_FIELD) is None:
                        raise DatasetValidationError(f"Invalid centroid case: {row.image_id}")
                elif row.pathology == "normal":
                    if row.mass_present or row.normal_false_positive is None:
                        raise DatasetValidationError(f"Invalid normal case: {row.image_id}")
                else:
                    raise DatasetValidationError(f"Unknown pathology: {row.pathology}")
    return runs


def evaluate_centroid(rows, positive_class):
    positives = [
        row for row in rows
        if row.mass_present and (positive_class == "combined" or row.pathology == positive_class)
    ]
    normal_cases = [row for row in rows if row.pathology == "normal"]
    if not positives or not normal_cases:
        raise DatasetValidationError(f"Missing positive or normal cases for {positive_class}")
    recall = mean(getattr(row, FLAG_FIELD) for row in positives)
    fp_rate = mean(row.normal_false_positive for row in normal_cases)
    return recall, fp_rate


def summarize_sizes(runs, positive_class):
    metrics = {
        "Detection": {"mean": [], "std": []},
        "False positive rate": {"mean": [], "std": []},
    }
    for size in SIZES:
        results = [evaluate_centroid(runs[size, seed], positive_class) for seed in SEEDS]
        for metric, values in zip(metrics.values(), zip(*results)):
            metric["mean"].append(mean(values))
            metric["std"].append(stdev(values))
    return metrics


def fmt(metric, index):
    return f"{metric['mean'][index]:.3f} ± {metric['std'][index]:.3f}"


def main():
    data = load_predictions_dataset()
    runs = select_runs(data.rows)
    stats = {group: summarize_sizes(runs, group) for _, group in GROUPS}
    title = f"Case-level centroid-based detection (deq={DEQ_FACTOR:g}) by training size"
    description = (
        "Data-scaling test runs: seeds 42/43/44, final checkpoints, "
        f"training sizes {', '.join(map(str, SIZES))}. Centroid detection = predicted "
        f"centroid within {DEQ_FACTOR:g}x GT equivalent diameter, noise floor "
        f"{NOISE_FLOOR}. Values are mean ± sample SD across seeds."
    )
    payload = {
        "title": title,
        "description": description,
        "noise_floor": NOISE_FLOOR,
        "centroid_definition": f"predicted centroid within {DEQ_FACTOR:g}x GT equivalent diameter",
        "deq_factor": DEQ_FACTOR,
        "dataset_snapshot_id": data.snapshot_id,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_split": EVALUATION_SPLIT,
        "checkpoint": CHECKPOINT,
        "seeds": SEEDS,
        "training_set_sizes": SIZES,
        **stats,
    }
    json_path = SCRIPT_DIR / f"{OUT_BASE}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        f"# {title}", "", description, "",
        f"Dataset snapshot: `{data.snapshot_id}`.", "",
        "A mass case is detected when any ground-truth mass component has a retained "
        f"predicted centroid within {DEQ_FACTOR:g}x its equivalent circular diameter "
        "(D = 2 × sqrt(area / π)).", "",
        "Each table reports case-level false positives computed on normal cases only. "
        "A normal case with any retained predicted mass component is a false alarm.", "",
        "Values are mean ± sample standard deviation across the three seeds.", "",
    ]
    for title, group in GROUPS:
        metrics = stats[group]
        lines.extend([
            f"## {title}", "",
            "| Training size | Detection | Normal cases FP rate |",
            "|--------------:|----------:|---------------------:|",
        ])
        for index, size in enumerate(SIZES):
            lines.append(
                f"| {size} | {fmt(metrics['Detection'], index)} | "
                f"{fmt(metrics['False positive rate'], index)} |"
            )
        lines.append("")
    report = "\n".join(lines)
    (SCRIPT_DIR / f"{OUT_BASE}.md").write_text(report, encoding="utf-8")
    print(report)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = list(range(len(SIZES)))
    for group, label, color in PLOT_SERIES:
        metric = stats[group]["Detection"]
        ax.errorbar(
            positions, metric["mean"], yerr=metric["std"], fmt="-o", color=color,
            label=label, markersize=7, linewidth=1.5, capsize=3,
            markeredgecolor="white", markeredgewidth=1.5, zorder=3,
        )
    ax.set_xlabel("Training images")
    ax.set_ylabel("Centroid recall")
    ax.set_xticks(positions, labels=SIZES)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([value / 10 for value in range(11)])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#c0c0c0")
    ax.tick_params(axis="both", which="both", color="#c0c0c0", labelcolor="#999999")
    ax.legend(loc="best", frameon=True, fancybox=False, borderpad=0.8, handlelength=2.5)
    fig.tight_layout(pad=1.2)
    for extension in ("pdf", "png"):
        output_path = SCRIPT_DIR / f"{OUT_BASE}.{extension}"
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(output_path, dpi=300, bbox_inches="tight", metadata=metadata)
        print(f"Wrote {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
