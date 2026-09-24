"""Case-level overlap detection (IoU > 0.2) by training size and mass area.

Uses canonical data-scaling test predictions at final checkpoints. Outputs
terminal tables, Markdown, JSON, PNG, and PDF with mean ± sample SD across seeds.

Run from the project root:
    python analysis/data_scaling/evaluate_mass_size_vs_detection/evaluate_mass_size_vs_detection_iou_02.py
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

FLAG_FIELD = "overlap_detection_iou_02_flag"
NORMAL_FLAG_FIELD = "normal_false_positive"
OUT_BASE = "evaluate_mass_size_vs_detection_iou_02"
IOU_THRESHOLD = 0.2
TITLE = f"Case-level overlap-based detection (IoU>{IOU_THRESHOLD:g}) by training size and mass size"
DEFINITION = (
    "A mass case is detected when the highest single-component IoU between a "
    f"retained predicted component and the ground-truth mass exceeds {IOU_THRESHOLD:g}. "
    "Off-target blobs are ignored, so this is comparable to centroid detection."
)
DEFINITION_METADATA = {
    "detection_definition": f"IoU > {IOU_THRESHOLD:g}",
    "iou_threshold": IOU_THRESHOLD,
}
DETECTION_LABEL = "Overlap Detection"
Y_LABEL = "Combined overlap recall"
NOISE_FLOOR = f"{MIN_PRED_AREA_FRACTION * 100:g}% of image area"
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"
CHECKPOINT = "final"
SEEDS = [42, 43, 44]
SIZES = [5, 10, 20, 40, 80, 160, 320, 625]
GROUPS = [("All masses", "combined"), ("Malignant masses", "malignant"), ("Benign masses", "benign")]
MASS_BINS = [
    ("0–3,000 px²", 0, 3001),
    (">3,000–10,000 px²", 3001, 10001),
    (">10,000–30,000 px²", 10001, 30001),
    (">30,000 px²", 30001, None),
]
BIN_COLORS = ["#2a78d6", "#eb6834", "#3b6d11", "#7B5BD6"]

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
            cases = {
                row.image_id: (
                    row.pathology, row.mass_present, row.ground_truth_mass_area_px,
                    row.image_height_px, row.image_width_px, row.reference_path,
                    row.original_data_path, row.evaluation_unit,
                )
                for row in run
            }
            if len(cases) != len(run):
                raise DatasetValidationError(f"Duplicate cases: size={size}, seed={seed}")
            if reference_cases is None:
                reference_cases = cases
            if cases != reference_cases:
                raise DatasetValidationError(f"Test cases differ: size={size}, seed={seed}")
            for row in run:
                if row.pathology in ("malignant", "benign"):
                    if (not row.mass_present or row.ground_truth_mass_area_px <= 0
                            or not isinstance(getattr(row, FLAG_FIELD), bool)):
                        raise DatasetValidationError(f"Invalid mass case: {row.image_id}")
                elif row.pathology == "normal":
                    if (row.mass_present or row.ground_truth_mass_area_px != 0
                            or not isinstance(getattr(row, NORMAL_FLAG_FIELD), bool)):
                        raise DatasetValidationError(f"Invalid normal case: {row.image_id}")
                else:
                    raise DatasetValidationError(f"Unknown pathology: {row.pathology}")
    return runs


def mass_cases_in_bin(rows, positive_class, low, high):
    return [
        row for row in rows
        if row.mass_present
        and (positive_class == "combined" or row.pathology == positive_class)
        and row.ground_truth_mass_area_px >= low
        and (high is None or row.ground_truth_mass_area_px < high)
    ]


def evaluate_bin(rows, positive_class, low, high):
    positives = mass_cases_in_bin(rows, positive_class, low, high)
    normal_cases = [row for row in rows if row.pathology == "normal"]
    if not positives or not normal_cases:
        raise DatasetValidationError(
            f"Missing mass-bin or normal cases: group={positive_class}, bin=[{low}, {high})"
        )
    return (
        mean(getattr(row, FLAG_FIELD) for row in positives),
        mean(getattr(row, NORMAL_FLAG_FIELD) for row in normal_cases),
    )


def summarize_sizes_by_bin(runs, positive_class):
    result = {}
    for label, low, high in MASS_BINS:
        metrics = {
            "Detection": {"mean": [], "std": []},
            "False positive rate": {"mean": [], "std": []},
        }
        for size in SIZES:
            values = [evaluate_bin(runs[size, seed], positive_class, low, high) for seed in SEEDS]
            for metric, per_seed in zip(metrics.values(), zip(*values)):
                metric["mean"].append(mean(per_seed))
                metric["std"].append(stdev(per_seed))
        result[label] = metrics
    return result


def fmt(metric, index):
    return f"{metric['mean'][index]:.3f} ± {metric['std'][index]:.3f}"


def main():
    data = load_predictions_dataset()
    runs = select_runs(data.rows)
    stats = {group: summarize_sizes_by_bin(runs, group) for _, group in GROUPS}
    reference_rows = runs[SIZES[0], SEEDS[0]]
    bin_counts = {
        group: {
            label: len(mass_cases_in_bin(reference_rows, group, low, high))
            for label, low, high in MASS_BINS
        }
        for _, group in GROUPS
    }
    description = (
        "Data-scaling test runs: seeds 42/43/44, final checkpoints, "
        f"training sizes {', '.join(map(str, SIZES))}. Noise floor {NOISE_FLOOR}. "
        "Mass-present cases split by ground_truth_mass_area_px into four bins. "
        "Values are mean ± sample SD across seeds."
    )
    payload = {
        "title": TITLE,
        "description": description,
        "noise_floor": NOISE_FLOOR,
        **DEFINITION_METADATA,
        "mass_bins": [{"label": label, "low": low, "high": high} for label, low, high in MASS_BINS],
        "bin_counts": bin_counts,
        "dataset_snapshot_id": data.snapshot_id,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_split": EVALUATION_SPLIT,
        "checkpoint": CHECKPOINT,
        "seeds": SEEDS,
        "training_set_sizes": SIZES,
        **stats,
    }
    (SCRIPT_DIR / f"{OUT_BASE}.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {TITLE}", "", description, "",
        f"Dataset snapshot: `{data.snapshot_id}`.", "", DEFINITION, "",
        "Area-bin lower bounds are inclusive and upper bounds exclusive: "
        "[0, 3001), [3001, 10001), [10001, 30001), [30001, ∞) pixels².", "",
        "False-positive rates use all normal cases, not mass-size bins, and are "
        "identical across bins and pathology groups at each training size. "
        "A normal case with any retained predicted mass component is a false alarm.", "",
        "The plot shows combined recall for each mass-size bin; error bars are "
        "sample SD across the three seeds. Training sizes are equally spaced.", "",
    ]
    for title, group in GROUPS:
        lines.extend([f"## {title}", ""])
        for label, _, _ in MASS_BINS:
            metrics = stats[group][label]
            lines.extend([
                f"### {label} (n={bin_counts[group][label]})", "",
                f"| Training size | {DETECTION_LABEL} | Normal cases FP rate |",
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
    for (label, _, _), color in zip(MASS_BINS, BIN_COLORS):
        metric = stats["combined"][label]["Detection"]
        ax.errorbar(
            positions, metric["mean"], yerr=metric["std"], fmt="-o", color=color,
            label=f"{label} (n={bin_counts['combined'][label]})",
            markersize=7, linewidth=1.5, capsize=3,
            markeredgecolor="white", markeredgewidth=1.5, zorder=3,
        )
    ax.set_xlabel("Training images")
    ax.set_ylabel(Y_LABEL)
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
