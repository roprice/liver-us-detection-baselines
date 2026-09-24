"""False positives vs. training size for the data-scaling experiment.

Uses the canonical prediction dataset's normal_false_positive flags to report
case-level false-alarm counts and rates on normal cases. Outputs terminal tables,
Markdown, JSON, PNG, and PDF with mean ± sample SD across seeds 42/43/44 at
final checkpoints.

Run from the project root:
    python analysis/data_scaling/evaluate_false_positives/evaluate_false_positives.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import (
    DatasetValidationError,
    load_predictions_dataset,
)

OUT_BASE = "evaluate_false_positives"
NOISE_FLOOR = f"{MIN_PRED_AREA_FRACTION * 100:g}% of image area"
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"
CHECKPOINT = "final"
SEEDS = [42, 43, 44]
SIZES = [5, 10, 20, 40, 80, 160, 320, 625]
FP_COUNT_COLOR = "#eb6834"
FP_RATE_COLOR = "#2a78d6"

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
            if len(cases) != len(run):
                raise DatasetValidationError(f"Duplicate test cases: size={size}, seed={seed}")
            if set(row.pathology for row in run) != {"malignant", "benign", "normal"}:
                raise DatasetValidationError(f"Missing or unknown pathology: size={size}, seed={seed}")
            if reference_cases is None:
                reference_cases = cases
            if cases != reference_cases:
                raise DatasetValidationError(f"Test cases differ: size={size}, seed={seed}")
            for row in run:
                if row.pathology in ("malignant", "benign"):
                    if not row.mass_present:
                        raise DatasetValidationError(f"Invalid mass case: {row.image_id}")
                elif row.pathology == "normal":
                    if (
                        row.mass_present
                        or not isinstance(row.normal_false_positive, bool)
                        or row.normal_false_positive != row.triage_detection_flag
                    ):
                        raise DatasetValidationError(f"Invalid normal case: {row.image_id}")
                else:
                    raise DatasetValidationError(f"Unknown pathology: {row.pathology}")
    return runs


def count_false_positives(rows):
    normal_cases = [row for row in rows if row.pathology == "normal"]
    if not normal_cases:
        raise DatasetValidationError("Missing normal cases")
    return sum(row.normal_false_positive for row in normal_cases), len(normal_cases)


def summarize_sizes(runs):
    counts_per_seed = []
    count_metrics = {"mean": [], "std": []}
    rate_metrics = {"mean": [], "std": []}
    for size in SIZES:
        results = [count_false_positives(runs[size, seed]) for seed in SEEDS]
        counts = [count for count, _ in results]
        rates = [count / n_normal for count, n_normal in results]
        counts_per_seed.append(counts)
        for metric, values in ((count_metrics, counts), (rate_metrics, rates)):
            metric["mean"].append(mean(values))
            metric["std"].append(stdev(values))
    return counts_per_seed, count_metrics, rate_metrics


def main():
    data = load_predictions_dataset()
    runs = select_runs(data.rows)
    counts_per_seed, count_metrics, rate_metrics = summarize_sizes(runs)
    _, n_normal = count_false_positives(runs[SIZES[0], SEEDS[0]])
    title = "False positives vs. training size"
    description = (
        "Data-scaling test runs: seeds 42/43/44, final checkpoints, "
        f"training sizes {', '.join(map(str, SIZES))}, noise floor {NOISE_FLOOR}. "
        "Values are mean ± sample SD across seeds."
    )
    payload = {
        "title": title,
        "description": description,
        "noise_floor": NOISE_FLOOR,
        "seeds": SEEDS,
        "training_set_sizes": SIZES,
        "normal_cases": n_normal,
        "dataset_snapshot_id": data.snapshot_id,
        "experiment_name": EXPERIMENT_NAME,
        "evaluation_split": EVALUATION_SPLIT,
        "checkpoint": CHECKPOINT,
        "fp_count_per_seed": counts_per_seed,
        "fp_count": count_metrics,
        "fp_rate": rate_metrics,
    }
    json_path = SCRIPT_DIR / f"{OUT_BASE}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        f"# {title}", "", description, "",
        f"Dataset snapshot: `{data.snapshot_id}`.", "",
        "A false positive is a Normal (mass-free) case the model flags as containing "
        "any retained mass. Every image is one patient with at most one mass, so "
        "this is a single case-level signal, not a count of predicted components.", "",
        "Values are mean ± sample standard deviation across the three seeds.", "",
        f"Normal test cases: {n_normal}.", "",
        "| Training size | Seed 42 | Seed 43 | Seed 44 | Mean FP rate | Mean FP count |",
        "|--------------:|:-------:|:-------:|:-------:|-------------:|--------------:|",
    ]
    for index, size in enumerate(SIZES):
        counts = counts_per_seed[index]
        lines.append(
            f"| {size} | {counts[0]}/{n_normal} | {counts[1]}/{n_normal} | {counts[2]}/{n_normal} | "
            f"{rate_metrics['mean'][index]:.4f} ± {rate_metrics['std'][index]:.4f} | "
            f"{count_metrics['mean'][index]:.2f} ± {count_metrics['std'][index]:.2f} |"
        )
    lines.append("")
    report = "\n".join(lines)
    (SCRIPT_DIR / f"{OUT_BASE}.md").write_text(report, encoding="utf-8")
    print(report)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = list(range(len(SIZES)))
    ax.errorbar(
        positions, count_metrics["mean"], yerr=count_metrics["std"], fmt="-o",
        color=FP_COUNT_COLOR, label="False positives", markersize=7,
        linewidth=1.5, capsize=3, markeredgecolor="white",
        markeredgewidth=1.5, zorder=3,
    )
    ax.set_xlabel("Training images")
    ax.set_ylabel("False positives (count)", color=FP_COUNT_COLOR)
    ax.set_xticks(positions, labels=SIZES)
    ax.tick_params(axis="y", labelcolor=FP_COUNT_COLOR)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.grid(False, axis="x")

    ax2 = ax.twinx()
    ax2.errorbar(
        positions, rate_metrics["mean"], yerr=rate_metrics["std"], fmt="-o",
        color=FP_RATE_COLOR, label="FP rate", markersize=7,
        linewidth=1.5, capsize=3, markeredgecolor="white",
        markeredgewidth=1.5, zorder=3,
    )
    ax2.set_ylabel("False-positive rate", color=FP_RATE_COLOR)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="y", labelcolor=FP_RATE_COLOR)
    ax2.grid(False)

    for spine in list(ax.spines.values()) + list(ax2.spines.values()):
        spine.set_color("#c0c0c0")
    ax.tick_params(axis="x", color="#c0c0c0", labelcolor="#999999")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="best", frameon=True,
              fancybox=False, borderpad=0.8, handlelength=2.5)
    fig.tight_layout(pad=1.2)
    for extension in ("pdf", "png"):
        output_path = SCRIPT_DIR / f"{OUT_BASE}.{extension}"
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(output_path, dpi=300, bbox_inches="tight", metadata=metadata)
        print(f"Wrote {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
