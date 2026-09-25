"""False positives vs. training size for the data-scaling experiment.

Uses the canonical prediction dataset's normal_false_positive flags to report
case-level false-alarm rates on normal cases. Outputs terminal tables,
Markdown, JSON, PNG, and PDF for seeds 42/43/44 at best checkpoints.

Per-seed rates are plotted individually. The seed mean carries a 95%
percentile bootstrap CI over normal test cases, which reflects test-set
sampling uncertainty. Seeds vary initialization only; training subsets are
nested and fixed, so subset-selection variance is not captured.

Run from the project root:
    python analysis/data_scaling/evaluate_false_positives_best/evaluate_false_positives_best.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import (
    DatasetValidationError,
    load_predictions_dataset,
)

OUT_BASE = "evaluate_false_positives_best"
NOISE_FLOOR = f"{MIN_PRED_AREA_FRACTION * 100:g}% of image area"
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"
CHECKPOINT = "best"
SEEDS = [42, 43, 44]
SIZES = [5, 10, 20, 40, 80, 160, 320, 625]
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95
SEED_X_OFFSETS = [0.93, 1.0, 1.07]  # multiplicative jitter on the log x-axis
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


def normal_fp_flags(runs, size):
    """Return sorted normal image IDs and a seeds x cases 0/1 array of FP flags."""
    per_seed = [
        {row.image_id: row.normal_false_positive for row in runs[size, seed] if row.pathology == "normal"}
        for seed in SEEDS
    ]
    if not per_seed[0]:
        raise DatasetValidationError("Missing normal cases")
    image_ids = sorted(per_seed[0])
    flags = np.array([[int(seed_flags[image_id]) for image_id in image_ids] for seed_flags in per_seed])
    return image_ids, flags


def summarize_sizes(flag_matrices):
    counts_per_seed = []
    count_metrics = {"mean": [], "std": []}
    rate_metrics = {"mean": [], "std": []}
    for flags in flag_matrices:
        n_normal = flags.shape[1]
        counts = [int(value) for value in flags.sum(axis=1)]
        rates = [count / n_normal for count in counts]
        counts_per_seed.append(counts)
        for metric, values in ((count_metrics, counts), (rate_metrics, rates)):
            metric["mean"].append(mean(values))
            metric["std"].append(stdev(values))
    return counts_per_seed, count_metrics, rate_metrics


def bootstrap_ci(flag_matrices, n_normal):
    """Percentile CI for the seed-mean FP rate, resampling normal cases.

    The same case resamples are used for every seed and training size, so
    seeds stay paired and sizes are comparable replicate by replicate.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n_normal, size=(BOOTSTRAP_REPLICATES, n_normal))
    tail = (1 - CI_LEVEL) / 2
    lows, highs = [], []
    for flags in flag_matrices:
        replicate_rates = flags[:, indices].mean(axis=2).mean(axis=0)
        low, high = np.quantile(replicate_rates, [tail, 1 - tail])
        lows.append(float(low))
        highs.append(float(high))
    return lows, highs


def main():
    data = load_predictions_dataset()
    runs = select_runs(data.rows)
    image_ids = None
    flag_matrices = []
    for size in SIZES:
        size_ids, flags = normal_fp_flags(runs, size)
        if image_ids is None:
            image_ids = size_ids
        elif size_ids != image_ids:
            raise DatasetValidationError(f"Normal cases differ: size={size}")
        flag_matrices.append(flags)
    n_normal = len(image_ids)
    counts_per_seed, count_metrics, rate_metrics = summarize_sizes(flag_matrices)
    ci_low, ci_high = bootstrap_ci(flag_matrices, n_normal)
    rate_metrics["ci_low"] = ci_low
    rate_metrics["ci_high"] = ci_high

    ci_method = (
        f"{CI_LEVEL:.0%} percentile bootstrap over the {n_normal} normal test cases "
        f"({BOOTSTRAP_REPLICATES:,} replicates, RNG seed {BOOTSTRAP_SEED}), applied to "
        "the seed-mean FP rate. Case resamples are shared across seeds and sizes."
    )
    title = "False positives vs. training size"
    description = (
        "Data-scaling test runs: seeds 42/43/44, best checkpoints, "
        f"training sizes {', '.join(map(str, SIZES))}, noise floor {NOISE_FLOOR}. "
        "Seeds vary initialization only; training subsets are nested. "
        f"Uncertainty: {ci_method}"
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
        "ci_method": ci_method,
        "fp_count_per_seed": counts_per_seed,
        "fp_count": count_metrics,
        "fp_rate": rate_metrics,
        "normal_image_ids": image_ids,
        "fp_flags_per_size_seed_case": [flags.tolist() for flags in flag_matrices],
    }
    json_path = SCRIPT_DIR / f"{OUT_BASE}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        f"# {title}", "", description, "",
        f"Dataset snapshot: `{data.snapshot_id}`.", "",
        "A false positive is a Normal (mass-free) case the model flags as containing "
        "any retained mass. Every image is one patient with at most one mass, so "
        "this is a single case-level signal, not a count of predicted components.", "",
        f"Normal test cases: {n_normal}.", "",
        "| Training size | Seed 42 | Seed 43 | Seed 44 | Mean FP rate ± SD | 95% CI |",
        "|--------------:|:-------:|:-------:|:-------:|------------------:|-------:|",
    ]
    for index, size in enumerate(SIZES):
        counts = counts_per_seed[index]
        lines.append(
            f"| {size} | {counts[0]}/{n_normal} | {counts[1]}/{n_normal} | {counts[2]}/{n_normal} | "
            f"{rate_metrics['mean'][index]:.3f} ± {rate_metrics['std'][index]:.3f} | "
            f"{ci_low[index]:.3f}–{ci_high[index]:.3f} |"
        )
    lines.append("")
    report = "\n".join(lines)
    (SCRIPT_DIR / f"{OUT_BASE}.md").write_text(report, encoding="utf-8")
    print(report)

    sizes = np.array(SIZES, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(
        sizes, ci_low, ci_high, color=FP_RATE_COLOR, alpha=0.15, linewidth=0,
        label=f"{CI_LEVEL:.0%} CI (case bootstrap)", zorder=1,
    )
    ax.plot(
        sizes, rate_metrics["mean"], "-", color=FP_RATE_COLOR, linewidth=1.5,
        label="Seed mean", zorder=2,
    )
    for seed_index, offset in enumerate(SEED_X_OFFSETS):
        ax.scatter(
            sizes * offset,
            [counts[seed_index] / n_normal for counts in counts_per_seed],
            s=22, color=FP_RATE_COLOR, edgecolors="white", linewidths=0.8,
            label="Individual seeds" if seed_index == 0 else None, zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xticks(SIZES, labels=[str(size) for size in SIZES])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(SIZES[0] * 0.75, SIZES[-1] * 1.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training images (log scale)")
    ax.set_ylabel(f"False-positive rate (n = {n_normal} normal cases)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#c0c0c0")
    ax.tick_params(color="#c0c0c0", labelcolor="#999999")
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
