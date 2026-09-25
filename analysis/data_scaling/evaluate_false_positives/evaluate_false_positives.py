"""False positives vs. training size for the data-scaling experiment.

Reports retained-mask normal-case FP rate, no-overlap FP regions per image,
and six criterion-specific FP region counts per image across all test images.
Every retained region on a normal image counts for each region metric.

Outputs terminal tables, Markdown, JSON, and PNG/PDF charts, for
seeds 42/43/44 at final checkpoints. Per-seed values are plotted individually.
Seed means carry a 95% percentile bootstrap CI over test images, reflecting
test-set sampling uncertainty. Case-level rates also get per-seed exact
binomial intervals. Seeds vary initialization only; training subsets are
nested and fixed, so subset-selection variance is not captured.

Run from the project root:
    python analysis/data_scaling/evaluate_false_positives/evaluate_false_positives.py
"""

import json
import math
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

OUT_BASE = "evaluate_false_positives"
CASE_FIGURE_BASE = "evaluate_case_level_false_positives"
LESION_FIGURE_BASE = "evaluate_lesion_level_false_positives"
CRITERIA = {
    "overlap_detection_iou_00_fp_count": "IoU > 0",
    "overlap_detection_iou_02_fp_count": "IoU > 0.2",
    "overlap_detection_iou_05_fp_count": "IoU > 0.5",
    "centroid_detection_deq_025_fp_count": "d_eq ≤ 0.25",
    "centroid_detection_deq_050_fp_count": "d_eq ≤ 0.50",
    "centroid_detection_deq_100_fp_count": "d_eq ≤ 1.00",
}
CRITERION_GROUPS = {
    "Overlap detection": list(CRITERIA)[:3],
    "Centroid detection": list(CRITERIA)[3:],
}
CRITERION_COLORS = ["#2a78d6", "#dc7435", "#35966a"]
NOISE_FLOOR = f"{MIN_PRED_AREA_FRACTION * 100:g}% of image area"
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"
CHECKPOINT = "final"
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
                total = row.retained_component_count
                unmatched = row.unmatched_retained_component_count
                if (
                    not isinstance(total, int)
                    or not isinstance(unmatched, int)
                    or not 0 <= unmatched <= total
                    or (total > 0) != row.triage_detection_flag
                    or (row.pathology == "normal" and unmatched != total)
                ):
                    raise DatasetValidationError(f"Invalid component counts: {row.image_id}")
                for field in CRITERIA:
                    count = getattr(row, field)
                    if not isinstance(count, int) or not 0 <= count <= total:
                        raise DatasetValidationError(f"Invalid {field}: {row.image_id}")
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


def region_counts(runs, size, field):
    """Return sorted image IDs, seeds x images region counts, and a normal mask."""
    per_seed = [
        {row.image_id: getattr(row, field) for row in runs[size, seed]}
        for seed in SEEDS
    ]
    pathology = {row.image_id: row.pathology for row in runs[size, SEEDS[0]]}
    image_ids = sorted(per_seed[0])
    counts = np.array([[seed_counts[image_id] for image_id in image_ids] for seed_counts in per_seed])
    is_normal = np.array([pathology[image_id] == "normal" for image_id in image_ids])
    return image_ids, counts, is_normal


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


def summarize_regions(matrices, n_images):
    per_seed = [[float(value) for value in matrix.mean(axis=1)] for matrix in matrices]
    ci_low, ci_high = bootstrap_ci(matrices, n_images)
    return {
        "regions_per_seed": [[int(value) for value in matrix.sum(axis=1)] for matrix in matrices],
        "per_image_per_seed": per_seed,
        "per_image": {
            "mean": [mean(values) for values in per_seed],
            "std": [stdev(values) for values in per_seed],
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
    }


def bootstrap_ci(value_matrices, n_images):
    """Percentile CI for the seed mean of a per-image value, resampling images.

    Each matrix is seeds x images. The same image resamples are used for every
    seed and training size, so seeds stay paired and sizes are comparable
    replicate by replicate.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, n_images, size=(BOOTSTRAP_REPLICATES, n_images))
    tail = (1 - CI_LEVEL) / 2
    lows, highs = [], []
    for values in value_matrices:
        replicate_rates = values[:, indices].mean(axis=2).mean(axis=0)
        low, high = np.quantile(replicate_rates, [tail, 1 - tail])
        lows.append(float(low))
        highs.append(float(high))
    return lows, highs


def plot_by_size(per_seed, seed_mean, ci_low, ci_high, ylabel, ylim_top, out_base, title):
    """Seed dots, seed-mean line, and bootstrap CI band on a log x-axis."""
    sizes = np.array(SIZES, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(
        sizes, ci_low, ci_high, color=FP_RATE_COLOR, alpha=0.15, linewidth=0,
        label=f"{CI_LEVEL:.0%} CI (image bootstrap)", zorder=1,
    )
    ax.plot(sizes, seed_mean, "-", color=FP_RATE_COLOR, linewidth=1.5, label="Seed mean", zorder=2)
    for seed_index, offset in enumerate(SEED_X_OFFSETS):
        ax.scatter(
            sizes * offset, [values[seed_index] for values in per_seed],
            s=22, color=FP_RATE_COLOR, edgecolors="white", linewidths=0.8,
            label="Individual seeds" if seed_index == 0 else None, zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xticks(SIZES, labels=[str(size) for size in SIZES])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(SIZES[0] * 0.75, SIZES[-1] * 1.3)
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel("Training images (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#c0c0c0")
    ax.tick_params(color="#c0c0c0", labelcolor="#999999")
    ax.legend(loc="best", frameon=True, fancybox=False, borderpad=0.8, handlelength=2.5)
    fig.tight_layout(pad=1.2)
    for extension in ("pdf", "png"):
        output_path = SCRIPT_DIR / f"{out_base}.{extension}"
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(output_path, dpi=300, bbox_inches="tight", metadata=metadata)
        print(f"Wrote {output_path}")
    plt.close(fig)


def plot_criteria_by_size(criteria, summaries, title, out_base):
    """Plot three criteria with seed means, image-bootstrap bands, and seed dots."""
    sizes = np.array(SIZES, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for color, field in zip(CRITERION_COLORS, criteria):
        summary = summaries[field]
        metrics = summary["per_image"]
        ax.fill_between(sizes, metrics["ci_low"], metrics["ci_high"],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(sizes, metrics["mean"], color=color, linewidth=1.6,
                label=summary["criterion"])
        for seed_index, offset in enumerate(SEED_X_OFFSETS):
            ax.scatter(sizes * offset,
                       [values[seed_index] for values in summary["per_image_per_seed"]],
                       s=13, color=color, alpha=0.7, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, labels=[str(size) for size in SIZES])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(SIZES[0] * 0.75, SIZES[-1] * 1.3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Training images (log scale)")
    ax.set_ylabel("FP regions per image (all test images)")
    ax.set_title(f"{title}: FP regions per image")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color("#c0c0c0")
    ax.tick_params(color="#c0c0c0", labelcolor="#999999")
    ax.legend(title="Criterion (lines: seed mean; bands: 95% bootstrap CI; dots: seeds)",
              loc="best", frameon=True, fancybox=False)
    fig.tight_layout(pad=1.2)
    for extension in ("pdf", "png"):
        output_path = SCRIPT_DIR / f"{out_base}.{extension}"
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(output_path, dpi=300, bbox_inches="tight", metadata=metadata)
        print(f"Wrote {output_path}")
    plt.close(fig)


def binomial_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, level=CI_LEVEL):
    """Exact binomial CI for k successes in n trials, solved by bisection."""
    tail = (1 - level) / 2

    def solve(predicate):
        low, high = 0.0, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            if predicate(mid):
                high = mid
            else:
                low = mid
        return (low + high) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - binomial_cdf(k - 1, n, p) >= tail)
    upper = 1.0 if k == n else solve(lambda p: binomial_cdf(k, n, p) <= tail)
    return lower, upper


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
    rate_metrics["label"] = "Retained-mask normal-case FP rate"
    exact_ci_per_seed = [
        [list(clopper_pearson(count, n_normal)) for count in counts]
        for counts in counts_per_seed
    ]

    all_image_ids = None
    region_matrices = []
    for size in SIZES:
        size_ids, counts, is_normal = region_counts(runs, size, "unmatched_retained_component_count")
        if all_image_ids is None:
            all_image_ids = size_ids
        elif size_ids != all_image_ids:
            raise DatasetValidationError(f"Test images differ: size={size}")
        region_matrices.append(counts)
    n_images = len(all_image_ids)
    lesion = summarize_regions(region_matrices, n_images)
    lesion["per_image"]["label"] = "No-overlap FP regions per image"
    lesion["per_normal_image_mean"] = [float(m[:, is_normal].mean()) for m in region_matrices]
    lesion["per_mass_image_mean"] = [float(m[:, ~is_normal].mean()) for m in region_matrices]
    lesion_low = lesion["per_image"]["ci_low"]
    lesion_high = lesion["per_image"]["ci_high"]

    criterion_summaries = {}
    for field, criterion in CRITERIA.items():
        matrices = []
        for size in SIZES:
            size_ids, counts, _ = region_counts(runs, size, field)
            if size_ids != all_image_ids:
                raise DatasetValidationError(f"Test images differ: size={size}, field={field}")
            matrices.append(counts)
        criterion_summaries[field] = {
            "criterion": criterion,
            **summarize_regions(matrices, n_images),
            "regions_per_size_seed_image": [matrix.tolist() for matrix in matrices],
        }

    ci_method = (
        f"{CI_LEVEL:.0%} percentile bootstrap over the {n_normal} normal test cases "
        f"({BOOTSTRAP_REPLICATES:,} replicates, RNG seed {BOOTSTRAP_SEED}), applied to "
        "the seed-mean FP rate. Case resamples are shared across seeds and sizes."
    )
    exact_ci_method = (
        f"{CI_LEVEL:.0%} exact binomial (Clopper-Pearson) interval for each seed's "
        f"k/{n_normal}. Conservative by construction; recommended at low counts, "
        "where the bootstrap interval tends to be too narrow."
    )
    title = "False positives vs. training size"
    description = (
        "Data-scaling test runs: seeds 42/43/44, final checkpoints, "
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
        "exact_ci_method": exact_ci_method,
        "case_level_label": "Retained-mask normal-case FP rate",
        "fp_rate_exact_ci_per_seed": exact_ci_per_seed,
        "fp_count_per_seed": counts_per_seed,
        "fp_count": count_metrics,
        "fp_rate": rate_metrics,
        "normal_image_ids": image_ids,
        "fp_flags_per_size_seed_case": [flags.tolist() for flags in flag_matrices],
        "lesion_level": {
            "label": "No-overlap FP regions per image",
            "definition": (
                "Retained predicted region with no pixel overlap with the ground-truth "
                "mass; every retained region on a normal image counts."
            ),
            "test_images": n_images,
            "ci_method": (
                f"{CI_LEVEL:.0%} percentile bootstrap over the {n_images} test images "
                f"({BOOTSTRAP_REPLICATES:,} replicates, RNG seed {BOOTSTRAP_SEED}), applied "
                "to the seed-mean FP regions per image."
            ),
            "image_ids": all_image_ids,
            **lesion,
            "unmatched_regions_per_size_seed_image": [m.tolist() for m in region_matrices],
        },
        "criterion_fp_regions": {
            "label": "Criterion-specific FP regions per image",
            "definition": (
                "Retained component count minus one if the criterion detects the mass "
                "on a mass image, otherwise all retained components; on normal "
                "images all retained components count. Averaged over all test images."
            ),
            "test_images": n_images,
            "ci_method": (
                f"{CI_LEVEL:.0%} percentile bootstrap over the {n_images} test images "
                f"({BOOTSTRAP_REPLICATES:,} replicates, RNG seed {BOOTSTRAP_SEED}), "
                "applied to the seed-mean FP regions per image. Image resamples "
                "are shared across seeds and sizes."
            ),
            "image_ids": all_image_ids,
            "criteria": criterion_summaries,
        },
    }
    json_path = SCRIPT_DIR / f"{OUT_BASE}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        f"# {title}", "", description, "",
        f"Dataset snapshot: `{data.snapshot_id}`.", "",
        "## Retained-mask normal-case FP rate", "",
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
    lines += [
        "",
        "## Exact binomial intervals per seed",
        "",
        exact_ci_method,
        "",
        "| Training size | Seed 42 | Seed 43 | Seed 44 |",
        "|--------------:|:-------:|:-------:|:-------:|",
    ]
    for index, size in enumerate(SIZES):
        cells = [
            f"{count}/{n_normal}: {low:.3f}–{high:.3f}"
            for count, (low, high) in zip(counts_per_seed[index], exact_ci_per_seed[index])
        ]
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## No-overlap FP regions per image",
        "",
        payload["lesion_level"]["definition"] + " "
        f"Reported as FP regions per image across all {n_images} test images. "
        "Seed columns give total FP regions.",
        "",
        payload["lesion_level"]["ci_method"],
        "",
        "| Training size | Seed 42 | Seed 43 | Seed 44 | No-overlap FP regions per image ± SD | 95% CI "
        "| Per normal image | Per mass image |",
        "|--------------:|:-------:|:-------:|:-------:|--------------------------:|-------:"
        "|-----------------:|---------------:|",
    ]
    for index, size in enumerate(SIZES):
        totals = lesion["regions_per_seed"][index]
        lines.append(
            f"| {size} | {totals[0]} | {totals[1]} | {totals[2]} | "
            f"{lesion['per_image']['mean'][index]:.3f} ± {lesion['per_image']['std'][index]:.3f} | "
            f"{lesion_low[index]:.3f}–{lesion_high[index]:.3f} | "
            f"{lesion['per_normal_image_mean'][index]:.3f} | "
            f"{lesion['per_mass_image_mean'][index]:.3f} |"
        )
    lines += [
        "", "## Criterion-specific FP regions per image", "",
        payload["criterion_fp_regions"]["definition"], "",
        payload["criterion_fp_regions"]["ci_method"], "",
        "Each cell shows the seed mean ± sample SD; [95% image-bootstrap CI]. "
        "All test images are included.",
    ]
    for group, fields in CRITERION_GROUPS.items():
        lines += [
            "", f"### {group}", "",
            "| Training size | " + " | ".join(CRITERIA[field] for field in fields) + " |",
            "|--------------:|" + "|".join("---------------------------:" for _ in fields) + "|",
        ]
        for index, size in enumerate(SIZES):
            cells = []
            for field in fields:
                metrics = criterion_summaries[field]["per_image"]
                cells.append(
                    f"{metrics['mean'][index]:.3f} ± {metrics['std'][index]:.3f} "
                    f"[{metrics['ci_low'][index]:.3f}–{metrics['ci_high'][index]:.3f}]"
                )
            lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines.append("")
    report = "\n".join(lines)
    (SCRIPT_DIR / f"{OUT_BASE}.md").write_text(report, encoding="utf-8")
    print(report)

    plot_by_size(
        [[count / n_normal for count in counts] for counts in counts_per_seed],
        rate_metrics["mean"], ci_low, ci_high,
        f"Retained-mask normal-case FP rate (n = {n_normal})", 1.0, CASE_FIGURE_BASE,
        "Retained-mask normal-case FP rate",
    )
    lesion_top = max(
        max(lesion_high), max(max(values) for values in lesion["per_image_per_seed"])
    ) * 1.15 or 1.0
    plot_by_size(
        lesion["per_image_per_seed"], lesion["per_image"]["mean"], lesion_low, lesion_high,
        f"No-overlap FP regions per image (n = {n_images})", lesion_top, LESION_FIGURE_BASE,
        "No-overlap FP regions per image",
    )
    for group, fields in CRITERION_GROUPS.items():
        out_base = f"evaluate_{'overlap' if group == 'Overlap detection' else 'centroid'}_fp_regions"
        plot_criteria_by_size(fields, criterion_summaries, group, out_base)


if __name__ == "__main__":
    main()
