"""Report raw-mask segmentation Dice across training sizes and seeds.

Run from the repository root:
    python3 analysis/data_scaling/data_scale_by_segmentation/evaluate_segmentation.py

Prints four Markdown tables and saves them to evaluate_segmentation.md beside
this script, along with evaluate_segmentation.png. Uses only the canonical
prediction dataset, not the saved masks.
"""

import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.load_predictions_dataset import (
    DatasetValidationError,
    load_predictions_dataset,
)

SIZES = (5, 10, 20, 40, 80, 160, 320, 625)
SEEDS = (42, 43, 44)
CHECKPOINT = "final"
OUTPUT_PATH = Path(__file__).with_suffix(".md")


def select_runs(rows):
    runs = defaultdict(list)
    for row in rows:
        if (
            row.experiment_name == "data_scaling"
            and row.evaluation_split == "test"
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
                row.image_id: (row.pathology, row.mass_present, row.ground_truth_mass_area_px)
                for row in run
            }
            if reference_cases is None:
                reference_cases = cases
            if cases != reference_cases:
                raise DatasetValidationError(f"Test cases differ: size={size}, seed={seed}")
            for row in run:
                if row.liver_dice is None:
                    raise DatasetValidationError(f"Missing liver Dice: {row.image_id}")
                if row.mass_present and (
                    row.pathology not in ("benign", "malignant")
                    or row.ground_truth_mass_area_px <= 0
                ):
                    raise DatasetValidationError(f"Invalid mass case: {row.image_id}")
    return runs


def raw_mass_dice(row):
    # mass_dice in the CSV uses retained components, not the raw prediction.
    return 2 * row.raw_intersection_area_px / (
        row.predicted_mass_area_raw_px + row.ground_truth_mass_area_px
    )


def summarize(values):
    if any(value is None for value in values):
        return "N/A"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def mass_mean(run, subset, pathology=None):
    scores = [
        raw_mass_dice(row)
        for row in run
        if row.mass_present
        and (pathology is None or row.pathology == pathology)
        and (subset != "nonempty" or row.predicted_mass_area_raw_px > 0)
        and (subset != "overlapping" or row.raw_intersection_area_px > 0)
    ]
    return mean(scores) if scores else None


def false_positive_rate(run):
    normal_cases = [row for row in run if row.pathology == "normal" and not row.mass_present]
    if not normal_cases:
        raise DatasetValidationError("No normal cases for false positive rate")
    return mean(row.predicted_mass_area_raw_px > 0 for row in normal_cases)


def scaling_table(runs, subset):
    liver_label = "Liver Dice" if subset == "all" else "Liver Dice (all cases)"
    lines = [
        f"| Size | {liver_label} | Malignant Dice | Benign Dice | Combined Mass Dice | False Positive Rate |",
        "|-----:|-----------:|---------------:|------------:|-------------------:|--------------------:|",
    ]
    for size in SIZES:
        seed_runs = [runs[size, seed] for seed in SEEDS]
        values = [summarize([mean(row.liver_dice for row in run) for run in seed_runs])]
        values.extend(
            summarize([mass_mean(run, subset, pathology) for run in seed_runs])
            for pathology in ("malignant", "benign", None)
        )
        values.append(summarize([false_positive_rate(run) for run in seed_runs]))
        lines.append(f"| {size} | " + " | ".join(values) + " |")
    return lines


def type_and_size_table(runs):
    size = max(SIZES)
    lines = [
        "| Class | n | Dice | Median area (px²) | Area range (px²) |",
        "|:------|--:|-----:|------------------:|-----------------:|",
    ]
    for pathology in ("benign", "malignant"):
        areas = [
            row.ground_truth_mass_area_px
            for row in runs[size, SEEDS[0]]
            if row.mass_present and row.pathology == pathology
        ]
        if not areas:
            raise DatasetValidationError(f"No mass-present {pathology} cases")
        dice = summarize([mass_mean(runs[size, seed], "all", pathology) for seed in SEEDS])
        lines.append(
            f"| {pathology} | {len(areas)} | {dice} | {median(areas):,.0f} | "
            f"{min(areas):,}–{max(areas):,} |"
        )
    return lines


def build_report(dataset):
    runs = select_runs(dataset.rows)
    case_count = len(runs[SIZES[0], SEEDS[0]])
    lines = [
        "# Checkpoint: final (aggregated across seeds, mean ± std)",
        "",
        f"Dataset snapshot: `{dataset.snapshot_id}`. Seeds: {', '.join(map(str, SEEDS))}.",
        "",
        "False positive rate is the fraction of normal test cases with any raw predicted "
        "mass pixels (no component-size filtering), reported as mean ± sample SD across "
        "seeds. It uses all normal cases in every table, regardless of the mass-case filter.",
        "",
        "## Mass Dice across all mass-present cases",
        "",
        "For each training-size and seed run, the mass Dice columns include all "
        "mass-present cases, including empty and non-overlapping raw predicted mass masks. "
        f"Liver Dice is calculated over all {case_count} test images. "
        "Values are mean ± sample SD across seeds.",
        "",
        *scaling_table(runs, "all"),
        "",
        "## Segmentation Dice by mass type and size",
        "",
        f"At the {max(SIZES)}-image training size, Dice is calculated per mass case "
        "from raw hard predictions, averaged within each pathology for each seed, "
        "then summarized across seeds. Area is the ground-truth mass area.",
        "",
        *type_and_size_table(runs),
        "",
        "## Mass Dice excluding empty raw mass predictions",
        "",
        "For each training-size and seed run, the mass Dice columns exclude "
        "mass-present cases whose raw predicted mass mask contains no mass pixels. "
        f"Liver Dice remains calculated over all {case_count} test images. "
        "Values are mean ± sample SD across seeds.",
        "",
        *scaling_table(runs, "nonempty"),
        "",
        "## Mass Dice with overlapping raw mass predictions",
        "",
        "For each training-size and seed run, the mass Dice columns include only "
        "mass-present cases whose raw predicted mass mask spatially overlaps the "
        f"ground-truth mass. Liver Dice remains calculated over all {case_count} test images. "
        "Values are mean ± sample SD across seeds.",
        "",
        *scaling_table(runs, "overlapping"),
        "",
    ]
    if any("N/A" in line for line in lines):
        lines.extend([
            "N/A indicates that at least one seed has no eligible cases for that metric.",
            "",
        ])
    return "\n".join(lines)


def plot_segmentation(runs):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 6.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    series = (
        ("Liver", "#2b7bde", "-", lambda run: mean(row.liver_dice for row in run)),
        ("Malignant mass", "#f46632", "-", lambda run: mass_mean(run, "all", "malignant")),
        ("Benign mass", "#3b7011", "-", lambda run: mass_mean(run, "all", "benign")),
        ("Combined mass", "#9467bd", "-", lambda run: mass_mean(run, "all")),
        ("False positive rate", "#808080", ":", false_positive_rate),
    )
    positions = list(range(len(SIZES)))
    for label, color, linestyle, metric in series:
        seed_values = [[metric(runs[size, seed]) for seed in SEEDS] for size in SIZES]
        axes.errorbar(
            positions,
            [mean(values) for values in seed_values],
            yerr=[stdev(values) for values in seed_values],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            marker="o",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.5,
            capsize=4,
            elinewidth=1.8,
        )
    axes.set_xticks(positions, labels=SIZES)
    axes.set_yticks([value / 10 for value in range(11)])
    axes.set_ylim(0, 1)
    axes.set_xlabel("Training images", fontsize=13)
    axes.set_ylabel("Dice / false positive rate", fontsize=13)
    axes.tick_params(labelsize=11, colors="#888888")
    axes.set_axisbelow(True)
    axes.grid(axis="y", color="#e5e5e5")
    for spine in axes.spines.values():
        spine.set_color("#bfbfbf")
    axes.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
        fontsize=10, framealpha=0.95, fancybox=False,
    )
    figure.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=200)


def main():
    dataset = load_predictions_dataset()
    report = build_report(dataset)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    plot_segmentation(select_runs(dataset.rows))
    print(report, end="")


if __name__ == "__main__":
    main()
