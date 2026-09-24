"""Analyze IoU-threshold sensitivity for the 625-image data-scaling runs.

Uses the canonical predictions loader and its retained-component metrics.
Reports pathology-specific mass-area quartiles, overall overlap recall, and
IoU-independent triage flagging as mean ± sample SD across training seeds.

Run from the project root:
    python analysis/data_scaling/analyze_iou_tresholds/analyze_iou_thresholds.py

Output: analyze_iou_thresholds.md beside this script.
"""

import sys
from pathlib import Path
from statistics import mean, stdev

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import (
    DatasetValidationError,
    load_predictions_dataset,
)

SIZE = 625
SEEDS = [42, 43, 44]
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"
CHECKPOINT = "final"
IOU_THRESHOLDS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
LITERATURE_ROWS = [
    (0.0, "[Vorontsov et al. (Radiology: AI, 2019)](https://doi.org/10.1148/ryai.2019180014)", "Any predicted lesion overlap"),
    (0.1, "[Wei et al. (Nature Communications, 2024)](https://doi.org/10.1038/s41467-024-51260-6)", "Low-overlap detection before classification"),
    (0.2, "[Tiyarattanachai et al. (PLOS ONE, 2021)](https://doi.org/10.1371/journal.pone.0252882)", "Ultrasound lesion localization"),
    (0.5, "[Vorontsov et al. (Radiology: AI, 2019)](https://doi.org/10.1148/ryai.2019180014)", "High-overlap detection; favors PPV over sensitivity"),
]


def select_runs(rows):
    runs = {seed: [] for seed in SEEDS}
    for row in rows:
        if (
            row.experiment_name == EXPERIMENT_NAME
            and row.evaluation_split == EVALUATION_SPLIT
            and row.training_set_size == SIZE
            and row.checkpoint == CHECKPOINT
            and row.seed in SEEDS
        ):
            runs[row.seed].append(row)

    reference_cases = None
    for seed, run in runs.items():
        if not run:
            raise DatasetValidationError(f"Missing scaling run: size={SIZE}, seed={seed}")
        if len({row.configuration_id for row in run}) != 1:
            raise DatasetValidationError(f"Multiple configurations for seed={seed}")
        cases = {
            row.image_id: (
                row.pathology, row.mass_present, row.ground_truth_mass_area_px,
                row.image_height_px, row.image_width_px, row.reference_path,
                row.original_data_path, row.evaluation_unit,
            )
            for row in run
        }
        if len(cases) != len(run):
            raise DatasetValidationError(f"Duplicate cases for seed={seed}")
        if reference_cases is None:
            reference_cases = cases
        if cases != reference_cases:
            raise DatasetValidationError(f"Test cases differ for seed={seed}")
        for row in run:
            if row.pathology in ("malignant", "benign"):
                iou = row.max_retained_component_iou
                if (not row.mass_present or row.ground_truth_mass_area_px <= 0
                        or iou is None or not 0.0 <= iou <= 1.0):
                    raise DatasetValidationError(f"Invalid mass case: {row.image_id}")
            elif row.pathology == "normal":
                if (row.mass_present or row.ground_truth_mass_area_px != 0
                        or row.max_retained_component_iou is not None
                        or row.normal_false_positive != row.triage_detection_flag):
                    raise DatasetValidationError(f"Invalid normal case: {row.image_id}")
            else:
                raise DatasetValidationError(f"Unknown pathology: {row.pathology}")
        for category in ("malignant", "benign", "normal"):
            count = sum(row.pathology == category for row in run)
            if count < (1 if category == "normal" else 4):
                raise DatasetValidationError(f"Insufficient {category} cases for seed={seed}")
    return runs


def mass_area_quartiles(rows, category):
    cases = sorted(
        (row.ground_truth_mass_area_px, row.image_id)
        for row in rows if row.pathology == category
    )
    group_size, remainder = divmod(len(cases), 4)
    quartiles = {}
    start = 0
    for index in range(4):
        end = start + group_size + (index < remainder)
        quartiles[f"Q{index + 1}"] = cases[start:end]
        start = end
    return quartiles


def evaluate_seed(rows, cases, threshold):
    by_image = {row.image_id: row for row in rows}
    return mean(
        by_image[image_id].max_retained_component_iou > threshold
        for _, image_id in cases
    )


def format_metric(values):
    return f"{mean(values):.3f} ± {stdev(values) if len(values) > 1 else 0.0:.3f}"


def main():
    data = load_predictions_dataset()
    runs = select_runs(data.rows)
    reference_rows = runs[SEEDS[0]]
    quartiles = {
        category: mass_area_quartiles(reference_rows, category)
        for category in ("malignant", "benign")
    }
    all_cases = {
        category: [case for cases in groups.values() for case in cases]
        for category, groups in quartiles.items()
    }

    def recall_metric(cases, threshold):
        return format_metric([
            evaluate_seed(runs[seed], cases, threshold) for seed in SEEDS
        ])

    lines = [
        "# IoU-threshold sensitivity", "",
        f"Training size: {SIZE} images; experiment: {EXPERIMENT_NAME}; "
        f"evaluation split: {EVALUATION_SPLIT}; checkpoint: {CHECKPOINT}. "
        f"The predicted-component noise floor is fixed at "
        f"{MIN_PRED_AREA_FRACTION * 100:g}% of image area "
        "(pixel cutoff = int(fraction × image height × image width)). "
        f"Values are mean ± sample standard deviation across {len(SEEDS)} "
        f"training seeds ({', '.join(map(str, SEEDS))}).", "",
        "Source: `analysis/predictions_dataset/predictions_dataset.csv` via the "
        f"canonical loader. Dataset snapshot: `{data.snapshot_id}`.", "",
        "Detection requires max_retained_component_iou > threshold, matching "
        "the canonical dataset's strict comparison. At IoU 0.00 this requires "
        "any positive overlap. Unlike the previous analysis, equality at a "
        "positive threshold is not counted as detection.", "",
        "## IoU thresholds in prior studies and this study", "",
        "Detection is combined benign/malignant recall on this study's cases. "
        "Literature descriptions were verified against the papers' full texts "
        "(September 2026). Attributions carried over from the previous analysis "
        "that could not be confirmed (Bellver et al. at IoU 0.3, Bilic et al. at "
        "IoU 0.5, Kour & Adilakshmi at IoU 0.9) were removed; Vorontsov et al., "
        "which reports detection at IoU greater than 0, 0.25, and 0.5, anchors "
        "both the 0.0 and 0.5 rows. Wei et al. "
        "and Tiyarattanachai et al. compute IoU on bounding boxes, whereas "
        "Vorontsov et al. and this study use mask-based IoU.", "",
        "| IoU | Detection | Paper | Usage |",
        "|----:|----------:|:------|:------|",
    ]
    combined_cases = all_cases["benign"] + all_cases["malignant"]
    for threshold, paper, usage in LITERATURE_ROWS:
        lines.append(
            f"| {threshold:.1f} | {recall_metric(combined_cases, threshold)} | "
            f"{paper} | {usage} |"
        )

    for category, groups in quartiles.items():
        lines.extend([
            "", f"## {category.capitalize()} overlap-recall by ground-truth mass-area quartile", "",
            f"Q1 contains the smallest {category} masses and Q4 the largest. "
            "Cases are sorted by area, then image ID to break ties, and split "
            "as evenly as possible; earlier quartiles receive any extra cases.", "",
            "| Quartile | n | Ground-truth mass area range (px²) |",
            "|:---------|--:|------------------------------------:|",
        ])
        for quartile, cases in groups.items():
            lines.append(
                f"| {quartile} | {len(cases)} | {cases[0][0]:,}–{cases[-1][0]:,} |"
            )
        lines.extend([
            "", "| IoU threshold | Q1 recall | Q2 recall | Q3 recall | Q4 recall |",
            "|--------------:|----------:|----------:|----------:|----------:|",
        ])
        for threshold in IOU_THRESHOLDS:
            metrics = [recall_metric(cases, threshold) for cases in groups.values()]
            lines.append(f"| {threshold:.2f} | " + " | ".join(metrics) + " |")

    lines.extend([
        "", "## Overall overlap recall by pathology", "",
        "Recall across all cases in each pathology group, independent of the "
        "pathology-specific quartiles above.", "",
        "| IoU threshold | Benign recall | Malignant recall |",
        "|--------------:|--------------:|-----------------:|",
    ])
    for threshold in IOU_THRESHOLDS:
        lines.append(
            f"| {threshold:.2f} | {recall_metric(all_cases['benign'], threshold)} | "
            f"{recall_metric(all_cases['malignant'], threshold)} |"
        )

    lines.extend([
        "", "## Triage flagging at the fixed noise floor", "",
        "Triage flagging does not use an IoU threshold: it requires any retained "
        "predicted mass component. It is reported once rather than repeated "
        "across the IoU sweep. The normal-case value is the false-positive rate.", "",
        "| Case group | n | Triage flagging / false-positive rate |",
        "|:-----------|--:|--------------------------------------:|",
    ])
    for category in ("benign", "malignant", "normal"):
        count = sum(row.pathology == category for row in reference_rows)
        rates = [
            mean(row.triage_detection_flag for row in runs[seed]
                 if row.pathology == category)
            for seed in SEEDS
        ]
        label = "Normal (false positive)" if category == "normal" else category.capitalize()
        lines.append(f"| {label} | {count} | {format_metric(rates)} |")

    output_path = SCRIPT_DIR / "analyze_iou_thresholds.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
