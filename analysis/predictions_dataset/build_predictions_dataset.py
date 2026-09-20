"""Build the canonical per-image prediction dataset from saved mask predictions.

Run from the repository root:
    python3 analysis/predictions_dataset/build_predictions_dataset.py

Output:
    analysis/predictions_dataset/predictions_dataset.csv

The CSV records direct image-level facts for segmentation, overlap-based
detection, and triage, for every evaluated prediction run. It intentionally
contains no aggregate statistics, confidence intervals, quartiles, or
interpretation; those live in downstream analyses.

This builder discovers prediction directories under ``predictions/`` so that new
runs (including future seeds and encoders) are picked up without editing this
file. Each prediction directory contributes one row per test image.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASET_DIR = PROJECT_ROOT / "nnUNet_raw/Dataset001_AUL"
LABELS_TS = DATASET_DIR / "labelsTs"
CASE_MAPPING = DATASET_DIR / "case_mapping.json"
PREDICTIONS_ROOT = PROJECT_ROOT / "predictions"
OUTPUT_PATH = SCRIPT_DIR / "predictions_dataset.csv"

MASS_VALUE = 2
MIN_PRED_AREA = 100
CONNECTIVITY = 2  # Full connectivity for 2D masks: 8-connected components.
VALID_LABELS = {0, 1, 2}

# The preliminary milestones trainer saves these named checkpoints. The mapping
# labels each to its training epoch; ``best`` and ``best_mass`` are selected
# dynamically and have no fixed epoch, so they are represented as None.
CHECKPOINT_EPOCHS = {
    "epoch50": 50,
    "epoch100": 100,
    "epoch150": 150,
    "epoch300": 300,
    "epoch500": 500,
    "epoch750": 750,
    "best": None,
    "best_mass": None,
    "final": 1000,
}

FIELDNAMES = (
    "configuration_id",
    "seed",
    "epoch",
    "checkpoint",
    "encoder",
    "prediction_path",
    "reference_path",
    "image_id",
    "pathology",
    "original_data_path",
    "evaluation_unit",
    "mass_present",
    "image_height_px",
    "image_width_px",
    "ground_truth_mass_area_px",
    "predicted_mass_area_raw_px",
    "predicted_mass_area_retained_px",
    "raw_intersection_area_px",
    "raw_union_area_px",
    "retained_intersection_area_px",
    "retained_union_area_px",
    "mass_dice",
    "mass_iou",
    "max_retained_component_iou",
    "triage_flag",
    "detection_flag",
    "normal_false_positive",
    "outcome_category",
)


class InputValidationError(RuntimeError):
    """Raised when an expected prediction or reference input is invalid."""


def relative_path(path):
    """Return a repository-relative POSIX path for portable public data."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_mask(path):
    """Load a two-dimensional hard-label mask with explicit validation."""
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise InputValidationError(
            f"Expected a 2D mask in {relative_path(path)}, got shape {mask.shape}"
        )
    values = set(np.unique(mask).tolist())
    if not values <= VALID_LABELS:
        raise InputValidationError(
            f"Unexpected labels {sorted(values)} in {relative_path(path)}; "
            f"expected a subset of {sorted(VALID_LABELS)}"
        )
    return mask


def retained_mass_mask(raw_mass_mask):
    """Retain 8-connected predicted mass components of at least 100 pixels."""
    components = label(raw_mass_mask, connectivity=CONNECTIVITY)
    retained = np.zeros_like(raw_mass_mask, dtype=bool)
    for component in range(1, components.max() + 1):
        component_mask = components == component
        if int(component_mask.sum()) >= MIN_PRED_AREA:
            retained[component_mask] = True
    return retained


def dice_score(prediction, reference):
    """Calculate Dice for binary masks, preserving the empty/empty convention."""
    intersection = int(np.logical_and(prediction, reference).sum())
    denominator = int(prediction.sum() + reference.sum())
    return 1.0 if denominator == 0 else 2 * intersection / denominator


def iou_score(prediction, reference):
    """Calculate IoU for binary masks, preserving the empty/empty convention."""
    union = int(np.logical_or(prediction, reference).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(prediction, reference).sum()) / union


def max_component_iou(prediction, reference):
    """Return the largest IoU between a retained component and the reference."""
    components = label(prediction, connectivity=CONNECTIVITY)
    ious = []
    for component in range(1, components.max() + 1):
        component_mask = components == component
        intersection = int(np.logical_and(component_mask, reference).sum())
        union = int(np.logical_or(component_mask, reference).sum())
        ious.append(intersection / union if union else 0.0)
    return max(ious, default=0.0)


def load_test_cases():
    """Load test references and the split/pathology mapping."""
    if not LABELS_TS.is_dir():
        raise InputValidationError(f"Missing reference directory: {relative_path(LABELS_TS)}")
    if not CASE_MAPPING.is_file():
        raise InputValidationError(f"Missing split manifest: {relative_path(CASE_MAPPING)}")

    with open(CASE_MAPPING) as file:
        mapping = json.load(file)

    test_entries = [entry for entry in mapping if entry.get("split") == "test"]
    by_case_id = {entry["case_name"]: entry for entry in test_entries}
    if len(by_case_id) != len(test_entries):
        raise InputValidationError("Duplicate test case IDs in case_mapping.json")

    label_case_ids = {path.stem for path in LABELS_TS.glob("*.png")}
    manifest_case_ids = set(by_case_id)
    missing_labels = sorted(manifest_case_ids - label_case_ids)
    unmapped_labels = sorted(label_case_ids - manifest_case_ids)
    if missing_labels or unmapped_labels:
        raise InputValidationError(
            "Test manifest/reference mismatch: "
            f"missing labels={missing_labels}, unmapped labels={unmapped_labels}"
        )

    cases = []
    for case_id in sorted(manifest_case_ids):
        entry = by_case_id[case_id]
        category = entry["category"]
        pathology = category.lower()
        if pathology not in {"benign", "malignant", "normal"}:
            raise InputValidationError(f"Unknown pathology '{category}' for {case_id}")
        if "original_file" not in entry:
            raise InputValidationError(f"Missing original_file for {case_id}")

        reference_path = LABELS_TS / f"{case_id}.png"
        reference = read_mask(reference_path)
        ground_truth_mass = reference == MASS_VALUE
        mass_present = pathology != "normal"
        if mass_present != bool(ground_truth_mass.any()):
            raise InputValidationError(
                f"Manifest pathology and ground-truth mass disagree for {case_id}"
            )

        cases.append({
            "image_id": case_id,
            "pathology": pathology,
            "original_data_path": (
                f"data/source/AUL/{category}/image/{entry['original_file']}"
            ),
            "reference_path": relative_path(reference_path),
            "reference_shape": reference.shape,
            "ground_truth_mass": ground_truth_mass,
            "mass_present": mass_present,
        })
    return cases


def parse_prediction_directory(directory_name):
    """Return (seed, epoch, checkpoint, encoder) for a prediction directory.

    Mirrors the preliminary milestones naming scheme
    ``predictions_milestones_625images_seed{SEED}_{checkpoint}``. Returned
    ``epoch`` is None for the dynamically selected ``best``/``best_mass``
    checkpoints. Unknown directory names raise so coverage stays explicit.
    """
    parts = directory_name.split("_")
    if not directory_name.startswith("predictions_milestones_625images_seed"):
        raise InputValidationError(
            f"Unrecognized prediction directory name: {directory_name}"
        )
    # Reconstruct the seed from the "seed{SEED}" token, and the checkpoint as
    # everything after that token (checkpoint names may contain underscores).
    seed = None
    checkpoint = None
    for i, part in enumerate(parts):
        if part.startswith("seed") and seed is None:
            seed = int(part[len("seed"):])
            checkpoint = "_".join(parts[i + 1:])
    if seed is None or not checkpoint:
        raise InputValidationError(f"Cannot parse seed/checkpoint from {directory_name}")
    if checkpoint not in CHECKPOINT_EPOCHS:
        raise InputValidationError(
            f"Unrecognized checkpoint '{checkpoint}' in {directory_name}"
        )
    return seed, CHECKPOINT_EPOCHS[checkpoint], checkpoint, "PlainConvUNet"


def discover_configurations():
    """Discover every prediction directory under predictions/, recursively."""
    configurations = []
    if not PREDICTIONS_ROOT.is_dir():
        raise InputValidationError(
            f"Missing predictions root: {relative_path(PREDICTIONS_ROOT)}"
        )
    for directory in sorted(PREDICTIONS_ROOT.rglob("predictions_milestones_*")):
        if not directory.is_dir():
            continue
        seed, epoch, checkpoint, encoder = parse_prediction_directory(directory.name)
        configurations.append({
            "configuration_id": directory.name,
            "seed": seed,
            "epoch": epoch,
            "checkpoint": checkpoint,
            "encoder": encoder,
            "prediction_directory": directory,
        })
    if not configurations:
        raise InputValidationError("No prediction directories discovered")
    return configurations


def validate_configuration(config, expected_filenames):
    """Require complete, exact prediction coverage for a configuration."""
    directory = config["prediction_directory"]
    if not directory.is_dir():
        raise InputValidationError(
            f"Missing prediction directory: {relative_path(directory)}"
        )

    filenames = {
        path.name for path in directory.glob("*.png")
        if path.name != "dataset.json"
    }
    missing = sorted(expected_filenames - filenames)
    unexpected = sorted(filenames - expected_filenames)
    if missing or unexpected:
        raise InputValidationError(
            f"Prediction coverage mismatch in {relative_path(directory)}: "
            f"missing={missing}, unexpected={unexpected}"
        )


def evaluate_case(config, case):
    """Return direct segmentation, detection, and triage facts for one image."""
    prediction_path = config["prediction_directory"] / f"{case['image_id']}.png"
    prediction = read_mask(prediction_path)
    if prediction.shape != case["reference_shape"]:
        raise InputValidationError(
            f"Shape mismatch for {case['image_id']} in {relative_path(prediction_path)}: "
            f"prediction={prediction.shape}, reference={case['reference_shape']}"
        )

    ground_truth = case["ground_truth_mass"]
    raw_prediction = prediction == MASS_VALUE
    retained_prediction = retained_mass_mask(raw_prediction)

    raw_intersection = int(np.logical_and(raw_prediction, ground_truth).sum())
    raw_union = int(np.logical_or(raw_prediction, ground_truth).sum())
    retained_intersection = int(np.logical_and(retained_prediction, ground_truth).sum())
    retained_union = int(np.logical_or(retained_prediction, ground_truth).sum())
    triage = bool(retained_prediction.any())

    if case["mass_present"]:
        detection = bool(retained_intersection > 0)
        if detection and not triage:
            raise AssertionError(
                f"Detection without triage for {case['image_id']} in {config['configuration_id']}"
            )
        if detection:
            outcome = "overlapping_detected"
        elif triage:
            outcome = "flagged_without_detection"
        else:
            outcome = "unflagged"
        mass_dice = dice_score(raw_prediction, ground_truth)
        mass_iou = iou_score(raw_prediction, ground_truth)
        max_retained_component_iou = max_component_iou(
            retained_prediction, ground_truth
        )
        normal_false_positive = None
    else:
        detection = None
        outcome = "normal_false_positive" if triage else "normal_true_negative"
        mass_dice = None
        mass_iou = None
        max_retained_component_iou = None
        normal_false_positive = triage

    return {
        "configuration_id": config["configuration_id"],
        "seed": config["seed"],
        "epoch": config["epoch"],
        "checkpoint": config["checkpoint"],
        "encoder": config["encoder"],
        "prediction_path": relative_path(prediction_path),
        "reference_path": case["reference_path"],
        "image_id": case["image_id"],
        "pathology": case["pathology"],
        "original_data_path": case["original_data_path"],
        "evaluation_unit": "image",
        "mass_present": case["mass_present"],
        "image_height_px": int(case["reference_shape"][0]),
        "image_width_px": int(case["reference_shape"][1]),
        "ground_truth_mass_area_px": int(ground_truth.sum()),
        "predicted_mass_area_raw_px": int(raw_prediction.sum()),
        "predicted_mass_area_retained_px": int(retained_prediction.sum()),
        "raw_intersection_area_px": raw_intersection,
        "raw_union_area_px": raw_union,
        "retained_intersection_area_px": retained_intersection,
        "retained_union_area_px": retained_union,
        "mass_dice": mass_dice,
        "mass_iou": mass_iou,
        "max_retained_component_iou": max_retained_component_iou,
        "triage_flag": triage,
        "detection_flag": detection,
        "normal_false_positive": normal_false_positive,
        "outcome_category": outcome,
    }


def write_rows(rows):
    """Write the canonical long-form CSV, using empty strings for null values."""
    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: "" if row[field] is None else row[field]
                for field in FIELDNAMES
            })


def main():
    cases = load_test_cases()
    expected_filenames = {f"{case['image_id']}.png" for case in cases}
    configurations = discover_configurations()

    configuration_ids = [config["configuration_id"] for config in configurations]
    if len(configuration_ids) != len(set(configuration_ids)):
        raise InputValidationError("Configuration IDs are not unique")

    rows = []
    for config in configurations:
        print(f"Evaluating {config['configuration_id']}...")
        validate_configuration(config, expected_filenames)
        rows.extend(evaluate_case(config, case) for case in cases)

    write_rows(rows)
    print(f"Wrote {relative_path(OUTPUT_PATH)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
