"""Explore the effect of a relative noise floor on detection. This script pulls only the final checkpoint results (not best)

Run from project root:
    python analysis/noise_floor_analysis/noise_floor_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.constants import MIN_PRED_AREA_FRACTION
from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

MASS_VALUE = 2
CONNECTIVITY = 2
EXPERIMENT_NAME = "data_scaling"
EVALUATION_SPLIT = "test"

# Weighted mean image area (px) of the external AUL set, from its size counts.
AUL_MEAN_AREA = 341420

# Relative floors to test, as fractions of image area (0.02% = 0.0002).
FRACTIONS = [
    0.00005,
    0.0001,
    0.0002,
    MIN_PRED_AREA_FRACTION,
    0.0004,
    0.0005,
]


def retained_mass_mask(raw_mass_mask, fraction):
    """Retain components >= ``fraction`` of the image area."""
    min_area = int(fraction * raw_mass_mask.size)
    components = label(raw_mass_mask, connectivity=CONNECTIVITY)
    retained = np.zeros_like(raw_mass_mask, dtype=bool)
    for component in range(1, components.max() + 1):
        component_mask = components == component
        if int(component_mask.sum()) >= min_area:
            retained[component_mask] = True
    return retained


def detection_results(reference, retained_prediction):
    """Evaluate all detection views from one retained-component labeling pass."""
    labeled_prediction = label(retained_prediction, connectivity=CONNECTIVITY)
    if labeled_prediction.max() == 0:
        return {criterion: False for criterion in CRITERIA}

    predicted_centroids = []
    max_iou = 0.0
    for component in range(1, labeled_prediction.max() + 1):
        component_mask = labeled_prediction == component
        predicted_centroids.append(np.argwhere(component_mask).mean(axis=0))
        intersection = int(np.logical_and(component_mask, reference).sum())
        union = int(np.logical_or(component_mask, reference).sum())
        if union:
            max_iou = max(max_iou, intersection / union)

    labeled_reference = label(reference, connectivity=CONNECTIVITY)
    closest_centroid_ratio = float("inf")
    predicted_centroids = np.asarray(predicted_centroids)
    for component in range(1, labeled_reference.max() + 1):
        coordinates = np.argwhere(labeled_reference == component)
        gt_centroid = coordinates.mean(axis=0)
        equivalent_diameter = 2 * np.sqrt(len(coordinates) / np.pi)
        closest_distance = np.linalg.norm(
            predicted_centroids - gt_centroid, axis=1
        ).min()
        closest_centroid_ratio = min(
            closest_centroid_ratio, closest_distance / equivalent_diameter
        )

    return {
        "triage": True,
        "centroid_025": closest_centroid_ratio <= 0.25,
        "centroid_050": closest_centroid_ratio <= 0.5,
        "centroid_100": closest_centroid_ratio <= 1.0,
        "overlap_000": max_iou > 0.0,
        "overlap_020": max_iou > 0.2,
        "overlap_050": max_iou > 0.5,
    }


CRITERIA = [
    "triage",
    "centroid_025",
    "centroid_050",
    "centroid_100",
    "overlap_000",
    "overlap_020",
    "overlap_050",
]
CRITERION_LABELS = {
    "triage": "Triage",
    "centroid_025": "Centroid 0.25",
    "centroid_050": "Centroid 0.5",
    "centroid_100": "Centroid 1.0",
    "overlap_000": "Overlap >0",
    "overlap_020": "Overlap >0.2",
    "overlap_050": "Overlap >0.5",
}
BASELINE_FIELDS = {
    "triage": "triage_detection_flag",
    "centroid_025": "centroid_detection_deq_025_flag",
    "centroid_050": "centroid_detection_deq_050_flag",
    "centroid_100": "centroid_detection_deq_100_flag",
    "overlap_000": "overlap_detection_iou_00_flag",
    "overlap_020": "overlap_detection_iou_02_flag",
    "overlap_050": "overlap_detection_iou_05_flag",
}


def load_mass_masks(row):
    """Return (prediction_mass_bool, reference_mass_bool) for one row."""
    pred_path = PROJECT_ROOT / row.prediction_path
    ref_path = PROJECT_ROOT / row.reference_path
    pred = np.asarray(Image.open(pred_path))
    ref = np.asarray(Image.open(ref_path))
    return pred == MASS_VALUE, ref == MASS_VALUE


def main():
    data = load_predictions_dataset()
    rows = [
        r for r in data.rows
        if r.experiment_name == EXPERIMENT_NAME
        and r.evaluation_split == EVALUATION_SPLIT
        and r.checkpoint == "final"
    ]
    # Only mass-present cases matter: detection is defined on positive masses.
    rows = [r for r in rows if r.mass_present]
    print(f"{len(rows)} mass-present rows")

    # Baseline detection under the selected relative floor: reuse the CSV flags.
    baseline = {
        criterion: {
            (r.configuration_id, r.image_id): getattr(r, BASELINE_FIELDS[criterion])
            for r in rows
        }
        for criterion in CRITERIA
    }

    # Pre-load masks once.
    masks = {
        (r.configuration_id, r.image_id): load_mass_masks(r)
        for r in rows
    }

    results = []
    headers = ["Fraction", "AUL average (px)"] + [
        CRITERION_LABELS[criterion] for criterion in CRITERIA
    ]
    print(f"\n{' | '.join(headers)}")
    print(" | ".join("-" * len(header) for header in headers))
    for fraction in FRACTIONS:
        lost = {c: 0 for c in CRITERIA}
        baseline_mismatches = []
        for r in rows:
            key = (r.configuration_id, r.image_id)
            pred_mass, ref_mass = masks[key]
            retained = retained_mass_mask(pred_mass, fraction)
            detections = detection_results(ref_mass, retained)
            for criterion in CRITERIA:
                was_detected = baseline[criterion][key]
                is_detected = detections[criterion]
                if was_detected and not is_detected:
                    lost[criterion] += 1
                if (
                    fraction == MIN_PRED_AREA_FRACTION
                    and was_detected != is_detected
                ):
                    baseline_mismatches.append(
                        (r.configuration_id, r.image_id, criterion,
                         was_detected, is_detected)
                    )
        if fraction == MIN_PRED_AREA_FRACTION and baseline_mismatches:
            raise AssertionError(
                "Noise-floor sweep does not reproduce the predictions-dataset "
                f"baseline at fraction {MIN_PRED_AREA_FRACTION}: "
                f"{baseline_mismatches[:10]}"
            )
        avg_px = int(fraction * AUL_MEAN_AREA)
        results.append({
            "fraction": fraction,
            "aul_average_px": avg_px,
            **{f"{criterion}_lost": lost[criterion] for criterion in CRITERIA},
        })
        values = [f"{fraction:.5f}", str(avg_px)] + [
            str(lost[criterion]) for criterion in CRITERIA
        ]
        print(" | ".join(values))

    write_markdown(results)
    write_json(results)


def write_markdown(results):
    """Write the sweep table plus a brief reasoning/process note."""
    lines = [
        "# Relative noise-floor sweep",
        "",
        "The pipeline drops predicted mass components smaller than a noise floor "
        "to remove speckle. Because image dimensions vary (both within this "
        "dataset and across external validation sets), a fixed pixel floor is "
        "inconsistent. This sweep replaces it with a *relative* floor expressed "
        "as a fraction of each image's area, and measures how many mass-present "
        "cases that are detected under the selected relative floor become missed "
        "under each candidate.",
        "",
        "Seven detection criteria are checked: triage (any retained mass), "
        "centroid distances of 0.25, 0.5, and 1.0 equivalent diameters, and "
        "overlap IoU thresholds of >0, >0.2, and >0.5. Each is compared with "
        "the selected relative-floor baseline flags already in the predictions "
        "dataset.",
        "",
        "The AUL-average column converts each fraction to pixels using the "
        "weighted mean image area of the external AUL set (~341,420 px).",
        "",
        "| Fraction | AUL average (px) | Triage | Centroid 0.25 | Centroid 0.5 | Centroid 1.0 | Overlap >0 | Overlap >0.2 | Overlap >0.5 |",
        "|------:|------:|------:|------:|------:|------:|------:|------:|------:|",
    ]
    for r in results:
        values = [
            r[f"{criterion}_lost"]
            for criterion in CRITERIA
        ]
        lines.append(
            f"| {r['fraction']:.5f} | {r['aul_average_px']} | "
            + " | ".join(str(value) for value in values)
            + " |"
        )
    lines.extend([
        "",
        "*All numeric detection columns are counts of predictions lost relative "
        "to the selected 0.03%-of-image-area baseline.*",
        "",
    ])
    out = SCRIPT_DIR / "noise_floor_analysis.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def write_json(results):
    import json
    payload = {
        "description": (
            "Sweep of a relative noise floor (fraction of image area) versus "
            "the selected relative baseline. 'lost' counts mass-present cases that "
            "were detected under the selected floor but missed under the candidate "
            "floor."
        ),
        "aul_mean_area_px": AUL_MEAN_AREA,
        "baseline_floor_fraction": MIN_PRED_AREA_FRACTION,
        "criteria": CRITERIA,
        "results": results,
    }
    out = SCRIPT_DIR / "noise_floor_analysis.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
