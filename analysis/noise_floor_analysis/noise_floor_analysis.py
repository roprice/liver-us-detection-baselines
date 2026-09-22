"""Explore the effect of a relative noise floor on detection.

The current pipeline drops predicted mass components smaller than a fixed 100 px.
Because image dimensions vary, a fixed floor is inconsistent across images. This
script sweeps a *relative* floor (a fraction of image area) and reports, per
criterion (triage, centroid 0.5, overlap 0.2), how many mass-present cases that
are detected under the fixed-100px baseline become *missed* when the relative
floor is used instead.

It mirrors the matching logic in ``build_predictions_dataset.py`` (full 2D
connectivity, per-component centroid / IoU criteria) so the "true prediction"
counts line up with the CSV flags, but it recomputes retention locally so the
relative floor can be swapped in without rebuilding the dataset.

Run from project root:
    python analysis/noise_floor_analysis/noise_floor_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.predictions_dataset.load_predictions_dataset import (
    load_predictions_dataset,
)

MASS_VALUE = 2
CONNECTIVITY = 2
FIXED_FLOOR = 100
EXPERIMENT_NAME = "milestones_pilot"
EVALUATION_SPLIT = "test"

# Weighted mean image area (px) of the external AUL set, from its size counts.
AUL_MEAN_AREA = 341420

# Relative floors to test, as fractions of image area (0.02% = 0.0002).
FRACTIONS = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]


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


def centroid_detects(reference, retained_prediction, diameter_factor=0.5):
    labeled_reference = label(reference, connectivity=CONNECTIVITY)
    labeled_prediction = label(retained_prediction, connectivity=CONNECTIVITY)
    centroids = []
    for c in range(1, labeled_prediction.max() + 1):
        coords = np.argwhere(labeled_prediction == c)
        centroids.append(coords.mean(axis=0))
    if not centroids:
        return False
    centroids = np.asarray(centroids)
    for c in range(1, labeled_reference.max() + 1):
        coords = np.argwhere(labeled_reference == c)
        gt_centroid = coords.mean(axis=0)
        diameter = 2 * np.sqrt(len(coords) / np.pi)
        dist = np.linalg.norm(centroids - gt_centroid, axis=1).min()
        if dist <= diameter_factor * diameter:
            return True
    return False


def max_component_iou(retained_prediction, reference):
    components = label(retained_prediction, connectivity=CONNECTIVITY)
    best = 0.0
    for c in range(1, components.max() + 1):
        component_mask = components == c
        inter = int(np.logical_and(component_mask, reference).sum())
        union = int(np.logical_or(component_mask, reference).sum())
        if union:
            best = max(best, inter / union)
    return best


def detect(retained_prediction, reference, criterion):
    """Apply one detection criterion to a retained prediction."""
    if criterion == "triage":
        return bool(retained_prediction.any())
    if criterion == "centroid_050":
        return centroid_detects(reference, retained_prediction, 0.5)
    if criterion == "overlap_020":
        return max_component_iou(retained_prediction, reference) > 0.2
    raise ValueError(f"unknown criterion {criterion}")


CRITERIA = ["triage", "centroid_050", "overlap_020"]


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
    ]
    # Only mass-present cases matter: detection is defined on positive masses.
    rows = [r for r in rows if r.mass_present]
    print(f"{len(rows)} mass-present rows")

    # Baseline detection under the fixed 100 px floor: reuse the CSV flags.
    baseline = {
        "triage": {r.image_id: r.triage_detection_flag for r in rows},
        "centroid_050": {r.image_id: r.centroid_detection_deq_050_flag for r in rows},
        "overlap_020": {r.image_id: r.overlap_detection_iou_02_flag for r in rows},
    }

    # Pre-load masks once.
    masks = {}
    for r in rows:
        masks[r.image_id] = load_mass_masks(r)

    results = []
    print("\nFraction | AUL average (px) | triage lost | centroid0.5 lost | overlap0.2 lost")
    print("---------|------------------|-------------|-----------------|----------------")
    for fraction in FRACTIONS:
        lost = {c: 0 for c in CRITERIA}
        for r in rows:
            pred_mass, ref_mass = masks[r.image_id]
            retained = retained_mass_mask(pred_mass, fraction)
            for criterion in CRITERIA:
                was_detected = baseline[criterion][r.image_id]
                is_detected = detect(retained, ref_mass, criterion)
                if was_detected and not is_detected:
                    lost[criterion] += 1
        avg_px = int(fraction * AUL_MEAN_AREA)
        results.append({
            "fraction": fraction,
            "aul_average_px": avg_px,
            "triage_lost": lost["triage"],
            "centroid_050_lost": lost["centroid_050"],
            "overlap_020_lost": lost["overlap_020"],
        })
        print(f"{fraction:.5f} | {avg_px:>16} | {lost['triage']:>11} | "
              f"{lost['centroid_050']:>15} | {lost['overlap_020']:>14}")

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
        "cases that are detected under the original fixed floor become missed "
        "under each candidate.",
        "",
        "Three detection criteria are checked: triage (any retained mass), "
        "centroid 0.5, and overlap IoU > 0.2. The loss is counted against the "
        "fixed 100 px baseline flags already in the predictions dataset.",
        "",
        "The AUL-average column converts each fraction to pixels using the "
        "weighted mean image area of the external AUL set (~341,420 px).",
        "",
        "| Fraction | AUL average (px) | Triage lost | Centroid 0.5 lost | Overlap 0.2 lost |",
        "|------:|------:|------:|------:|------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['fraction']:.5f} | {r['aul_average_px']} | "
            f"{r['triage_lost']} | {r['centroid_050_lost']} | {r['overlap_020_lost']} |"
        )
    lines.append("")
    out = SCRIPT_DIR / "noise_floor_analysis.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def write_json(results):
    import json
    payload = {
        "description": (
            "Sweep of a relative noise floor (fraction of image area) versus "
            "the fixed 100 px baseline. 'lost' counts mass-present cases that "
            "were detected under 100 px but missed under the candidate floor."
        ),
        "aul_mean_area_px": AUL_MEAN_AREA,
        "baseline_floor_px": FIXED_FLOOR,
        "criteria": CRITERIA,
        "results": results,
    }
    out = SCRIPT_DIR / "noise_floor_analysis.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
