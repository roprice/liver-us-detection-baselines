"""Centroid detection vs. epoch for the preliminary milestones run (seed 42, 625 images).

Centroid detection uses the LUNA16-style 2D criterion: for each ground-truth
mass component, find the closest retained predicted component's centroid. The
component is detected when that predicted centroid lies within half the
ground-truth component's equivalent circular diameter ``D = 2 * sqrt(area / pi)``.
A case is detected when any ground-truth component is detected.

This is scale-relative (the tolerance grows with the mass's own size), not a
fixed physical distance, because this dataset carries no physical spacings.

On this dataset every image is one patient with at most one mass, so detection
is effectively case-level. Reports, per milestone checkpoint and per mass
grouping (combined / malignant / benign): case-level recall, precision, F1, and
false-positive rate over Normal cases.

The 100 px noise floor filters speckle from predicted components before
centroids are computed, matching the other detection scripts.

Single-seed run (seed 42, 625 images, joint-selected checkpoints); no error bars.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from PIL import Image
from skimage.measure import label

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# --- Detection config (mirrors the overlap detection script) ---
LABELS_TS = PROJECT_ROOT / "nnUNet_raw/Dataset001_AUL/labelsTs"
CASE_MAPPING = PROJECT_ROOT / "nnUNet_raw/Dataset001_AUL/case_mapping.json"

MASS_VALUE = 2
MIN_PRED_AREA = 100  # noise floor (px^2), same as the detection script

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'sans-serif'],
    'font.size': 10,
    'font.weight': '400',
    'axes.labelsize': 10,
    'axes.labelcolor': '#999999',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#e0e0e0',
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': True,
    'axes.spines.right': True,
})

PRED_ROOT = PROJECT_ROOT / "predictions/preliminary_milestones_test/seed42"
OUT_DIR = SCRIPT_DIR

# Folder name -> epoch number.
EPOCH_DIRS = [
    ("predictions_milestones_625images_seed42_epoch50", 50),
    ("predictions_milestones_625images_seed42_epoch100", 100),
    ("predictions_milestones_625images_seed42_epoch150", 150),
    ("predictions_milestones_625images_seed42_epoch300", 300),
    ("predictions_milestones_625images_seed42_epoch500", 500),
    ("predictions_milestones_625images_seed42_epoch750", 750),
]

METRICS = {
    "case_f1":          ("Case F1",          "#eb6834"),
    "case_precision":   ("Case precision",   "#f2a17e"),
    "case_recall":      ("Case recall",      "#f5c0aa"),
    "case_fp_rate":     ("Case FP rate",     "#3b6d11"),
}

PLOT_SERIES = [
    ("combined",  "case_recall", "Combined centroid",  "#2a78d6"),
    ("malignant", "case_recall", "Malignant centroid", "#eb6834"),
    ("benign",    "case_recall", "Benign centroid",    "#3b6d11"),
]


def get_class_map():
    """Build filename -> class mapping from case_mapping.json (test split)."""
    if not CASE_MAPPING.exists():
        print(f"WARNING: {CASE_MAPPING} not found")
        return {}
    with open(CASE_MAPPING) as f:
        mapping = json.load(f)
    return {e["case_name"] + ".png": e["category"].lower()
            for e in mapping if e["split"] == "test"}


def filter_small_components(mask, min_area):
    """Return a binary mask containing only components >= min_area pixels."""
    labeled = label(mask)
    filtered = np.zeros_like(mask)
    for component in range(1, labeled.max() + 1):
        component_mask = labeled == component
        if component_mask.sum() >= min_area:
            filtered[component_mask] = 1
    return filtered


def centroid_detects_ground_truth(gt_mass, filtered_prediction):
    """Whether any GT component has a retained predicted centroid within half
    its equivalent circular diameter (LUNA16-style 2D centroid criterion)."""
    labeled_gt = label(gt_mass)
    labeled_prediction = label(filtered_prediction)
    predicted_centroids = []
    for component in range(1, labeled_prediction.max() + 1):
        coordinates = np.argwhere(labeled_prediction == component)
        predicted_centroids.append(coordinates.mean(axis=0))

    if not predicted_centroids:
        return False

    predicted_centroids = np.asarray(predicted_centroids)
    for component in range(1, labeled_gt.max() + 1):
        coordinates = np.argwhere(labeled_gt == component)
        gt_centroid = coordinates.mean(axis=0)
        equivalent_diameter = 2 * np.sqrt(len(coordinates) / np.pi)
        closest_distance = np.linalg.norm(
            predicted_centroids - gt_centroid, axis=1).min()
        if closest_distance <= 0.5 * equivalent_diameter:
            return True
    return False


def evaluate_centroid(pred_dir, test_files, class_map, min_pred_area,
                      positive_class="malignant"):
    """Return (tp, fp, fn, n_normal) at case level for centroid detection."""
    tp = fn = 0
    fp = 0
    n_normal = 0

    for fname in test_files:
        cls = class_map.get(fname, "unknown")
        pred_path = pred_dir / fname
        if not pred_path.exists():
            continue
        gt = np.array(Image.open(LABELS_TS / fname))
        pred = np.array(Image.open(pred_path))
        gt_mass = (gt == MASS_VALUE).astype(np.uint8)
        filtered_pred = filter_small_components(
            (pred == MASS_VALUE).astype(np.uint8), min_pred_area)

        is_positive = (
            cls in ("malignant", "benign")
            if positive_class == "combined"
            else cls == positive_class
        )
        if is_positive:
            if centroid_detects_ground_truth(gt_mass, filtered_pred):
                tp += 1
            else:
                fn += 1
        elif cls == "normal":
            n_normal += 1
            if filtered_pred.any():
                fp += 1

    return int(tp), int(fp), int(fn), int(n_normal)


def evaluate_epoch(pred_dir, test_files, class_map, positive_class="malignant"):
    """Run centroid eval for one checkpoint, return the 4 metrics."""
    ctp, cfp, cfn, n_normal = evaluate_centroid(
        pred_dir, test_files, class_map, MIN_PRED_AREA,
        positive_class=positive_class)

    cprec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
    crec = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
    cf1 = 2 * cprec * crec / (cprec + crec) if (cprec + crec) > 0 else 0.0
    cfp_rate = cfp / n_normal if n_normal > 0 else 0.0

    return {
        "case_f1": cf1,
        "case_precision": cprec,
        "case_recall": crec,
        "case_fp_rate": cfp_rate,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    class_map = get_class_map()
    if not LABELS_TS.exists():
        print(f"ERROR: {LABELS_TS} not found")
        return

    test_files = sorted(f for f in os.listdir(LABELS_TS) if f.endswith(".png"))
    print(f"Test set: {len(test_files)} files, centroid detection = predicted "
          f"centroid within 0.5x GT equivalent diameter, noise floor "
          f"{MIN_PRED_AREA} px")

    epochs = []
    metrics = {key: [] for key in METRICS}
    benign_metrics = {key: [] for key in METRICS}
    combined_metrics = {key: [] for key in METRICS}
    for folder_name, epoch in EPOCH_DIRS:
        pred_dir = PRED_ROOT / folder_name
        if not pred_dir.exists():
            print(f"WARNING: no predictions in {pred_dir}, skipping epoch {epoch}")
            continue
        r = evaluate_epoch(pred_dir, test_files, class_map, positive_class="malignant")
        rb = evaluate_epoch(pred_dir, test_files, class_map, positive_class="benign")
        rc = evaluate_epoch(pred_dir, test_files, class_map, positive_class="combined")
        epochs.append(epoch)
        for key in METRICS:
            metrics[key].append(r[key])
            benign_metrics[key].append(rb[key])
            combined_metrics[key].append(rc[key])

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal tables ---
    header = "Epoch | CF1 | CPrec | CRec | CFP"
    sep = "------|------|-------|------|-----"
    for title, table in [("All masses", combined_metrics),
                         ("Malignant", metrics),
                         ("Benign", benign_metrics)]:
        print(f"\n{title} — {header}")
        print(f"{'':>7}  {sep}")
        for i, ep in enumerate(epochs):
            print(f"{ep:>5} | {table['case_f1'][i]:.3f} | "
                  f"{table['case_precision'][i]:.3f} | "
                  f"{table['case_recall'][i]:.3f} | "
                  f"{table['case_fp_rate'][i]:.3f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_centroid_detection.json"
    payload = {
        "noise_floor_px": MIN_PRED_AREA,
        "centroid_definition": "predicted centroid within 0.5x GT equivalent diameter",
        "seed": 42,
        "images": 625,
        "checkpoint_selection": "joint",
        "epochs": epochs,
        "combined": {key: combined_metrics[key] for key in METRICS},
        "malignant": {key: metrics[key] for key in METRICS},
        "benign": {key: benign_metrics[key] for key in METRICS},
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# Centroid detection vs. epoch",
        "",
        f"Single preliminary milestones run: seed 42, 625 images, joint-selected "
        f"checkpoints, centroid detection = predicted centroid within 0.5x GT "
        f"equivalent diameter, noise floor {MIN_PRED_AREA} px.",
        "",
        "A mass case is detected when the closest retained predicted centroid "
        "lies within half the ground-truth mass's equivalent circular diameter. "
        "A Normal case with any retained prediction is a false alarm.",
        "",
        "No error bars (single seed).",
        "",
    ]

    for title, table in [("All masses", combined_metrics),
                         ("Malignant masses", metrics),
                         ("Benign masses", benign_metrics)]:
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append("| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |")
        md_lines.append("|------:|--------:|----------:|---------:|-------------:|")
        for i, ep in enumerate(epochs):
            md_lines.append(
                f"| {ep} | {table['case_f1'][i]:.3f} | "
                f"{table['case_precision'][i]:.3f} | "
                f"{table['case_recall'][i]:.3f} | "
                f"{table['case_fp_rate'][i]:.3f} |"
            )
        md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_centroid_detection.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for src, key, label, color in PLOT_SERIES:
        data = {
            "combined": combined_metrics,
            "malignant": metrics,
            "benign": benign_metrics,
        }[src][key]
        ax.plot(epochs, data, '-o', color=color, label=label,
                markersize=7, linewidth=1.5, markeredgecolor='white',
                markeredgewidth=1.5, zorder=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Centroid recall')
    ax.set_xlim(left=0, right=epochs[-1])
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.grid(False, axis='x')

    for sp in ax.spines.values():
        sp.set_color('#c0c0c0')
    ax.tick_params(axis='both', which='both', color='#c0c0c0',
                   labelcolor='#999999')

    ax.legend(loc='best', frameon=True, fancybox=False, borderpad=0.8,
              handlelength=2.5)

    plt.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'epoch_convergence_by_centroid_detection.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
