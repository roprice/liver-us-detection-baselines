"""Overlap detection quality vs. epoch for the preliminary milestones run
(seed 42, 625 images) at IoU > 0.

Evaluates the six milestone checkpoints (50/100/150/300/500/750) on mass
*detection* metrics using overlap-based matching with IoU > 0 (any overlap),
reported for all masses combined and separately for malignant and benign masses.

Per mass grouping, reports case-level (patient triage) recall and
false-positive rate over Normal cases.

Overlap detection matching (noise floor 100 px) is defined locally so the script
is self-contained.

This is a single-seed run (seed 42, 625 images, joint-selected checkpoints), so
there are no error bars.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_00.py
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

# Script lives at analysis/epoch_convergence/epoch_convergence_by_overlap_detection/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# --- Detection config ---
LABELS_TS = PROJECT_ROOT / "nnUNet_raw/Dataset001_AUL/labelsTs"
CASE_MAPPING = PROJECT_ROOT / "nnUNet_raw/Dataset001_AUL/case_mapping.json"

MASS_VALUE = 2
# Overlap detection: IoU > 0 (any overlap) counts as a match. No clinically
# validated threshold exists for B-mode liver ultrasound, so this is the most
# permissive overlap criterion.
IOU_THRESHOLD = 0.0
# Fixed noise floor: 100 px^2 = ~10x10 px = ~1-2 mm (images cover ~12-20 cm
# at 1024 px, so ~0.12-0.20 mm/px). The smallest annotated mass is 398 px^2,
# so 100 px only removes speckle, never a true lesion.
MIN_PRED_AREA = 100

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

# Metric key -> (legend label, color).
METRICS = {
    "case_recall":      ("Case recall",      "#f5c0aa"),
    "case_fp_rate":     ("Case FP rate",     "#3b6d11"),
}

# Case-level overlap-based detection across all mass-present cases and by
# pathology. Combined is plotted first so the legend follows the table order.
PLOT_SERIES = [
    ("combined",  "case_recall", "Combined detection",  "#2a78d6"),
    ("malignant", "case_recall", "Malignant detection", "#eb6834"),
    ("benign",    "case_recall", "Benign detection",    "#3b6d11"),
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


def evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=0.0,
                   positive_class="malignant"):
    """Run detection eval for one checkpoint, return case recall and FP rate."""
    case_tp = case_fn = 0
    case_fp = 0
    n_normal = 0

    for fname in test_files:
        cls = class_map.get(fname, "unknown")
        pred_path = pred_dir / fname
        if not pred_path.exists():
            continue
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(LABELS_TS / fname))
        gt_mass = (gt == MASS_VALUE).astype(np.uint8)
        pred_mass = (pred == MASS_VALUE).astype(np.uint8)

        # Filter predicted components (speckle).
        pred_labeled = label(pred_mass)
        n_pred = pred_labeled.max()
        keep = set()
        for p in range(1, n_pred + 1):
            if (pred_labeled == p).sum() >= MIN_PRED_AREA:
                keep.add(p)
        filtered_pred = np.zeros_like(pred_mass)
        for p in keep:
            filtered_pred[pred_labeled == p] = 1

        is_positive = (
            cls in ("malignant", "benign")
            if positive_class == "combined"
            else cls == positive_class
        )
        if is_positive:
            # Case-level: any retained prediction overlapping the GT mass counts
            # as detected. With iou_threshold == 0.0 this is overlap (IoU > 0).
            n_gt = label(gt_mass).max()
            detected = False
            if n_gt > 0 and filtered_pred.sum() > 0:
                g = label(gt_mass)
                for gi in range(1, n_gt + 1):
                    gr = (g == gi)
                    inter = np.logical_and(gr, filtered_pred).sum()
                    union = np.logical_or(gr, filtered_pred).sum()
                    iou = inter / union if union > 0 else 0.0
                    meets = (iou > 0.0) if iou_threshold == 0.0 else (iou >= iou_threshold)
                    if meets:
                        detected = True
                        break
            if detected:
                case_tp += 1
            else:
                case_fn += 1
        elif cls == "normal":
            n_normal += 1
            # Any retained prediction on a mass-free image is a false alarm.
            if filtered_pred.sum() > 0:
                case_fp += 1

    case_recall = case_tp / (case_tp + case_fn) if (case_tp + case_fn) > 0 else 0.0
    case_fp_rate = case_fp / n_normal if n_normal > 0 else 0.0

    return case_recall, case_fp_rate


def main():
    iou_threshold = IOU_THRESHOLD
    definition = ("overlap (IoU > 0)" if iou_threshold == 0.0
                  else f"IoU >= {iou_threshold:g}")

    os.makedirs(OUT_DIR, exist_ok=True)

    class_map = get_class_map()
    if not LABELS_TS.exists():
        print(f"ERROR: {LABELS_TS} not found")
        return

    test_files = sorted(f for f in os.listdir(LABELS_TS) if f.endswith(".png"))
    print(f"Test set: {len(test_files)} files, detection = {definition}, "
          f"noise floor {MIN_PRED_AREA} px")

    epochs = []
    metrics = {key: [] for key in METRICS}
    benign_metrics = {key: [] for key in METRICS}
    combined_metrics = {key: [] for key in METRICS}
    for folder_name, epoch in EPOCH_DIRS:
        pred_dir = PRED_ROOT / folder_name
        if not pred_dir.exists():
            print(f"WARNING: no predictions in {pred_dir}, skipping epoch {epoch}")
            continue
        crec, cfp = evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=iou_threshold,
                                   positive_class="malignant")
        rb_rec, rb_fp = evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=iou_threshold,
                                       positive_class="benign")
        rc_rec, rc_fp = evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=iou_threshold,
                                       positive_class="combined")
        epochs.append(epoch)
        metrics["case_recall"].append(crec)
        metrics["case_fp_rate"].append(cfp)
        benign_metrics["case_recall"].append(rb_rec)
        benign_metrics["case_fp_rate"].append(rb_fp)
        combined_metrics["case_recall"].append(rc_rec)
        combined_metrics["case_fp_rate"].append(rc_fp)

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal table (all masses) ---
    print("\nAll masses — Epoch | CRec | CFP")
    print("-------------------|------|-----")
    for i, ep in enumerate(epochs):
        print(f"{ep:>18} | {combined_metrics['case_recall'][i]:.3f} | "
              f"{combined_metrics['case_fp_rate'][i]:.3f}")

    # --- Terminal table (malignant) ---
    print("\nMalignant — Epoch | CRec | CFP")
    print("------------------|------|-----")
    for i, ep in enumerate(epochs):
        print(f"{ep:>18} | {metrics['case_recall'][i]:.3f} | "
              f"{metrics['case_fp_rate'][i]:.3f}")

    # --- Terminal table (benign) ---
    print("\nBenign   — Epoch | CRec | CFP")
    print("------------------|------|-----")
    for i, ep in enumerate(epochs):
        print(f"{ep:>18} | {benign_metrics['case_recall'][i]:.3f} | "
              f"{benign_metrics['case_fp_rate'][i]:.3f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_overlap_detection_iou_00.json"
    payload = {
        "title": "Case-level overlap-based detection (IoU>0) by saved milestone epoch",
        "description": ("Single preliminary milestones test run: seed 42, 625 images, "
                        f"detection = {definition}, noise floor {MIN_PRED_AREA} px."),
        "noise_floor_px": MIN_PRED_AREA,
        "iou_threshold": iou_threshold,
        "detection_definition": definition,
        "seed": 42,
        "images": 625,
        "epochs": epochs,
        "combined": {
            "Detection": combined_metrics["case_recall"],
            "False positive rate": combined_metrics["case_fp_rate"],
        },
        "malignant": {
            "Detection": metrics["case_recall"],
            "False positive rate": metrics["case_fp_rate"],
        },
        "benign": {
            "Detection": benign_metrics["case_recall"],
            "False positive rate": benign_metrics["case_fp_rate"],
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# Case-level overlap-based detection (IoU>0) by saved milestone epoch",
        "",
        f"Single preliminary milestones test run: seed 42, 625 images, "
        f"detection = {definition}, noise floor {MIN_PRED_AREA} px.",
        "",
        "No error bars (single seed).",
        "",
    ]

    md_lines.append("## All masses")
    md_lines.append("")
    md_lines.append("| Epoch | Detection | False positive rate |")
    md_lines.append("|------:|----------:|-------------------:|")
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {combined_metrics['case_recall'][i]:.3f} | "
            f"{combined_metrics['case_fp_rate'][i]:.3f} |"
        )
    md_lines.append("")

    md_lines.append("## Malignant masses")
    md_lines.append("")
    md_lines.append("| Epoch | Detection | False positive rate |")
    md_lines.append("|------:|----------:|-------------------:|")
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {metrics['case_recall'][i]:.3f} | "
            f"{metrics['case_fp_rate'][i]:.3f} |"
        )
    md_lines.append("")

    md_lines.append("## Benign masses")
    md_lines.append("")
    md_lines.append("| Epoch | Detection | False positive rate |")
    md_lines.append("|------:|----------:|-------------------:|")
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {benign_metrics['case_recall'][i]:.3f} | "
            f"{benign_metrics['case_fp_rate'][i]:.3f} |"
        )
    md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_overlap_detection_iou_00.md"
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
    ax.set_ylabel('Detection rate')
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
        out = OUT_DIR / f'epoch_convergence_by_overlap_detection_iou_00.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
