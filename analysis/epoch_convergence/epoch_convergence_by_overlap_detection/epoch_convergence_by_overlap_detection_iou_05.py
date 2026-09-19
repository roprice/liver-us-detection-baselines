"""Overlap detection quality vs. epoch for the preliminary milestones run
(seed 42, 625 images) at IoU >= 0.5.

Evaluates the six milestone checkpoints (50/100/150/300/500/750) on mass
*detection* metrics using overlap-based matching with IoU >= 0.5, reported for
all masses combined and separately for malignant and benign masses.

Two views are reported per mass grouping, both per epoch:
  - Lesion-level: recall / precision / F1 over matched lesions in malignant cases.
  - Case-level (patient triage): recall / precision / F1 and false-positive rate.

Overlap detection matching (noise floor 100 px) is defined locally so the script
is self-contained.

This is a single-seed run (seed 42, 625 images, joint-selected checkpoints), so
there are no error bars.

Run from project root:
    python analysis/epoch_convergence/epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_05.py
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
# Overlap detection: IoU >= 0.5 (requires substantial overlap).
IOU_THRESHOLD = 0.5
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

# Metric key -> (legend label, color). All seven are computed and written to
# the JSON/Markdown artifacts for future investigation; only a subset is plotted.
METRICS = {
    "lesion_f1":        ("Lesion F1",        "#2a78d6"),
    "lesion_precision": ("Lesion precision", "#6aa7e8"),
    "lesion_recall":    ("Lesion recall",    "#9cc4f2"),
    "case_f1":          ("Case F1",          "#eb6834"),
    "case_precision":   ("Case precision",   "#f2a17e"),
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


def match_lesions(gt_mask, pred_mask, iou_threshold, min_pred_area):
    """Greedy one-to-one lesion matching at a given IoU threshold.

    Returns (tp, fp, fn). When iou_threshold == 0.0 this is overlap-based
    (any overlap counts as a match), since the >= threshold is 0.
    """
    gt_labeled = label(gt_mask)
    pred_labeled = label(pred_mask)
    n_gt = gt_labeled.max()
    n_pred = pred_labeled.max()

    # Filter small predicted components
    if n_pred > 0:
        keep = set()
        for p in range(1, n_pred + 1):
            if (pred_labeled == p).sum() >= min_pred_area:
                keep.add(p)
        if len(keep) < n_pred:
            filtered = np.zeros_like(pred_labeled)
            for i, p in enumerate(sorted(keep), 1):
                filtered[pred_labeled == p] = i
            pred_labeled = filtered
            n_pred = len(keep)

    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0

    if n_gt == 0:
        return 0, n_pred, 0

    if n_pred == 0:
        return 0, 0, n_gt

    # IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for g in range(1, n_gt + 1):
        gt_r = (gt_labeled == g)
        for p in range(1, n_pred + 1):
            pred_r = (pred_labeled == p)
            inter = np.logical_and(gt_r, pred_r).sum()
            union = np.logical_or(gt_r, pred_r).sum()
            iou_matrix[g - 1, p - 1] = inter / union if union > 0 else 0.0

    # Greedy matching. With iou_threshold == 0.0, overlap-based: require any
    # overlap (IoU > 0), not IoU >= 0 (which would also match disjoint pairs).
    matched_gt, matched_pred = set(), set()
    tp = 0
    if iou_threshold == 0.0:
        pairs = [(iou_matrix[g, p], g, p) for g in range(n_gt) for p in range(n_pred)
                 if iou_matrix[g, p] > 0.0]
    else:
        pairs = [(iou_matrix[g, p], g, p) for g in range(n_gt) for p in range(n_pred)
                 if iou_matrix[g, p] >= iou_threshold]
    pairs.sort(reverse=True)

    for iou_val, g, p in pairs:
        if g not in matched_gt and p not in matched_pred:
            matched_gt.add(g)
            matched_pred.add(p)
            tp += 1

    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)

    return tp, fp, fn


def evaluate_run(pred_dir, test_files, class_map, min_pred_area,
                 iou_threshold=0.0, positive_class="malignant"):
    """Evaluate one seed/size run at a given IoU threshold (default overlap).

    positive_class selects which mass-bearing cases are treated as the positive
    class (``malignant`` or ``benign``); normal cases are always the negative
    class. Masses in the other (non-positive) mass-bearing class are ignored.

    Returns (lesion, case) where:
      lesion = (tp, fp, fn) at lesion level over positive-class cases
      case   = (tp, fp, fn, n_normal) at patient level:
               tp = positive case with an overlapping prediction
               fn = positive case with no overlapping prediction
               fp = normal case (no mass) flagged by the model
               n_normal = number of normal (mass-free) cases evaluated
    """
    lesion_tp = lesion_fp = lesion_fn = 0
    case_tp = case_fn = 0
    case_fp = 0  # false alarms on normal cases
    n_normal = 0

    for fname in test_files:
        cls = class_map.get(fname, "unknown")
        gt = np.array(Image.open(LABELS_TS / fname))
        pred_path = pred_dir / fname
        if not pred_path.exists():
            continue
        pred = np.array(Image.open(pred_path))
        gt_mass = (gt == MASS_VALUE).astype(np.uint8)
        pred_mass = (pred == MASS_VALUE).astype(np.uint8)

        # Filter predicted components (speckle), for both the mass region used
        # in overlap detection and the component count on normals.
        pred_labeled = label(pred_mass)
        n_pred = pred_labeled.max()
        keep = set()
        for p in range(1, n_pred + 1):
            if (pred_labeled == p).sum() >= min_pred_area:
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
            tp, fp, fn = match_lesions(gt_mass, pred_mass, iou_threshold, min_pred_area)
            lesion_tp += tp
            lesion_fp += fp
            lesion_fn += fn

            # Case-level: any retained prediction overlapping the GT mass counts
            # as detected (overlap), or IoU >= iou_threshold when a threshold is
            # supplied.
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

    lesion = (int(lesion_tp), int(lesion_fp), int(lesion_fn))
    case = (int(case_tp), int(case_fp), int(case_fn), int(n_normal))
    return lesion, case


def evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=0.0,
                   positive_class="malignant"):
    """Run detection eval for one checkpoint, return the 7 metrics."""
    lesion, case = evaluate_run(pred_dir, test_files, class_map, MIN_PRED_AREA,
                                iou_threshold=iou_threshold,
                                positive_class=positive_class)

    ltp, lfp, lfn = lesion
    lprec = ltp / (ltp + lfp) if (ltp + lfp) > 0 else 0.0
    lrec = ltp / (ltp + lfn) if (ltp + lfn) > 0 else 0.0
    lf1 = 2 * lprec * lrec / (lprec + lrec) if (lprec + lrec) > 0 else 0.0

    ctp, cfp, cfn, n_normal = case
    cprec = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
    crec = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
    cf1 = 2 * cprec * crec / (cprec + crec) if (cprec + crec) > 0 else 0.0
    cfp_rate = cfp / n_normal if n_normal > 0 else 0.0

    return {
        "lesion_f1": lf1,
        "lesion_precision": lprec,
        "lesion_recall": lrec,
        "case_f1": cf1,
        "case_precision": cprec,
        "case_recall": crec,
        "case_fp_rate": cfp_rate,
    }


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
        r = evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=iou_threshold,
                           positive_class="malignant")
        rb = evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=iou_threshold,
                            positive_class="benign")
        rc = evaluate_epoch(pred_dir, test_files, class_map, iou_threshold=iou_threshold,
                            positive_class="combined")
        epochs.append(epoch)
        for key in METRICS:
            metrics[key].append(r[key])
            benign_metrics[key].append(rb[key])
            combined_metrics[key].append(rc[key])

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal table (all masses) ---
    print("\nAll masses — Epoch | LF1 | LPrec | LRec | CF1 | CPrec | CRec | CFP")
    print("-------------------|------|-------|------|------|-------|------|-----")
    for i, ep in enumerate(epochs):
        print(f"{ep:>18} | {combined_metrics['lesion_f1'][i]:.3f} | "
              f"{combined_metrics['lesion_precision'][i]:.3f} | "
              f"{combined_metrics['lesion_recall'][i]:.3f} | "
              f"{combined_metrics['case_f1'][i]:.3f} | "
              f"{combined_metrics['case_precision'][i]:.3f} | "
              f"{combined_metrics['case_recall'][i]:.3f} | "
              f"{combined_metrics['case_fp_rate'][i]:.3f}")

    # --- Terminal table (malignant) ---
    print("\nMalignant — Epoch | LF1 | LPrec | LRec | CF1 | CPrec | CRec | CFP")
    print("------------------|------|-------|------|------|-------|------|-----")
    for i, ep in enumerate(epochs):
        print(f"{ep:>18} | {metrics['lesion_f1'][i]:.3f} | "
              f"{metrics['lesion_precision'][i]:.3f} | "
              f"{metrics['lesion_recall'][i]:.3f} | "
              f"{metrics['case_f1'][i]:.3f} | "
              f"{metrics['case_precision'][i]:.3f} | "
              f"{metrics['case_recall'][i]:.3f} | "
              f"{metrics['case_fp_rate'][i]:.3f}")

    # --- Terminal table (benign) ---
    print("\nBenign   — Epoch | LF1 | LPrec | LRec | CF1 | CPrec | CRec | CFP")
    print("------------------|------|-------|------|------|-------|------|-----")
    for i, ep in enumerate(epochs):
        print(f"{ep:>18} | {benign_metrics['lesion_f1'][i]:.3f} | "
              f"{benign_metrics['lesion_precision'][i]:.3f} | "
              f"{benign_metrics['lesion_recall'][i]:.3f} | "
              f"{benign_metrics['case_f1'][i]:.3f} | "
              f"{benign_metrics['case_precision'][i]:.3f} | "
              f"{benign_metrics['case_recall'][i]:.3f} | "
              f"{benign_metrics['case_fp_rate'][i]:.3f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_overlap_detection_iou_05.json"
    payload = {
        "noise_floor_px": MIN_PRED_AREA,
        "iou_threshold": iou_threshold,
        "detection_definition": definition,
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
        "# Overlap detection quality vs. epoch",
        "",
        f"Single preliminary milestones run: seed 42, 625 images, joint-selected "
        f"checkpoints, detection = {definition}, noise floor {MIN_PRED_AREA} px.",
        "",
        "No error bars (single seed).",
        "",
    ]

    md_lines.append("## All masses")
    md_lines.append("")
    md_lines.append("| Epoch | Lesion F1 | Lesion Prec | Lesion Rec | Case F1 | Case Prec | "
        "Case Rec | Case FP rate |")
    md_lines.append("|------:|----------:|------------:|-----------:|--------:|----------:|"
        "----------:|-------------:|")
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {combined_metrics['lesion_f1'][i]:.3f} | "
            f"{combined_metrics['lesion_precision'][i]:.3f} | "
            f"{combined_metrics['lesion_recall'][i]:.3f} | "
            f"{combined_metrics['case_f1'][i]:.3f} | "
            f"{combined_metrics['case_precision'][i]:.3f} | "
            f"{combined_metrics['case_recall'][i]:.3f} | "
            f"{combined_metrics['case_fp_rate'][i]:.3f} |"
        )
    md_lines.append("")

    md_lines.append("## Malignant masses")
    md_lines.append("")
    md_lines.append("| Epoch | Lesion F1 | Lesion Prec | Lesion Rec | Case F1 | Case Prec | "
        "Case Rec | Case FP rate |")
    md_lines.append("|------:|----------:|------------:|-----------:|--------:|----------:|"
        "----------:|-------------:|")
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {metrics['lesion_f1'][i]:.3f} | "
            f"{metrics['lesion_precision'][i]:.3f} | "
            f"{metrics['lesion_recall'][i]:.3f} | "
            f"{metrics['case_f1'][i]:.3f} | "
            f"{metrics['case_precision'][i]:.3f} | "
            f"{metrics['case_recall'][i]:.3f} | "
            f"{metrics['case_fp_rate'][i]:.3f} |"
        )
    md_lines.append("")

    md_lines.append("## Benign masses")
    md_lines.append("")
    md_lines.append("| Epoch | Lesion F1 | Lesion Prec | Lesion Rec | Case F1 | Case Prec | "
        "Case Rec | Case FP rate |")
    md_lines.append("|------:|----------:|------------:|-----------:|--------:|----------:|"
        "----------:|-------------:|")
    for i, ep in enumerate(epochs):
        md_lines.append(
            f"| {ep} | {benign_metrics['lesion_f1'][i]:.3f} | "
            f"{benign_metrics['lesion_precision'][i]:.3f} | "
            f"{benign_metrics['lesion_recall'][i]:.3f} | "
            f"{benign_metrics['case_f1'][i]:.3f} | "
            f"{benign_metrics['case_precision'][i]:.3f} | "
            f"{benign_metrics['case_recall'][i]:.3f} | "
            f"{benign_metrics['case_fp_rate'][i]:.3f} |"
        )
    md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_overlap_detection_iou_05.md"
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
        out = OUT_DIR / f'epoch_convergence_by_overlap_detection_iou_05.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
