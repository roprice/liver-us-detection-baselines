"""False positives vs. epoch for the preliminary milestones run (seed 42, 625 images).

Reports, per milestone checkpoint, how many Normal (mass-free) test cases the
model flags as containing a mass. On this dataset every image is one patient and
holds at most one mass, so a "false positive" collapses to a single case-level
signal: a Normal case where the model predicts any retained mass.

Two views per epoch:
  - Raw false-alarm count over the Normal test cases.
  - False-alarm rate (count / number of Normal cases).

A Normal case has no ground-truth mass, so there is no overlap to threshold
against: any retained predicted mass is a false alarm. The IoU threshold from the
detection script therefore does not apply here; only the 100 px noise floor does.

Single-seed run (seed 42, 625 images, joint-selected checkpoints); no error bars.

Run from project root:
    python analysis/epoch_convergence_by_false_positives/epoch_convergence_by_false_positives.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

FP_COUNT_COLOR = "#eb6834"
FP_RATE_COLOR = "#2a78d6"


def get_class_map():
    """Build filename -> class mapping from case_mapping.json (test split)."""
    if not CASE_MAPPING.exists():
        print(f"WARNING: {CASE_MAPPING} not found")
        return {}
    with open(CASE_MAPPING) as f:
        mapping = json.load(f)
    return {e["case_name"] + ".png": e["category"].lower()
            for e in mapping if e["split"] == "test"}


def retained_mass_present(pred_mask, min_pred_area):
    """Whether the prediction has any mass component >= min_pred_area.

    Applies the same speckle filter as the detection script: only a retained
    (>= noise floor) predicted mass component raises a false alarm.
    """
    pred_mass = (pred_mask == MASS_VALUE).astype(np.uint8)
    pred_labeled = label(pred_mass)
    n_pred = pred_labeled.max()
    for p in range(1, n_pred + 1):
        if (pred_labeled == p).sum() >= min_pred_area:
            return True
    return False


def evaluate_false_positives(pred_dir, normal_files, min_pred_area):
    """Count false alarms (Normal cases flagged with a retained mass).

    Returns (fp_count, n_normal).
    """
    fp_count = 0
    for fname in normal_files:
        pred_path = pred_dir / fname
        if not pred_path.exists():
            continue
        pred = np.array(Image.open(pred_path))
        if retained_mass_present(pred, min_pred_area):
            fp_count += 1
    return fp_count, len(normal_files)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    class_map = get_class_map()
    if not LABELS_TS.exists():
        print(f"ERROR: {LABELS_TS} not found")
        return

    normal_files = sorted(
        f for f in os.listdir(LABELS_TS)
        if f.endswith(".png") and class_map.get(f) == "normal"
    )
    print(f"Normal test cases: {len(normal_files)}, noise floor {MIN_PRED_AREA} px")

    epochs = []
    fp_counts = []
    fp_rates = []
    for folder_name, epoch in EPOCH_DIRS:
        pred_dir = PRED_ROOT / folder_name
        if not pred_dir.exists():
            print(f"WARNING: no predictions in {pred_dir}, skipping epoch {epoch}")
            continue
        fp_count, n_normal = evaluate_false_positives(
            pred_dir, normal_files, MIN_PRED_AREA)
        rate = fp_count / n_normal if n_normal > 0 else 0.0
        epochs.append(epoch)
        fp_counts.append(fp_count)
        fp_rates.append(rate)

    if not epochs:
        print("No per-epoch predictions found, exiting")
        return

    # --- Terminal table ---
    print("\nEpoch | FP count | FP rate")
    print("------|----------|---------")
    for i, ep in enumerate(epochs):
        print(f"{ep:>5} | {fp_counts[i]:>8} | {fp_rates[i]:.4f}")

    # --- JSON (source of truth) ---
    json_path = OUT_DIR / "epoch_convergence_by_false_positives.json"
    payload = {
        "noise_floor_px": MIN_PRED_AREA,
        "seed": 42,
        "images": 625,
        "normal_cases": len(normal_files),
        "epochs": epochs,
        "fp_count": fp_counts,
        "fp_rate": fp_rates,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # --- Markdown ---
    md_lines = [
        "# False positives vs. epoch",
        "",
        f"Single preliminary milestones run: seed 42, 625 images, joint-selected "
        f"checkpoints, noise floor {MIN_PRED_AREA} px.",
        "",
        "A false positive is a Normal (mass-free) case the model flags as "
        "containing a mass. Every image is one patient with at most one mass, so "
        "this is a single case-level signal.",
        "",
        "No error bars (single seed).",
        "",
        f"Normal test cases: {len(normal_files)}.",
        "",
        "| Epoch | False positives | FP rate |",
        "|------:|----------------:|--------:|",
    ]
    for i, ep in enumerate(epochs):
        md_lines.append(f"| {ep} | {fp_counts[i]} | {fp_rates[i]:.4f} |")
    md_lines.append("")
    md_path = OUT_DIR / "epoch_convergence_by_false_positives.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot: FP count (left) and rate (right) on twin axes ---
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(epochs, fp_counts, '-o', color=FP_COUNT_COLOR, label='False positives',
            markersize=7, linewidth=1.5, markeredgecolor='white',
            markeredgewidth=1.5, zorder=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('False positives (count)', color=FP_COUNT_COLOR)
    ax.set_xlim(left=0, right=epochs[-1])
    ax.set_xticks(epochs)
    ax.tick_params(axis='y', labelcolor=FP_COUNT_COLOR)
    ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
    ax.grid(False, axis='x')

    ax2 = ax.twinx()
    ax2.plot(epochs, fp_rates, '-o', color=FP_RATE_COLOR, label='FP rate',
             markersize=7, linewidth=1.5, markeredgecolor='white',
             markeredgewidth=1.5, zorder=3)
    ax2.set_ylabel('False-positive rate', color=FP_RATE_COLOR)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor=FP_RATE_COLOR)
    ax2.grid(False)

    for sp in list(ax.spines.values()) + list(ax2.spines.values()):
        sp.set_color('#c0c0c0')
    ax.tick_params(axis='x', color='#c0c0c0', labelcolor='#999999')

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc='best', frameon=True,
              fancybox=False, borderpad=0.8, handlelength=2.5)

    plt.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'epoch_convergence_by_false_positives.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
