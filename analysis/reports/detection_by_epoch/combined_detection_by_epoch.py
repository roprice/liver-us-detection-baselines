"""Combined (all-masses) detection vs. epoch, seven criteria on one chart.

Plots combined case-level recall across two detection families plus triage,
so the permissiveness ladder can be read off against epoch in one view:

  - triage:          any retained predicted mass anywhere (no overlap required)
  - overlap (IoU):   IoU > 0, IoU > 0.2, IoU > 0.5 (per-component)
  - centroid (deq):  1.0x, 0.5x, 0.25x the GT equivalent diameter

The values are read from the per-criterion JSON artifacts already produced by
the individual analysis scripts (recomputation would duplicate their logic).
Combined detection is the case-level recall over all mass-present cases.

Three seeds (42/43/44); error bars are the across-seed std.

Run from project root:
    python analysis/reports/detection_by_epoch/combined_detection_by_epoch.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUT_DIR = SCRIPT_DIR

VIEW = "combined"
OUT_BASE = "combined_detection_by_epoch"

# Epoch at which ``best`` / ``best_mass`` were last selected, per seed. Recovered
# from each seed's training log (the epoch of the final EMA-improving update).
BEST_EPOCHS = {
    ("best", 42): 838,
    ("best", 43): 695,
    ("best", 44): 999,
    ("best_mass", 42): 838,
    ("best_mass", 43): 708,
    ("best_mass", 44): 999,
}

# Snapshot rows for the "selected and final" table, in display order.
SNAPSHOTS = [
    ("best", "Best"),
    ("best_mass", "Best mass"),
    ("final", "1000 (Final)"),
]

# Root holding the individual per-criterion analysis outputs.
ANALYSIS_ROOT = PROJECT_ROOT / "analysis/epoch_convergence"

# Criterion -> (source JSON path, legend label, color, line style). Ordered
# triage first, then the IoU family (lenient -> strict) and the centroid family
# (lenient -> strict). Each family shares one color, varying only the line style.
CRITERIA = [
    ("triage",   "epoch_convergence_by_triage_detection/epoch_convergence_by_triage_detection.json",
     "Triage",                "#2a78d6", "-"),
    ("iou_0",    "epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_00.json",
     "Overlap (IoU>0.0)",     "#eb6834", "-"),
    ("iou_02",   "epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_02.json",
     "Overlap (IoU>0.2)",     "#eb6834", "--"),
    ("iou_05",   "epoch_convergence_by_overlap_detection/epoch_convergence_by_overlap_detection_iou_05.json",
     "Overlap (IoU>0.5)",     "#eb6834", ":"),
    ("deq_100",  "epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection_deq_100.json",
     "Centroid (1.0xD)",      "#3b6d11", "-"),
    ("deq_050",  "epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection_deq_050.json",
     "Centroid (0.5xD)",      "#3b6d11", "--"),
    ("deq_025",  "epoch_convergence_by_centroid_detection/epoch_convergence_by_centroid_detection_deq_025.json",
     "Centroid (0.25xD)",     "#3b6d11", ":"),
]

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


def load_recall(rel_path):
    """Return (epochs, mean, std) of case recall from a criterion JSON."""
    path = ANALYSIS_ROOT / rel_path
    with open(path) as f:
        data = json.load(f)
    det = data[VIEW]["Detection"]
    return data["epochs"], det["mean"], det["std"]


def load_snapshot(rel_path, suffix):
    """Return (mean, std) of case recall for one snapshot checkpoint, or None."""
    path = ANALYSIS_ROOT / rel_path
    with open(path) as f:
        data = json.load(f)
    snaps = data.get("snapshots", {}).get(VIEW, [])
    for e in snaps:
        if e["suffix"] == suffix:
            return e["detection"]["mean"], e["detection"]["std"]
    return None


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    epochs = None
    series = []  # (label, mean_list, std_list, color, linestyle)
    for key, rel_path, label, color, linestyle in CRITERIA:
        p = ANALYSIS_ROOT / rel_path
        if not p.exists():
            print(f"WARNING: {p} not found, skipping {key}")
            continue
        ep, mean, std = load_recall(rel_path)
        if epochs is None:
            epochs = ep
        elif ep != epochs:
            print(f"WARNING: epoch mismatch for {key}, expected {epochs}, got {ep}")
        series.append((label, mean, std, color, linestyle))

    if epochs is None:
        print("No criterion JSONs found, exiting")
        return

    # --- Terminal table ---
    header = f"{'Checkpoint':>10} | " + " | ".join(f"{l:>22}" for l, *_ in series)
    print(header)
    print("-" * len(header))
    for i, ep in enumerate(epochs):
        print(f"{ep:>5} | " + " | ".join(f"{fmt(m[i], s[i]):>22}" for _, m, s, *_ in series))

    # --- Markdown ---
    md_lines = [
        "# Combined detection vs. epoch",
        "",
        "Preliminary milestones run: seeds 42/43/44, 625 images each.",
        "",
        "Values are combined case-level recall (fraction of all mass-present "
        "cases detected) under seven criteria: triage, three overlap IoU tiers, "
        "and three centroid tolerance tiers. Mean ± standard deviation across seeds.",
        "",
        "**Predetermined saved checkpoints**",
        "| Checkpoint |" + "".join(f" {l} |" for l, *_ in series),
    ]
    md_lines.append("|------:|" + "".join("--------------:|" for _ in series))
    for i, ep in enumerate(epochs):
        md_lines.append(f"| {ep} |" +
                        "".join(f" {fmt(m[i], s[i])} |" for _, m, s, *_ in series))

    # Selected and final checkpoints table: one row per snapshot, one column per criterion.
    md_lines.extend([
        "",
        "**Selected and final checkpoints**",
        "| Checkpoint |" + "".join(f" {l} |" for l, *_ in series),
    ])
    md_lines.append("|------:|" + "".join("--------------:|" for _ in series))
    for suffix, display in SNAPSHOTS:
        cells = []
        for key, rel_path, label, color, linestyle in CRITERIA:
            snap = load_snapshot(rel_path, suffix)
            cells.append(fmt(snap[0], snap[1]) if snap else "—")
        md_lines.append(f"| {display} |" + "".join(f" {c} |" for c in cells))

    best_s = (f"`Best` epoch was {BEST_EPOCHS[('best', 42)]} for seed 42, "
              f"{BEST_EPOCHS[('best', 43)]} for seed 43, and {BEST_EPOCHS[('best', 44)]} for seed 44.")
    best_mass_s = (f"`Best mass` epoch was {BEST_EPOCHS[('best_mass', 42)]} for seed 42, "
                   f"{BEST_EPOCHS[('best_mass', 43)]} for seed 43, and {BEST_EPOCHS[('best_mass', 44)]} for seed 44.")
    md_lines.extend(["", best_s, best_mass_s, ""])

    md_path = OUT_DIR / f"{OUT_BASE}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {md_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, mean, std, color, linestyle in series:
        ax.errorbar(epochs, mean, yerr=std, fmt=linestyle + 'o', color=color,
                    label=label, markersize=7, linewidth=1.5, capsize=3,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Combined detection (case recall)')
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

    ax.legend(loc='lower right', frameon=True, fancybox=False, borderpad=0.8,
              handlelength=2.5)

    plt.tight_layout(pad=1.2)

    for ext in ('pdf', 'png'):
        out = OUT_DIR / f'{OUT_BASE}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None})
        print(f'Wrote {out}')
    plt.close(fig)


if __name__ == "__main__":
    main()
