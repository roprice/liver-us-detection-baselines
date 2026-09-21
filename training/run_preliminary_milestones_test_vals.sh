#!/bin/bash
set -e

# Predict milestone checkpoints on the fold-0 validation images.
#
# This confirms the epoch-convergence plateau observed on the test set
# using an independent evaluation surface. The validation images were
# held out from gradient updates during training (nnU-Net uses them
# only for pseudo-Dice logging and best-checkpoint selection).
#
# Excluded checkpoints:
#   - checkpoint_best.pth: selected on val pseudo-Dice, so it would
#     appear artificially optimistic on this same image set.
#   - checkpoint_best_mass.pth: same selection mechanism (EMA mass Dice).
#
# Prerequisites:
#   - Completed training from run_preliminary_milestones_test.sh
#   - splits_final.json in nnUNet_preprocessed/Dataset001_AUL/
#   - All milestone .pth files present in nnUNet_results
#
# Usage:
#   bash training/run_preliminary_milestones_val.sh 2>&1 | tee preliminary_milestones_val.log

START_TIME=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"

EXPERIMENT_NAME=""

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

DATASET_ID=1
DATASET_NAME="Dataset001_AUL"
SEEDS=(42 43 44)
FOLD=0

# Milestone checkpoints only. Best and best_mass are excluded because
# they were selected using val-set pseudo-Dice and would be biased.
CHECKPOINT_ORDER=(
    checkpoint_epoch50.pth
    checkpoint_epoch100.pth
    checkpoint_epoch150.pth
    checkpoint_epoch300.pth
    checkpoint_epoch500.pth
    checkpoint_epoch750.pth
    checkpoint_final.pth
)
declare -A CHECKPOINT_LABELS
CHECKPOINT_LABELS[checkpoint_epoch50.pth]=epoch50
CHECKPOINT_LABELS[checkpoint_epoch100.pth]=epoch100
CHECKPOINT_LABELS[checkpoint_epoch150.pth]=epoch150
CHECKPOINT_LABELS[checkpoint_epoch300.pth]=epoch300
CHECKPOINT_LABELS[checkpoint_epoch500.pth]=epoch500
CHECKPOINT_LABELS[checkpoint_epoch750.pth]=epoch750
CHECKPOINT_LABELS[checkpoint_final.pth]=final

# --- Log directories ---
LOGS_DIR="${REPO_DIR}/logs/val_predictions"
mkdir -p "$LOGS_DIR"

PRED_TIMES_CSV="${LOGS_DIR}/val_prediction_times.csv"
echo "size,seed,checkpoint,wall_clock_seconds,case_count" > "$PRED_TIMES_CSV"

# =====================================================================
# Build validation image directory from the fold-0 split
# =====================================================================
SPLITS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"
if [ ! -f "$SPLITS_FILE" ]; then
    echo "ERROR: splits_final.json not found at ${SPLITS_FILE}"
    echo "Training must complete before running this script."
    exit 1
fi

VAL_INPUT_DIR="${REPO_DIR}/tmp/val_images_fold${FOLD}"
rm -rf "$VAL_INPUT_DIR"
mkdir -p "$VAL_INPUT_DIR"

echo "=== Validation predictions (milestone epoch-convergence confirmation) ==="
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Extract fold-0 val case IDs and symlink their images into a flat
# directory that nnUNetv2_predict can consume.
echo "--- Building val image directory from fold ${FOLD} split ---"
VAL_COUNT=$(python -c "
import json, sys, os
from pathlib import Path

splits = json.load(open('${SPLITS_FILE}'))
val_ids = splits[${FOLD}]['val']
raw_dir = Path('${nnUNet_raw}/${DATASET_NAME}/imagesTr')
out_dir = Path('${VAL_INPUT_DIR}')

linked = 0
for case_id in sorted(val_ids):
    src = raw_dir / f'{case_id}_0000.png'
    if src.exists():
        dst = out_dir / f'{case_id}_0000.png'
        dst.symlink_to(src.resolve())
        linked += 1
    else:
        print(f'WARNING: {src} not found', file=sys.stderr)

print(linked)
")
echo "Linked ${VAL_COUNT} validation images into ${VAL_INPUT_DIR}"
echo ""

if [ "$VAL_COUNT" -eq 0 ]; then
    echo "ERROR: No validation images found. Check splits_final.json and imagesTr."
    exit 1
fi

# =====================================================================
# Prediction loop
# =====================================================================
for SEED in "${SEEDS[@]}"; do
    TRAINER="nnUNetTrainerMilestones_seed${SEED}"
    CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__2d/fold_0"

    echo ""
    echo "========================================"
    echo "=== Seed ${SEED} ==="
    echo "========================================"

    for CHK in "${CHECKPOINT_ORDER[@]}"; do
        LBL="${CHECKPOINT_LABELS[$CHK]}"

        if [ ! -f "${CKPT_DIR}/${CHK}" ]; then
            echo "SKIP: ${CHK} not found (seed ${SEED})"
            continue
        fi

        PRED_DIR="${nnUNet_results}/predictions_milestones_val_625images_seed${SEED}_${LBL}"
        echo ""
        echo "--- Predicting (val): ${LBL} (${CHK}, seed ${SEED}) ---"
        echo "Predict start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        PRED_START=$(date +%s)
        /usr/bin/time -v -o "${LOGS_DIR}/time_predict_val_s${SEED}_${LBL}.txt" \
            nnUNetv2_predict \
                -i "$VAL_INPUT_DIR" \
                -o "$PRED_DIR" \
                -d $DATASET_ID -c 2d -f 0 \
                -tr $TRAINER \
                -chk "$CHK"
        PRED_END=$(date +%s)
        PRED_SECS=$(( PRED_END - PRED_START ))
        CASE_COUNT=$(ls "$PRED_DIR"/*.png 2>/dev/null | wc -l)
        echo "Predict end: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "Prediction: ${PRED_SECS}s, ${CASE_COUNT} cases"
        echo "625,${SEED},${LBL},${PRED_SECS},${CASE_COUNT}" >> "$PRED_TIMES_CSV"
    done

    echo ""
    echo "=== Seed ${SEED} complete ==="
done

# =====================================================================
# Summary
# =====================================================================
END_TIME=$(date +%s)
echo ""
echo "=== Validation predictions complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Total wall clock: $(( END_TIME - START_TIME ))s ($(( (END_TIME - START_TIME) / 60 ))m)"
echo ""
echo "=== Output ==="
echo "  ${PRED_TIMES_CSV}"
echo ""
echo "  Prediction directories:"
for SEED in "${SEEDS[@]}"; do
    for CHK in "${CHECKPOINT_ORDER[@]}"; do
        LBL="${CHECKPOINT_LABELS[$CHK]}"
        echo "    predictions_milestones_val_625images_seed${SEED}_${LBL}/"
    done
done
echo ""
echo "  Timing logs: ${LOGS_DIR}/time_predict_val_s{SEED}_{label}.txt"
echo ""
echo "  Temp val image directory (symlinks, safe to delete):"
echo "    ${VAL_INPUT_DIR}"
echo ""
echo "=== Notes ==="
echo "  checkpoint_best and checkpoint_best_mass are excluded because they"
echo "  were selected using val-set pseudo-Dice during training."
echo "  7 checkpoints × 3 seeds = 21 prediction directories."
echo ""
echo "  Val images are the fold-${FOLD} validation split from splits_final.json."
echo "  These images were held out from gradient updates during training."
echo "  nnU-Net used them only for pseudo-Dice monitoring and best-checkpoint"
echo "  selection. The milestone checkpoints (50-750) and final were not"
echo "  selected based on val performance, so val is an unbiased surface"
echo "  for comparing them."