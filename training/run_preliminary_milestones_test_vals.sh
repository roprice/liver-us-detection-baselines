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

EXPERIMENT_NAME="milestones_pilot_vals"

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
LOGS_DIR="${REPO_DIR}/experiment_logs/${EXPERIMENT_NAME}/logs"
VAL_PRED_LOGS="${LOGS_DIR}/val_predictions"
BENCH_DIR="${LOGS_DIR}/inference"
mkdir -p "$VAL_PRED_LOGS" "$BENCH_DIR"

PRED_TIMES_CSV="${VAL_PRED_LOGS}/val_prediction_times.csv"
echo "size,seed,checkpoint,wall_clock_seconds,case_count" > "$PRED_TIMES_CSV"

# --- GPU monitor cleanup on exit ---
GPU_MONITOR_PID=""
cleanup_gpu_monitor() {
    if [ -n "$GPU_MONITOR_PID" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}
trap cleanup_gpu_monitor EXIT

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
# Environment
# =====================================================================
echo "=== Environment ==="
echo "Host: $(hostname)"
echo "OS: $(uname -s) $(uname -r)"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo "GPU count: ${GPU_COUNT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi query failed"

echo "CPU: $(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:\s*//' || echo 'unknown')"
echo "CPU cores: $(nproc 2>/dev/null || echo 'unknown')"
echo "RAM total: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo 'unknown')"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" \
    || echo "WARNING: could not determine PyTorch version"
python -c "import torch; print(f'GPUs visible to PyTorch: {torch.cuda.device_count()}')" \
    || echo "WARNING: could not query PyTorch GPU count"
python -c "from importlib.metadata import version; print(f'nnU-Net {version(\"nnunetv2\")}')" \
    || echo "WARNING: could not determine nnU-Net version"

# nnU-Net version and source revision.
NNUNET_VERSION=$(python -c "from importlib.metadata import version; print(version('nnunetv2'))" 2>/dev/null || echo "unknown")
NNUNET_PATH=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])" 2>/dev/null || echo "")
if [ -n "$NNUNET_PATH" ]; then
    NNUNET_SHA=$(git -C "$NNUNET_PATH" rev-parse HEAD 2>/dev/null || echo "n/a (pip install)")
else
    NNUNET_SHA="n/a (could not locate install path)"
fi
echo "nnU-Net version: ${NNUNET_VERSION}"
echo "nnU-Net source SHA: ${NNUNET_SHA}"

# This repo's revision
REPO_SHA=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "not a git repo")
echo "Repo SHA: ${REPO_SHA}"

echo ""
echo "Seeds: ${SEEDS[*]}"
echo "Trainer: nnUNetTrainerMilestones_seed{42,43,44}"
echo "Configuration: 2d, fold ${FOLD}, PlainConvUNet"
echo ""

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

    # --- Start GPU monitoring (prediction) ---
    GPU_MONITOR_LOG="${LOGS_DIR}/gpu_monitor_val_s${SEED}.csv"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
        --format=csv \
        --loop-ms=1000 \
        > "$GPU_MONITOR_LOG" 2>&1 &
    GPU_MONITOR_PID=$!
    echo "GPU monitor started (PID ${GPU_MONITOR_PID}, log: ${GPU_MONITOR_LOG})"

    # --- Checkpoint file sizes ---
    echo ""
    echo "--- Checkpoint file sizes (seed ${SEED}) ---"
    ls -lh "${CKPT_DIR}"/*.pth 2>/dev/null || echo "WARNING: no checkpoint files found"
    echo ""

    # --- Model footprint ---
    echo "--- Model footprint (seed ${SEED}) ---"
    python -c "
import torch, os, sys
ckpt_dir = sys.argv[1]
final = os.path.join(ckpt_dir, 'checkpoint_final.pth')
if not os.path.exists(final):
    print('WARNING: checkpoint_final.pth not found', file=sys.stderr)
    sys.exit(0)
ckpt = torch.load(final, weights_only=False, map_location='cpu')
weights = ckpt.get('network_weights', {})
num_params = sum(v.numel() for v in weights.values())
print(f'Parameters: {num_params:,}')
print(f'Checkpoint file size: {os.path.getsize(final):,} bytes')
tmp = '/tmp/weights_only.pth'
torch.save(weights, tmp)
print(f'Weights-only file size: {os.path.getsize(tmp):,} bytes')
os.remove(tmp)
" "$CKPT_DIR" || echo "WARNING: could not compute model footprint"
    echo ""

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
        /usr/bin/time -v -o "${VAL_PRED_LOGS}/time_predict_val_s${SEED}_${LBL}.txt" \
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

    # --- Stop GPU monitoring ---
    echo ""
    cleanup_gpu_monitor
    echo "GPU monitor stopped (seed ${SEED})"

    echo ""
    echo "=== Seed ${SEED} complete ==="
done

# =====================================================================
# GPU inference benchmark (per-image latency on the val split)
# =====================================================================
echo ""
echo "========================================"
echo "=== GPU inference benchmark (val) ==="
echo "========================================"

/usr/bin/time -v -o "${LOGS_DIR}/time_gpu_inference_benchmark_val.txt" \
    python "${SCRIPT_DIR}/benchmark_gpu_inference.py" \
        --nnunet-raw "${nnUNet_raw}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-id "${DATASET_ID}" \
        --images-dir "${VAL_INPUT_DIR}" \
        --seeds "${SEEDS[@]}" \
        --trainer-prefix nnUNetTrainerMilestones_seed \
        --checkpoint checkpoint_final.pth \
        --device cuda \
        --output-dir "$BENCH_DIR"

echo "GPU inference benchmark complete. Results in ${BENCH_DIR}/"
echo "To repeat on CPU later: same command with --device cpu (see"
echo "  ${BENCH_DIR}/inference_settings_cuda.json for exact settings)."

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
echo "  Timing logs: ${VAL_PRED_LOGS}/time_predict_val_s{SEED}_{label}.txt"
echo ""
echo "  Per-seed files:"
for S in "${SEEDS[@]}"; do
    echo "    gpu_monitor_val_s${S}.csv         GPU utilization/memory/power/temp samples (~1/s)"
done
echo ""
echo "  GPU inference benchmark (${BENCH_DIR}):"
echo "    inference_per_image_cuda.csv       per-image inference seconds, one row per case x seed"
echo "    inference_summary_cuda.csv         per-seed model load time, median/mean/stdev/IQR/range"
echo "    inference_settings_cuda.json       exact settings to replicate on CPU later"
echo "    ${LOGS_DIR}/time_gpu_inference_benchmark_val.txt   /usr/bin/time: peak RSS, CPU time for the benchmark script"
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
