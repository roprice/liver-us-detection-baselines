#!/bin/bash
set -e

# Preliminary experiment: train the 625-image set to 1000 epochs across
# 3 seeds (42, 43, 44), saving milestone checkpoints at 50/100/150/300/500/750,
# plus best_mass, best_joint, and final (epoch 1000).
# Then predict from all 9 checkpoints for each seed.
#
# Instrumentation:
#   - nvidia-smi GPU sampling (~1/s): utilization, memory, power, temperature
#   - /usr/bin/time -v: peak RSS and CPU time per command
#   - Per-checkpoint prediction timing and case count
#   - Checkpoint file sizes and model footprint (parameters, weights-only size)
#   - nnU-Net plans and dataset fingerprint archived
#   - PyTorch peak GPU memory logged by the custom trainer at milestones
#
# Prerequisites:
#   - Dataset001_LiverUS in nnUNet_raw
#   - nnUNetTrainer1000Milestones.py in custom_trainers/
#
# Usage:
#   script -c "bash training/run_preliminary_1000_epochs.sh" logs/training/preliminary_experiment.log

START_TIME=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

DATASET_ID=1
DATASET_NAME="Dataset001_LiverUS"
SEEDS=(42 43 44)

# Ordered list of checkpoints for deterministic iteration
CHECKPOINT_ORDER=(
    checkpoint_ep50.pth
    checkpoint_ep100.pth
    checkpoint_ep150.pth
    checkpoint_ep300.pth
    checkpoint_ep500.pth
    checkpoint_ep750.pth
    checkpoint_best_joint.pth
    checkpoint_best_mass.pth
    checkpoint_final.pth
)
declare -A CHECKPOINT_LABELS
CHECKPOINT_LABELS[checkpoint_ep50.pth]=ep50
CHECKPOINT_LABELS[checkpoint_ep100.pth]=ep100
CHECKPOINT_LABELS[checkpoint_ep150.pth]=ep150
CHECKPOINT_LABELS[checkpoint_ep300.pth]=ep300
CHECKPOINT_LABELS[checkpoint_ep500.pth]=ep500
CHECKPOINT_LABELS[checkpoint_ep750.pth]=ep750
CHECKPOINT_LABELS[checkpoint_best_joint.pth]=joint
CHECKPOINT_LABELS[checkpoint_best_mass.pth]=mass
CHECKPOINT_LABELS[checkpoint_final.pth]=final

# --- Log directories ---
LOGS_DIR="${REPO_DIR}/logs/training"
mkdir -p "$LOGS_DIR"

TIMES_CSV="${LOGS_DIR}/training_times.csv"
PRED_TIMES_CSV="${LOGS_DIR}/prediction_times.csv"

# Training times CSV. Note: wall_clock_seconds covers this invocation only.
# If training was resumed via --c, earlier epochs are not included in the
# shell-level timing. Per-epoch wall-clock times are recorded by the custom
# trainer in nnU-Net's training_log_*.txt.
echo "size,seed,epochs,wall_clock_seconds" > "$TIMES_CSV"
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
# Environment
# =====================================================================
echo "=== Preliminary experiment (625 images, seeds 42/43/44, 1000 epochs) ==="
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
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
python -c "import nnunetv2; print(f'nnU-Net {nnunetv2.__version__}')" \
    || echo "WARNING: could not determine nnU-Net version"

# nnU-Net version and source revision.
# A pip install (the standard path here) has no .git directory, so the
# SHA is reported as n/a rather than a misleading "not a git repo".
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
echo "Training seeds: 42, 43, 44"
echo "Trainer: nnUNetTrainer1000Milestones"
echo "Configuration: 2d, fold 0, PlainConvUNet"
echo "Patch size, batch size, and architecture details are recorded"
echo "  in nnU-Net's training_log and in the plans JSON below."
echo ""

# =====================================================================
# Preprocessing
# =====================================================================
echo "--- Preprocessing ---"
PREPROCESS_START=$(date +%s)
/usr/bin/time -v -o "${LOGS_DIR}/time_preprocess.txt" \
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
PREPROCESS_END=$(date +%s)
echo "Preprocessing wall clock: $(( PREPROCESS_END - PREPROCESS_START ))s"
echo ""

# Archive nnU-Net's auto-configuration decisions
echo "--- nnU-Net plans ---"
PLANS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"
if [ -f "$PLANS_FILE" ]; then
    cat "$PLANS_FILE"
    cp "$PLANS_FILE" "${LOGS_DIR}/nnUNetPlans.json"
else
    echo "WARNING: plans file not found at ${PLANS_FILE}"
fi

FINGERPRINT_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/dataset_fingerprint.json"
if [ -f "$FINGERPRINT_FILE" ]; then
    echo ""
    echo "--- Dataset fingerprint ---"
    cat "$FINGERPRINT_FILE"
    cp "$FINGERPRINT_FILE" "${LOGS_DIR}/dataset_fingerprint.json"
fi

# Record prediction defaults for the installed nnU-Net version
nnUNetv2_predict --help > "${LOGS_DIR}/predict_defaults.txt" 2>&1 || true

echo ""

# =====================================================================
# Training and prediction loop
# =====================================================================
for SEED in "${SEEDS[@]}"; do
    TRAINER="nnUNetTrainer1000Milestones_s${SEED}"
    CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__2d/fold_0"

    echo ""
    echo "========================================"
    echo "=== Seed ${SEED} ==="
    echo "========================================"

    # --- Start GPU monitoring (training + prediction) ---
    GPU_MONITOR_LOG="${LOGS_DIR}/gpu_monitor_s${SEED}.csv"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
        --format=csv \
        --loop-ms=1000 \
        > "$GPU_MONITOR_LOG" 2>&1 &
    GPU_MONITOR_PID=$!
    echo "GPU monitor started (PID ${GPU_MONITOR_PID}, log: ${GPU_MONITOR_LOG})"

    # --- Train ---
    echo ""
    echo "--- Training (1000 epochs, seed ${SEED}) ---"
    echo "Train start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    TRAIN_START=$(date +%s)
    /usr/bin/time -v -o "${LOGS_DIR}/time_train_s${SEED}.txt" \
        nnUNetv2_train $DATASET_ID 2d 0 --npz -tr $TRAINER --c
    TRAIN_END=$(date +%s)
    TRAIN_SECS=$(( TRAIN_END - TRAIN_START ))
    echo "Train end: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Training wall clock (seed ${SEED}): ${TRAIN_SECS}s ($(( TRAIN_SECS / 60 ))m)"
    echo "625,${SEED},1000,${TRAIN_SECS}" >> "$TIMES_CSV"
    echo ""

    # --- Checkpoint file sizes ---
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

    # --- Predict from all checkpoints ---
    echo "--- Predictions (seed ${SEED}) ---"
    for CHK in "${CHECKPOINT_ORDER[@]}"; do
        LBL="${CHECKPOINT_LABELS[$CHK]}"

        if [ ! -f "${CKPT_DIR}/${CHK}" ]; then
            echo "SKIP: ${CHK} not found (seed ${SEED})"
            continue
        fi

        PRED_DIR="${nnUNet_results}/predictions_625_s${SEED}_1000ep_${LBL}"
        echo ""
        echo "--- Predicting: ${LBL} (${CHK}, seed ${SEED}) ---"
        echo "Predict start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        PRED_START=$(date +%s)
        /usr/bin/time -v -o "${LOGS_DIR}/time_predict_s${SEED}_${LBL}.txt" \
            nnUNetv2_predict \
                -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
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
# GPU inference benchmark (per-image latency, run before the rented
# GPU session ends; CPU benchmark is repeated later on a different
# machine using the same script with --device cpu)
# =====================================================================
echo ""
echo "========================================"
echo "=== GPU inference benchmark ==="
echo "========================================"
BENCH_DIR="${LOGS_DIR}/../inference"
mkdir -p "$BENCH_DIR"

/usr/bin/time -v -o "${LOGS_DIR}/time_gpu_inference_benchmark.txt" \
    python "${SCRIPT_DIR}/benchmark_gpu_inference.py" \
        --nnunet-raw "${nnUNet_raw}" \
        --dataset-name "${DATASET_NAME}" \
        --dataset-id "${DATASET_ID}" \
        --seeds "${SEEDS[@]}" \
        --trainer-prefix nnUNetTrainer1000Milestones_s \
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
echo "=== Preliminary experiment complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Total wall clock: $(( END_TIME - START_TIME ))s ($(( (END_TIME - START_TIME) / 60 ))m)"
echo ""
echo "=== Log manifest ==="
echo "  ${TIMES_CSV}                        per-seed training wall clock"
echo "  ${PRED_TIMES_CSV}                   per-checkpoint prediction timing"
echo ""
echo "  Per-seed files:"
for S in "${SEEDS[@]}"; do
    echo "    gpu_monitor_s${S}.csv              GPU utilization/memory/power/temp samples (~1/s)"
    echo "    time_train_s${S}.txt               /usr/bin/time: peak RSS, CPU time for training"
    for CHK in "${CHECKPOINT_ORDER[@]}"; do
        LBL="${CHECKPOINT_LABELS[$CHK]}"
        echo "    time_predict_s${S}_${LBL}.txt"
    done
done
echo ""
echo "  Configuration:"
echo "    nnUNetPlans.json                   nnU-Net auto-configuration (patch size, batch size, etc.)"
echo "    dataset_fingerprint.json           dataset statistics"
echo "    predict_defaults.txt               nnUNetv2_predict --help output"
echo "    time_preprocess.txt                /usr/bin/time: peak RSS, CPU time for preprocessing"
echo ""
echo "  PyTorch peak GPU memory (allocated/reserved) is logged by the custom"
echo "  trainer at initialization, milestones (50/100/150/300/500/750), and training"
echo "  completion in nnU-Net's training_log_*.txt files under:"
echo "    ${nnUNet_results}/${DATASET_NAME}/nnUNetTrainer1000Milestones_s{SEED}__nnUNetPlans__2d/fold_0/"
echo ""
echo "  GPU inference benchmark (${BENCH_DIR}):"
echo "    inference_per_image_cuda.csv       per-image inference seconds, one row per case x seed"
echo "    inference_summary_cuda.csv         per-seed model load time, median/mean/stdev/IQR/range"
echo "    inference_settings_cuda.json       exact settings to replicate on CPU later"
echo "    ${LOGS_DIR}/time_gpu_inference_benchmark.txt   /usr/bin/time: peak RSS, CPU time for the benchmark script"
echo ""
echo "Predictions: predictions_625_s{42,43,44}_1000ep_{ep50,ep100,ep150,ep300,ep500,ep750,joint,mass,final}/"
