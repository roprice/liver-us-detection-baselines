#!/usr/bin/env bash
set -euo pipefail

# ResEnc M architecture ablation against the size-matched PlainConvUNet run
# from the data-scaling experiment. The dataset size defaults to 625 until
# the data-scaling analysis selects the final comparison size.
#
# Usage:
#   ABLATION_DATASET_SIZE=625 \
#     bash experiments/run_resenc_ablation.sh 2>&1 | \
#     tee experiment_logs/resenc_ablation/resenc_ablation.log

START_TIME=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

DATASET_SIZE="${ABLATION_DATASET_SIZE:-625}"
FOLD=0
EPOCHS=150
SEEDS=(42 43 44)
PLANNER="nnUNetPlannerResEncM"
PLANS_NAME="nnUNetResEncUNetMPlans"
TRAINER_PREFIX="nnUNetTrainerResEncAblation_seed"
PREDICTION_CHECKPOINT="checkpoint_final.pth"
PREDICTION_LABEL="final"
NNUNET_RELEASE_TAG_SHA="468cf803df9b267150ae2b6c0c59b8ac84f16227"
PIP_INSTALL_REPORT="${PIP_INSTALL_REPORT:-$HOME/pip-install-report.json}"

declare -A DATASET_IDS=(
    [5]=2 [10]=3 [20]=4 [40]=5 [80]=6 [160]=7 [320]=8 [625]=1
)
declare -A DATASET_NAMES=(
    [5]="Dataset002_AUL_005"
    [10]="Dataset003_AUL_010"
    [20]="Dataset004_AUL_020"
    [40]="Dataset005_AUL_040"
    [80]="Dataset006_AUL_080"
    [160]="Dataset007_AUL_160"
    [320]="Dataset008_AUL_320"
    [625]="Dataset001_AUL"
)

if [[ -z "${DATASET_IDS[$DATASET_SIZE]+x}" ]]; then
    echo "ERROR: ABLATION_DATASET_SIZE must be one of: 5 10 20 40 80 160 320 625" >&2
    exit 1
fi

DATASET_ID=${DATASET_IDS[$DATASET_SIZE]}
DATASET_NAME=${DATASET_NAMES[$DATASET_SIZE]}
DATASET_DIR="${nnUNet_raw}/${DATASET_NAME}"
EXPERIMENT_DIR="${REPO_DIR}/experiment_logs/resenc_ablation"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PREPROCESS_LOGS_DIR="${LOGS_DIR}/preprocessing/size${DATASET_SIZE}"
INFERENCE_LOGS_DIR="${LOGS_DIR}/inference/size${DATASET_SIZE}"
DATA_SCALING_MANIFEST="${REPO_DIR}/experiment_logs/data_scaling/subsets_manifest.json"
mkdir -p "$PREPROCESS_LOGS_DIR" "$INFERENCE_LOGS_DIR"

TRAIN_TIMES_CSV="${LOGS_DIR}/training_times.csv"
PREDICTION_TIMES_CSV="${LOGS_DIR}/prediction_times.csv"
if [ ! -f "$TRAIN_TIMES_CSV" ]; then
    echo "size,dataset_id,seed,epochs,architecture,plans,wall_clock_seconds" > "$TRAIN_TIMES_CSV"
fi
if [ ! -f "$PREDICTION_TIMES_CSV" ]; then
    echo "size,dataset_id,seed,checkpoint,architecture,plans,wall_clock_seconds,case_count" \
        > "$PREDICTION_TIMES_CSV"
fi

GPU_MONITOR_PID=""
cleanup_gpu_monitor() {
    if [ -n "$GPU_MONITOR_PID" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
        wait "$GPU_MONITOR_PID" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}
trap cleanup_gpu_monitor EXIT

count_png_files() {
    find "$1" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}

echo "=== ResEnc architecture ablation ==="
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Dataset size: ${DATASET_SIZE} (default 625; final size TBD)"
echo "Dataset: ${DATASET_NAME} (ID ${DATASET_ID})"
echo "Architecture preset: ResEnc M"
echo ""

if [ ! -d "${DATASET_DIR}/imagesTr" ] || [ ! -d "${DATASET_DIR}/labelsTr" ]; then
    echo "ERROR: ${DATASET_NAME} is missing from nnUNet_raw." >&2
    if [ "$DATASET_SIZE" != "625" ]; then
        echo "Run the data-scaling workflow to create this exact reduced dataset first." >&2
    fi
    exit 1
fi
TRAIN_CASE_COUNT=$(count_png_files "${DATASET_DIR}/imagesTr")
TEST_CASE_COUNT=$(count_png_files "${DATASET_DIR}/imagesTs")
if [ "$TRAIN_CASE_COUNT" -ne "$DATASET_SIZE" ]; then
    echo "ERROR: expected ${DATASET_SIZE} training images in ${DATASET_NAME}, found ${TRAIN_CASE_COUNT}" >&2
    exit 1
fi
if [ "$TEST_CASE_COUNT" -ne 110 ]; then
    echo "ERROR: expected 110 held-out test images in ${DATASET_NAME}, found ${TEST_CASE_COUNT}" >&2
    exit 1
fi
if [ ! -f "${DATASET_DIR}/case_mapping.json" ]; then
    echo "ERROR: ${DATASET_DIR}/case_mapping.json is required by the shared split policy" >&2
    exit 1
fi
if [ ! -f "$DATA_SCALING_MANIFEST" ]; then
    echo "ERROR: data-scaling manifest not found at ${DATA_SCALING_MANIFEST}." >&2
    echo "Restore the completed data-scaling evidence before running this comparison." >&2
    exit 1
fi

python - "$DATA_SCALING_MANIFEST" "$DATASET_SIZE" "$DATASET_DIR" "${nnUNet_raw}/Dataset001_AUL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
size = int(sys.argv[2])
dataset = Path(sys.argv[3])
source = Path(sys.argv[4])
manifest = json.loads(manifest_path.read_text())
subset = next((item for item in manifest["subsets"] if item["size"] == size), None)
if subset is None:
    raise SystemExit(f"data-scaling manifest has no {size}-case subset")
if subset["dataset_name"] != dataset.name:
    raise SystemExit(
        f"manifest maps size {size} to {subset['dataset_name']}, not {dataset.name}")

image_suffix = "_0000.png"
training_images = {
    path.name.removesuffix(image_suffix)
    for path in (dataset / "imagesTr").glob(f"*{image_suffix}")
}
training_labels = {path.stem for path in (dataset / "labelsTr").glob("*.png")}
expected_training = set(subset["case_ids"])
if training_images != expected_training or training_labels != expected_training:
    raise SystemExit("selected dataset does not match the exact data-scaling case IDs")

source_test_images = {path.name for path in (source / "imagesTs").glob("*.png")}
source_test_labels = {path.name for path in (source / "labelsTs").glob("*.png")}
if {path.name for path in (dataset / "imagesTs").glob("*.png")} != source_test_images:
    raise SystemExit("selected dataset does not use the source data-scaling test images")
if {path.name for path in (dataset / "labelsTs").glob("*.png")} != source_test_labels:
    raise SystemExit("selected dataset does not use the source data-scaling test labels")
if len(source_test_images) != manifest["source_test_image_count"]:
    raise SystemExit("source test count differs from the data-scaling manifest")

digest = hashlib.sha256((source / "dataset.json").read_bytes()).hexdigest()
if digest != manifest["source_dataset_json_sha256"]:
    raise SystemExit("source dataset.json differs from the data-scaling manifest")
print(f"Verified exact {size}-case data-scaling dataset from {manifest_path}")
PY

if [ ! -f "$PIP_INSTALL_REPORT" ]; then
    echo "ERROR: pip installation report not found at ${PIP_INSTALL_REPORT}." >&2
    echo "Install with: python -m pip install --report \"${PIP_INSTALL_REPORT}\" -r requirements.txt --break-system-packages" >&2
    exit 1
fi

cp "$PIP_INSTALL_REPORT" "${LOGS_DIR}/pip-install-report.json"
python - "$PIP_INSTALL_REPORT" "${LOGS_DIR}/environment.json" "$REPO_DIR" "$NNUNET_RELEASE_TAG_SHA" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlparse

report_path = Path(sys.argv[1]).resolve()
output_path = Path(sys.argv[2]).resolve()
repo = Path(sys.argv[3]).resolve()
release_tag_sha = sys.argv[4]
report = json.loads(report_path.read_text())
record = next(
    (
        item for item in report.get("install", [])
        if item.get("metadata", {}).get("name", "").lower().replace("_", "-")
        == "nnunetv2"
    ),
    None,
)
if record is None:
    raise SystemExit(
        "pip report does not contain nnunetv2; use a fresh environment and reinstall "
        "requirements with --report")

version = record.get("metadata", {}).get("version")
installed_version = metadata.version("nnunetv2")
if version != "2.8.1" or installed_version != "2.8.1":
    raise SystemExit(
        f"Expected nnunetv2 2.8.1, report has {version!r} and environment has "
        f"{installed_version!r}")

download = record.get("download_info", {})
url = download.get("url", "")
archive = download.get("archive_info", {})
sha256 = archive.get("hashes", {}).get("sha256")
if not sha256:
    legacy_hash = archive.get("hash", "")
    if legacy_hash.startswith("sha256="):
        sha256 = legacy_hash.removeprefix("sha256=")
if not url or not sha256:
    raise SystemExit("pip report lacks the nnunetv2 artifact URL or SHA256")

hostname = urlparse(url).hostname or ""
source = "PyPI" if hostname.endswith(("pypi.org", "pythonhosted.org")) else hostname or url
repo_sha = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())

provenance = {
    "python": sys.version,
    "platform": platform.platform(),
    "nnunet": {
        "install_source": source,
        "version": installed_version,
        "artifact_url": url,
        "artifact_sha256": sha256,
        "release_tag": "v2.8.1",
        "release_tag_sha": release_tag_sha,
    },
    "study_repository": {"sha": repo_sha, "dirty": dirty},
}
output_path.write_text(json.dumps(provenance, indent=2) + "\n")
print(f"nnU-Net install source: {source}")
print(f"nnU-Net version: {installed_version}")
print(f"nnU-Net artifact SHA256: {sha256}")
print(f"nnU-Net artifact URL: {url}")
print(f"nnU-Net v2.8.1 tag SHA: {release_tag_sha}")
print(f"Repo SHA: {repo_sha}")
print(f"Repo dirty: {str(dirty).lower()}")
PY

echo "Host: $(hostname)"
echo "OS: $(uname -s) $(uname -r)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "WARNING: nvidia-smi query failed"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}; GPUs: {torch.cuda.device_count()}')"
echo "Training seeds: ${SEEDS[*]}"
echo "Trainer: ${TRAINER_PREFIX}{42,43,44}"
echo "Configuration: 2d, fold ${FOLD}, ${PLANS_NAME}"
echo "Epoch budget: ${EPOCHS}"
echo "Prediction checkpoint: ${PREDICTION_CHECKPOINT}"
echo ""

PREPROCESS_MARKER="${PREPROCESS_LOGS_DIR}/complete"
PLANS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS_NAME}.json"
FINGERPRINT_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/dataset_fingerprint.json"
PREPROCESSED_DATA_DIR="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d"
PREPROCESSED_CASE_COUNT=0
if [ -d "$PREPROCESSED_DATA_DIR" ]; then
    PREPROCESSED_CASE_COUNT=$(find "$PREPROCESSED_DATA_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')
fi
if [ ! -f "$PREPROCESS_MARKER" ] || [ ! -f "$PLANS_FILE" ] || \
        [ "$PREPROCESSED_CASE_COUNT" -ne "$DATASET_SIZE" ]; then
    echo "--- Planning and preprocessing ${DATASET_NAME} with ${PLANNER} ---"
    PREPROCESS_START=$(date +%s)
    /usr/bin/time -v -o "${PREPROCESS_LOGS_DIR}/time_preprocess.txt" \
        nnUNetv2_plan_and_preprocess -d "$DATASET_ID" -pl "$PLANNER" \
            --verify_dataset_integrity
    PREPROCESS_END=$(date +%s)
    echo "Planning/preprocessing wall clock: $(( PREPROCESS_END - PREPROCESS_START ))s"
    touch "$PREPROCESS_MARKER"
else
    echo "SKIP: ResEnc plans and preprocessing marker already exist"
fi

if [ ! -f "$PLANS_FILE" ]; then
    echo "ERROR: expected ResEnc plans not found at ${PLANS_FILE}" >&2
    exit 1
fi
PREPROCESSED_CASE_COUNT=$(find "$PREPROCESSED_DATA_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')
if [ "$PREPROCESSED_CASE_COUNT" -ne "$DATASET_SIZE" ]; then
    echo "ERROR: expected ${DATASET_SIZE} preprocessed cases, found ${PREPROCESSED_CASE_COUNT}" >&2
    exit 1
fi
cp "$PLANS_FILE" "${PREPROCESS_LOGS_DIR}/${PLANS_NAME}.json"
if [ -f "$FINGERPRINT_FILE" ]; then
    cp "$FINGERPRINT_FILE" "${PREPROCESS_LOGS_DIR}/dataset_fingerprint.json"
fi
nnUNetv2_predict --help > "${LOGS_DIR}/predict_defaults.txt" 2>&1 || true

for SEED in "${SEEDS[@]}"; do
    TRAINER="${TRAINER_PREFIX}${SEED}"
    CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS_NAME}__2d/fold_${FOLD}"
    FINAL_CHECKPOINT="${CKPT_DIR}/${PREDICTION_CHECKPOINT}"

    echo ""
    echo "--- ResEnc M, size ${DATASET_SIZE}, seed ${SEED} ---"
    GPU_MONITOR_LOG="${LOGS_DIR}/gpu_monitor_size${DATASET_SIZE}_s${SEED}.csv"
    nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
        --format=csv --loop-ms=1000 > "$GPU_MONITOR_LOG" 2>&1 &
    GPU_MONITOR_PID=$!

    if [ -f "$FINAL_CHECKPOINT" ]; then
        python - "$FINAL_CHECKPOINT" "${SCRIPT_DIR}/custom_trainers" "$TRAINER" "$PLANS_NAME" "$EPOCHS" <<'PY'
import sys
import torch

sys.path.insert(0, sys.argv[2])
from nnUNetTrainerDataScaling import nnUNetTrainerDataScaling

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected_trainer = sys.argv[3]
expected_plans = sys.argv[4]
expected_epochs = int(sys.argv[5])
if checkpoint.get("data_scaling_split_policy") != nnUNetTrainerDataScaling.SPLIT_POLICY:
    raise RuntimeError("Final checkpoint does not use the required data-scaling split policy")
if checkpoint.get("trainer_name") != expected_trainer:
    raise RuntimeError(
        f"Final checkpoint trainer is {checkpoint.get('trainer_name')}, expected {expected_trainer}")
if checkpoint.get("current_epoch") != expected_epochs:
    raise RuntimeError(
        f"Final checkpoint epoch is {checkpoint.get('current_epoch')}, expected {expected_epochs}")
checkpoint_plans = checkpoint.get("init_args", {}).get("plans", {}).get("plans_name")
if checkpoint_plans != expected_plans:
    raise RuntimeError(
        f"Final checkpoint plans are {checkpoint_plans}, expected {expected_plans}")
PY
        echo "SKIP: final checkpoint already exists at ${FINAL_CHECKPOINT}"
    else
        TRAIN_START=$(date +%s)
        /usr/bin/time -v -o "${LOGS_DIR}/time_train_size${DATASET_SIZE}_s${SEED}.txt" \
            nnUNetv2_train "$DATASET_ID" 2d "$FOLD" --npz \
                -p "$PLANS_NAME" -tr "$TRAINER" --c
        TRAIN_END=$(date +%s)
        TRAIN_SECONDS=$(( TRAIN_END - TRAIN_START ))
        echo "${DATASET_SIZE},${DATASET_ID},${SEED},${EPOCHS},ResEncM,${PLANS_NAME},${TRAIN_SECONDS}" \
            >> "$TRAIN_TIMES_CSV"
    fi

    if [ ! -f "$FINAL_CHECKPOINT" ]; then
        echo "ERROR: expected final checkpoint not found: ${FINAL_CHECKPOINT}" >&2
        exit 1
    fi

    python - "$CKPT_DIR" <<'PY'
import os
import sys
import torch

checkpoint_path = os.path.join(sys.argv[1], "checkpoint_final.pth")
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
weights = checkpoint.get("network_weights", {})
print(f"Parameters: {sum(weight.numel() for weight in weights.values()):,}")
print(f"Checkpoint file size: {os.path.getsize(checkpoint_path):,} bytes")
weights_path = "/tmp/nnunet_resenc_ablation_weights_only.pth"
torch.save(weights, weights_path)
print(f"Weights-only file size: {os.path.getsize(weights_path):,} bytes")
os.remove(weights_path)
PY

    PREDICTION_DIR="${nnUNet_results}/predictions_resenc_ablation_${DATASET_SIZE}images_seed${SEED}_${PREDICTION_LABEL}"
    EXISTING_PREDICTION_COUNT=0
    if [ -d "$PREDICTION_DIR" ]; then
        EXISTING_PREDICTION_COUNT=$(count_png_files "$PREDICTION_DIR")
    fi
    if [ "$EXISTING_PREDICTION_COUNT" -eq "$TEST_CASE_COUNT" ]; then
        echo "SKIP: ${PREDICTION_DIR} already contains ${TEST_CASE_COUNT} masks"
    else
        if [ -d "$PREDICTION_DIR" ]; then
            rm -rf "$PREDICTION_DIR"
        fi
        PREDICTION_START=$(date +%s)
        /usr/bin/time -v -o "${LOGS_DIR}/time_predict_size${DATASET_SIZE}_s${SEED}_${PREDICTION_LABEL}.txt" \
            nnUNetv2_predict \
                -i "${DATASET_DIR}/imagesTs" \
                -o "$PREDICTION_DIR" \
                -d "$DATASET_ID" -c 2d -f "$FOLD" \
                -p "$PLANS_NAME" -tr "$TRAINER" \
                -chk "$PREDICTION_CHECKPOINT"
        PREDICTION_END=$(date +%s)
        PREDICTION_SECONDS=$(( PREDICTION_END - PREDICTION_START ))
        PREDICTION_CASE_COUNT=$(count_png_files "$PREDICTION_DIR")
        if [ "$PREDICTION_CASE_COUNT" -ne "$TEST_CASE_COUNT" ]; then
            echo "ERROR: prediction count is ${PREDICTION_CASE_COUNT}, expected ${TEST_CASE_COUNT}" >&2
            exit 1
        fi
        echo "${DATASET_SIZE},${DATASET_ID},${SEED},${PREDICTION_LABEL},ResEncM,${PLANS_NAME},${PREDICTION_SECONDS},${PREDICTION_CASE_COUNT}" \
            >> "$PREDICTION_TIMES_CSV"
    fi

    cleanup_gpu_monitor
    echo "GPU monitor stopped (size ${DATASET_SIZE}, seed ${SEED})"
done

INFERENCE_SUMMARY="${INFERENCE_LOGS_DIR}/inference_summary_cuda.csv"
if [ -f "$INFERENCE_SUMMARY" ]; then
    echo "SKIP: GPU inference benchmark already completed"
else
    /usr/bin/time -v -o "${INFERENCE_LOGS_DIR}/time_gpu_inference_benchmark.txt" \
        python "${SCRIPT_DIR}/benchmark_gpu_inference.py" \
            --nnunet-raw "$nnUNet_raw" \
            --dataset-name "$DATASET_NAME" \
            --dataset-id "$DATASET_ID" \
            --seeds "${SEEDS[@]}" \
            --trainer-prefix "$TRAINER_PREFIX" \
            --plans "$PLANS_NAME" \
            --checkpoint "$PREDICTION_CHECKPOINT" \
            --device cuda \
            --output-dir "$INFERENCE_LOGS_DIR"
fi

END_TIME=$(date +%s)
echo ""
echo "=== ResEnc architecture ablation complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Total wall clock: $(( END_TIME - START_TIME ))s ($(( (END_TIME - START_TIME) / 60 ))m)"
echo "Outputs: ${EXPERIMENT_DIR}"
echo "Predictions: ${nnUNet_results}/predictions_resenc_ablation_${DATASET_SIZE}images_seed{42,43,44}_final/"
