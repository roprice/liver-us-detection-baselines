#!/usr/bin/env bash
set -euo pipefail

# Data-scaling experiment: train nested, pathology-stratified subsets
# of Dataset001_AUL at 5, 10, 20, 40, 80, 160, 320, and 625 cases. Every
# dataset uses the same 110-image held-out test set, a fixed 150-epoch
# budget, fold 0, and initialization seeds 42, 43, and 44. Predictions use
# checkpoint_final.pth so epoch count, rather than a size-specific internal
# validation checkpoint, determines model selection.
#
# The script creates Dataset002_AUL_005 through Dataset008_AUL_320 from the
# source Dataset001_AUL. Dataset001_AUL itself is the 625-case pool. A
# reproducibility manifest of the exact nested selections is written to
# experiment_logs/data_scaling/subsets_manifest.json.
#
# Prerequisites:
#   - Dataset001_AUL in nnUNet_raw
#   - nnUNetTrainerDataScaling.py in training/custom_trainers/
#
# Usage:
#   bash training/run_data_scaling.sh 2>&1 | \
#     tee experiment_logs/data_scaling/data_scaling.log

START_TIME=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
export nnUNet_extTrainer="${SCRIPT_DIR}/custom_trainers"

: "${nnUNet_raw:?Set nnUNet_raw before running}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed before running}"
: "${nnUNet_results:?Set nnUNet_results before running}"

EXPERIMENT_NAME="data_scaling"
SOURCE_DATASET_ID=1
SOURCE_DATASET_NAME="Dataset001_AUL"
FOLD=0
EPOCHS=150
SEEDS=(42 43 44)
SUBSET_SIZES=(5 10 20 40 80 160 320 625)
PREDICTION_CHECKPOINT="checkpoint_final.pth"
PREDICTION_LABEL="final"

# Dataset001_AUL is the full source pool. The generated nested subsets use
# IDs 2 through 8, which must remain unused by other nnU-Net datasets.
declare -A DATASET_IDS=(
    [5]=2
    [10]=3
    [20]=4
    [40]=5
    [80]=6
    [160]=7
    [320]=8
    [625]=1
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

EXPERIMENT_DIR="${REPO_DIR}/experiment_logs/${EXPERIMENT_NAME}"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PREPROCESS_LOGS_DIR="${LOGS_DIR}/preprocessing"
INFERENCE_LOGS_DIR="${LOGS_DIR}/inference"
mkdir -p "$PREPROCESS_LOGS_DIR" "$INFERENCE_LOGS_DIR"

TRAIN_TIMES_CSV="${LOGS_DIR}/training_times.csv"
PREDICTION_TIMES_CSV="${LOGS_DIR}/prediction_times.csv"
if [ ! -f "$TRAIN_TIMES_CSV" ]; then
    echo "size,dataset_id,seed,epochs,wall_clock_seconds" > "$TRAIN_TIMES_CSV"
fi
if [ ! -f "$PREDICTION_TIMES_CSV" ]; then
    echo "size,dataset_id,seed,checkpoint,wall_clock_seconds,case_count" \
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

# =====================================================================
# Create and verify the nested datasets
# =====================================================================
echo "=== Data-scaling experiment ==="
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "--- Creating/verifying nested data subsets ---"

python - "$nnUNet_raw" "$EXPERIMENT_DIR/subsets_manifest.json" <<'PY'
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_DATASET_NAME = "Dataset001_AUL"
SUBSET_SEED = 42
SUBSETS = (
    (5, 2, "Dataset002_AUL_005"),
    (10, 3, "Dataset003_AUL_010"),
    (20, 4, "Dataset004_AUL_020"),
    (40, 5, "Dataset005_AUL_040"),
    (80, 6, "Dataset006_AUL_080"),
    (160, 7, "Dataset007_AUL_160"),
    (320, 8, "Dataset008_AUL_320"),
    (625, 1, "Dataset001_AUL"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_case_ids(directory):
    suffix = "_0000.png"
    return sorted(
        path.name.removesuffix(suffix)
        for path in directory.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    )


def label_case_ids(directory):
    return sorted(path.stem for path in directory.glob("*.png"))


def has_mass(label_path):
    return bool(np.any(np.asarray(Image.open(label_path)) == 2))


def expected_selection(malignant_cases, benign_cases, no_mass_cases, size):
    mass_case_count = len(malignant_cases) + len(benign_cases)
    total_case_count = mass_case_count + len(no_mass_cases)
    selected_mass_count = int(round(size * mass_case_count / total_case_count))
    selected_no_mass_count = size - selected_mass_count
    selected_malignant_count = int(round(
        selected_mass_count * len(malignant_cases) / mass_case_count))
    selected_benign_count = selected_mass_count - selected_malignant_count

    return (
        selected_malignant_count,
        selected_benign_count,
        selected_no_mass_count,
    )


def verify_existing_dataset(destination, selected_cases, source_test_images,
                            source_test_labels, expected_dataset_json):
    expected_images = {f"{case}_0000.png" for case in selected_cases}
    expected_labels = {f"{case}.png" for case in selected_cases}
    actual_images = {path.name for path in (destination / "imagesTr").glob("*.png")}
    actual_labels = {path.name for path in (destination / "labelsTr").glob("*.png")}
    actual_test_images = {
        path.name for path in (destination / "imagesTs").glob("*.png")}
    actual_test_labels = {
        path.name for path in (destination / "labelsTs").glob("*.png")}

    if actual_images != expected_images or actual_labels != expected_labels:
        raise RuntimeError(
            f"Existing {destination.name} does not match the expected nested "
            "training-case selection. Refusing to overwrite it.")
    if actual_test_images != source_test_images or actual_test_labels != source_test_labels:
        raise RuntimeError(
            f"Existing {destination.name} does not preserve the source test set. "
            "Refusing to overwrite it.")
    with (destination / "dataset.json").open() as file:
        actual_dataset_json = json.load(file)
    if actual_dataset_json != expected_dataset_json:
        raise RuntimeError(
            f"Existing {destination.name}/dataset.json differs from the expected "
            "source metadata and training-case count. Refusing to overwrite it.")


def copy_dataset(source, destination, selected_cases, dataset_json, mapping):
    temporary_directory = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for directory_name in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (temporary_directory / directory_name).mkdir()

        for case in selected_cases:
            shutil.copy2(
                source / "imagesTr" / f"{case}_0000.png",
                temporary_directory / "imagesTr" / f"{case}_0000.png")
            shutil.copy2(
                source / "labelsTr" / f"{case}.png",
                temporary_directory / "labelsTr" / f"{case}.png")

        for directory_name in ("imagesTs", "labelsTs"):
            for source_file in sorted((source / directory_name).glob("*.png")):
                shutil.copy2(source_file, temporary_directory / directory_name / source_file.name)

        with (temporary_directory / "dataset.json").open("w") as file:
            json.dump(dataset_json, file, indent=2)
            file.write("\n")

        if mapping is not None:
            selected_case_names = set(selected_cases)
            test_case_names = set(image_case_ids(source / "imagesTs"))
            filtered_mapping = [
                record for record in mapping
                if record.get("case_name") in selected_case_names | test_case_names
            ]
            with (temporary_directory / "case_mapping.json").open("w") as file:
                json.dump(filtered_mapping, file, indent=2)
                file.write("\n")

        os.replace(temporary_directory, destination)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


raw_base = Path(sys.argv[1]).expanduser().resolve()
manifest_path = Path(sys.argv[2]).expanduser().resolve()
source = raw_base / SOURCE_DATASET_NAME
for required_directory in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
    if not (source / required_directory).is_dir():
        raise RuntimeError(f"Missing required source directory: {source / required_directory}")

source_train_images = image_case_ids(source / "imagesTr")
source_train_labels = label_case_ids(source / "labelsTr")
if source_train_images != source_train_labels:
    raise RuntimeError("Source imagesTr and labelsTr do not contain the same case IDs")
if len(source_train_images) != 625:
    raise RuntimeError(
        f"Expected 625 source training cases, found {len(source_train_images)}")

source_test_images = {path.name for path in (source / "imagesTs").glob("*.png")}
source_test_labels = {path.name for path in (source / "labelsTs").glob("*.png")}
if not source_test_images or not source_test_labels:
    raise RuntimeError("Source test images or labels are missing")

with (source / "dataset.json").open() as file:
    source_dataset_json = json.load(file)
source_mapping_path = source / "case_mapping.json"
if not source_mapping_path.exists():
    raise RuntimeError(
        f"Pathology stratification requires {source_mapping_path}")
with source_mapping_path.open() as file:
    source_mapping = json.load(file)

training_categories = {
    record["case_name"]: record["category"]
    for record in source_mapping
    if record.get("split") == "train"
}
if set(training_categories) != set(source_train_images):
    raise RuntimeError(
        "Training cases in case_mapping.json do not match imagesTr")

malignant_cases = []
benign_cases = []
no_mass_cases = []
for case in source_train_images:
    category = training_categories[case]
    mass_is_present = has_mass(source / "labelsTr" / f"{case}.png")
    if category == "Malignant":
        if not mass_is_present:
            raise RuntimeError(f"Expected a mass in malignant case {case}")
        malignant_cases.append(case)
    elif category == "Benign":
        if not mass_is_present:
            raise RuntimeError(f"Expected a mass in benign case {case}")
        benign_cases.append(case)
    elif category == "Normal":
        if mass_is_present:
            raise RuntimeError(f"Unexpected mass in normal case {case}")
        no_mass_cases.append(case)
    else:
        raise RuntimeError(
            f"Unsupported training pathology category {category!r} for {case}")

rng = np.random.RandomState(SUBSET_SEED)
malignant_shuffled = rng.permutation(malignant_cases).tolist()
benign_shuffled = rng.permutation(benign_cases).tolist()
no_mass_shuffled = rng.permutation(no_mass_cases).tolist()

selections = {}
manifest_subsets = []
for size, dataset_id, dataset_name in SUBSETS:
    malignant_count, benign_count, no_mass_count = expected_selection(
        malignant_cases, benign_cases, no_mass_cases, size)
    selected_cases = sorted(
        malignant_shuffled[:malignant_count]
        + benign_shuffled[:benign_count]
        + no_mass_shuffled[:no_mass_count])
    selections[size] = set(selected_cases)

    manifest_subsets.append({
        "size": size,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "malignant_cases": malignant_count,
        "benign_cases": benign_count,
        "mass_negative_cases": no_mass_count,
        "case_ids": selected_cases,
    })

for smaller, larger in zip(SUBSETS, SUBSETS[1:]):
    smaller_size = smaller[0]
    larger_size = larger[0]
    if not selections[smaller_size].issubset(selections[larger_size]):
        raise RuntimeError(
            f"Nesting violated: {smaller_size} is not a subset of {larger_size}")

for subset in manifest_subsets:
    size = subset["size"]
    dataset_name = subset["dataset_name"]
    selected_cases = subset["case_ids"]
    destination = raw_base / dataset_name

    if size == 625:
        if destination != source:
            raise RuntimeError("The 625-case dataset must be Dataset001_AUL")
        continue

    dataset_json = dict(source_dataset_json)
    dataset_json["numTraining"] = len(selected_cases)
    if destination.exists():
        verify_existing_dataset(
            destination, selected_cases, source_test_images, source_test_labels,
            dataset_json)
        print(f"Verified existing {dataset_name}: {size} cases")
    else:
        copy_dataset(
            source, destination, selected_cases, dataset_json, source_mapping)
        print(
            f"Created {dataset_name}: {size} cases "
            f"({subset['malignant_cases']} malignant, "
            f"{subset['benign_cases']} benign, "
            f"{subset['mass_negative_cases']} mass-negative)")

manifest = {
    "source_dataset": SOURCE_DATASET_NAME,
    "source_dataset_json_sha256": sha256(source / "dataset.json"),
    "source_training_case_count": len(source_train_images),
    "source_malignant_cases": len(malignant_cases),
    "source_benign_cases": len(benign_cases),
    "source_mass_negative_cases": len(no_mass_cases),
    "source_test_image_count": len(source_test_images),
    "subset_seed": SUBSET_SEED,
    "stratification": "malignant, benign, and mass-negative pathology groups", 
    "nested": True,
    "subsets": manifest_subsets,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w") as file:
    json.dump(manifest, file, indent=2)
    file.write("\n")

print(
    f"Source pool: {len(source_train_images)} cases "
    f"({len(malignant_cases)} malignant, {len(benign_cases)} benign, "
    f"{len(no_mass_cases)} mass-negative)")
print("Nesting verified. Manifest written to " + str(manifest_path))
PY

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
python -c "from importlib.metadata import version; print(f'nnU-Net {version(\"nnunetv2\")}')" \
    || echo "WARNING: could not determine nnU-Net version"

NNUNET_VERSION=$(python -c "from importlib.metadata import version; print(version('nnunetv2'))" 2>/dev/null || echo "unknown")
NNUNET_PATH=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])" 2>/dev/null || echo "")
if [ -n "$NNUNET_PATH" ]; then
    NNUNET_SHA=$(git -C "$NNUNET_PATH" rev-parse HEAD 2>/dev/null || echo "n/a (pip install)")
else
    NNUNET_SHA="n/a (could not locate install path)"
fi
REPO_SHA=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "not a git repo")
echo "nnU-Net version: ${NNUNET_VERSION}"
echo "nnU-Net source SHA: ${NNUNET_SHA}"
echo "Repo SHA: ${REPO_SHA}"
echo ""
echo "Subset sizes: ${SUBSET_SIZES[*]}"
echo "Subset selection seed: 42"
echo "Training seeds: ${SEEDS[*]}"
echo "Trainer: nnUNetTrainerDataScaling_seed{42,43,44}"
echo "Configuration: 2d, fold ${FOLD}, PlainConvUNet"
echo "Epoch budget: ${EPOCHS}"
echo "Prediction checkpoint: ${PREDICTION_CHECKPOINT}"
echo ""

# =====================================================================
# Preprocess each dataset and archive its nnU-Net configuration
# =====================================================================
for SIZE in "${SUBSET_SIZES[@]}"; do
    DATASET_ID=${DATASET_IDS[$SIZE]}
    DATASET_NAME=${DATASET_NAMES[$SIZE]}
    SIZE_PREPROCESS_LOGS="${PREPROCESS_LOGS_DIR}/size${SIZE}"
    PREPROCESS_MARKER="${SIZE_PREPROCESS_LOGS}/complete"
    mkdir -p "$SIZE_PREPROCESS_LOGS"

    echo "--- Preprocessing ${DATASET_NAME} (${SIZE} cases) ---"
    if [ ! -f "$PREPROCESS_MARKER" ]; then
        PREPROCESS_START=$(date +%s)
        /usr/bin/time -v -o "${SIZE_PREPROCESS_LOGS}/time_preprocess.txt" \
            nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity
        PREPROCESS_END=$(date +%s)
        echo "Preprocessing wall clock: $(( PREPROCESS_END - PREPROCESS_START ))s"
        touch "$PREPROCESS_MARKER"
    else
        echo "SKIP: preprocessing already completed according to ${PREPROCESS_MARKER}"
    fi

    PLANS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"
    FINGERPRINT_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/dataset_fingerprint.json"
    if [ -f "$PLANS_FILE" ]; then
        cp "$PLANS_FILE" "${SIZE_PREPROCESS_LOGS}/nnUNetPlans.json"
    else
        echo "WARNING: plans file not found at ${PLANS_FILE}"
    fi
    if [ -f "$FINGERPRINT_FILE" ]; then
        cp "$FINGERPRINT_FILE" "${SIZE_PREPROCESS_LOGS}/dataset_fingerprint.json"
    else
        echo "WARNING: dataset fingerprint not found at ${FINGERPRINT_FILE}"
    fi
    echo ""
done

nnUNetv2_predict --help > "${LOGS_DIR}/predict_defaults.txt" 2>&1 || true

# =====================================================================
# Train and predict every dataset-size/seed combination
# =====================================================================
for SIZE in "${SUBSET_SIZES[@]}"; do
    DATASET_ID=${DATASET_IDS[$SIZE]}
    DATASET_NAME=${DATASET_NAMES[$SIZE]}

    echo ""
    echo "========================================"
    echo "=== ${SIZE}-case dataset (${DATASET_NAME}) ==="
    echo "========================================"

    for SEED in "${SEEDS[@]}"; do
        TRAINER="nnUNetTrainerDataScaling_seed${SEED}"
        CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__2d/fold_${FOLD}"
        FINAL_CHECKPOINT="${CKPT_DIR}/${PREDICTION_CHECKPOINT}"

        echo ""
        echo "--- Size ${SIZE}, seed ${SEED} ---"

        GPU_MONITOR_LOG="${LOGS_DIR}/gpu_monitor_size${SIZE}_s${SEED}.csv"
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
            --format=csv \
            --loop-ms=1000 \
            > "$GPU_MONITOR_LOG" 2>&1 &
        GPU_MONITOR_PID=$!
        echo "GPU monitor started (PID ${GPU_MONITOR_PID}, log: ${GPU_MONITOR_LOG})"

        if [ -f "$FINAL_CHECKPOINT" ]; then
            echo "SKIP: final checkpoint already exists at ${FINAL_CHECKPOINT}"
        else
            echo "Train start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            TRAIN_START=$(date +%s)
            /usr/bin/time -v -o "${LOGS_DIR}/time_train_size${SIZE}_s${SEED}.txt" \
                nnUNetv2_train "$DATASET_ID" 2d "$FOLD" --npz -tr "$TRAINER" --c
            TRAIN_END=$(date +%s)
            TRAIN_SECONDS=$(( TRAIN_END - TRAIN_START ))
            echo "Train end: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            echo "Training wall clock: ${TRAIN_SECONDS}s ($(( TRAIN_SECONDS / 60 ))m)"
            echo "${SIZE},${DATASET_ID},${SEED},${EPOCHS},${TRAIN_SECONDS}" \
                >> "$TRAIN_TIMES_CSV"
        fi

        if [ ! -f "$FINAL_CHECKPOINT" ]; then
            echo "ERROR: expected final checkpoint not found: ${FINAL_CHECKPOINT}"
            exit 1
        fi

        echo "--- Checkpoint file sizes (size ${SIZE}, seed ${SEED}) ---"
        ls -lh "${CKPT_DIR}"/*.pth 2>/dev/null \
            || echo "WARNING: no checkpoint files found"

        echo "--- Model footprint (size ${SIZE}, seed ${SEED}) ---"
        python - "$CKPT_DIR" <<'PY'
import os
import sys

import torch

checkpoint_directory = sys.argv[1]
checkpoint_path = os.path.join(checkpoint_directory, "checkpoint_final.pth")
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
weights = checkpoint.get("network_weights", {})
parameter_count = sum(weight.numel() for weight in weights.values())
print(f"Parameters: {parameter_count:,}")
print(f"Checkpoint file size: {os.path.getsize(checkpoint_path):,} bytes")
weights_path = "/tmp/nnunet_data_scaling_weights_only.pth"
torch.save(weights, weights_path)
print(f"Weights-only file size: {os.path.getsize(weights_path):,} bytes")
os.remove(weights_path)
PY

        PREDICTION_DIR="${nnUNet_results}/predictions_data_scaling_${SIZE}images_seed${SEED}_${PREDICTION_LABEL}"
        TEST_CASE_COUNT=$(count_png_files "${nnUNet_raw}/${DATASET_NAME}/imagesTs")
        EXISTING_PREDICTION_COUNT=0
        if [ -d "$PREDICTION_DIR" ]; then
            EXISTING_PREDICTION_COUNT=$(count_png_files "$PREDICTION_DIR")
        fi

        if [ "$EXISTING_PREDICTION_COUNT" -eq "$TEST_CASE_COUNT" ]; then
            echo "SKIP: ${PREDICTION_DIR} already contains ${TEST_CASE_COUNT} masks"
        else
            if [ -d "$PREDICTION_DIR" ]; then
                echo "Removing incomplete prediction directory: ${PREDICTION_DIR}"
                rm -rf "$PREDICTION_DIR"
            fi
            echo "--- Predicting final checkpoint (size ${SIZE}, seed ${SEED}) ---"
            echo "Predict start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            PREDICTION_START=$(date +%s)
            /usr/bin/time -v -o "${LOGS_DIR}/time_predict_size${SIZE}_s${SEED}_${PREDICTION_LABEL}.txt" \
                nnUNetv2_predict \
                    -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
                    -o "$PREDICTION_DIR" \
                    -d "$DATASET_ID" -c 2d -f "$FOLD" \
                    -tr "$TRAINER" \
                    -chk "$PREDICTION_CHECKPOINT"
            PREDICTION_END=$(date +%s)
            PREDICTION_SECONDS=$(( PREDICTION_END - PREDICTION_START ))
            PREDICTION_CASE_COUNT=$(count_png_files "$PREDICTION_DIR")
            echo "Predict end: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
            echo "Prediction: ${PREDICTION_SECONDS}s, ${PREDICTION_CASE_COUNT} cases"
            echo "${SIZE},${DATASET_ID},${SEED},${PREDICTION_LABEL},${PREDICTION_SECONDS},${PREDICTION_CASE_COUNT}" \
                >> "$PREDICTION_TIMES_CSV"
        fi

        cleanup_gpu_monitor
        echo "GPU monitor stopped (size ${SIZE}, seed ${SEED})"
    done
done

# =====================================================================
# Per-image GPU inference benchmark for every final model
# =====================================================================
echo ""
echo "========================================"
echo "=== GPU inference benchmarks ==="
echo "========================================"
for SIZE in "${SUBSET_SIZES[@]}"; do
    DATASET_ID=${DATASET_IDS[$SIZE]}
    DATASET_NAME=${DATASET_NAMES[$SIZE]}
    SIZE_INFERENCE_LOGS="${INFERENCE_LOGS_DIR}/size${SIZE}"
    INFERENCE_SUMMARY="${SIZE_INFERENCE_LOGS}/inference_summary_cuda.csv"
    mkdir -p "$SIZE_INFERENCE_LOGS"

    if [ -f "$INFERENCE_SUMMARY" ]; then
        echo "SKIP: GPU inference benchmark already completed for size ${SIZE}"
        continue
    fi

    echo "--- GPU inference benchmark: ${SIZE} cases ---"
    /usr/bin/time -v -o "${SIZE_INFERENCE_LOGS}/time_gpu_inference_benchmark.txt" \
        python "${SCRIPT_DIR}/benchmark_gpu_inference.py" \
            --nnunet-raw "$nnUNet_raw" \
            --dataset-name "$DATASET_NAME" \
            --dataset-id "$DATASET_ID" \
            --seeds "${SEEDS[@]}" \
            --trainer-prefix nnUNetTrainerDataScaling_seed \
            --checkpoint "$PREDICTION_CHECKPOINT" \
            --device cuda \
            --output-dir "$SIZE_INFERENCE_LOGS"
done

# =====================================================================
# Summary
# =====================================================================
END_TIME=$(date +%s)
echo ""
echo "=== Data-scaling experiment complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Total wall clock: $(( END_TIME - START_TIME ))s ($(( (END_TIME - START_TIME) / 60 ))m)"
echo ""
echo "=== Outputs ==="
echo "  ${EXPERIMENT_DIR}/subsets_manifest.json"
echo "  ${TRAIN_TIMES_CSV}"
echo "  ${PREDICTION_TIMES_CSV}"
echo "  ${PREPROCESS_LOGS_DIR}/size{5,10,20,40,80,160,320,625}/"
echo "  ${INFERENCE_LOGS_DIR}/size{5,10,20,40,80,160,320,625}/"
echo ""
echo "  GPU monitor logs:"
echo "    ${LOGS_DIR}/gpu_monitor_size{SIZE}_s{42,43,44}.csv"
echo ""
echo "  Per-run nnU-Net training logs and checkpoints:"
echo "    ${nnUNet_results}/Dataset{001..008}_AUL*/"
echo ""
echo "  Prediction directories (24 total):"
echo "    ${nnUNet_results}/predictions_data_scaling_{SIZE}images_seed{42,43,44}_final/"
