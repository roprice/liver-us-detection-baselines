#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export nnUNet_raw="${nnUNet_raw:-${REPO_DIR}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_DIR}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_DIR}/nnUNet_results}"
export nnUNet_extTrainer="${REPO_DIR}/experiments/custom_trainers"
export nnUNet_compile=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-mps}"
INPUT_DIR="${nnUNet_raw}/Dataset001_AUL/imagesTs"
LOG_DIR="${REPO_DIR}/experiment_logs/data_scaling_best_mac"
DATASETS=(Dataset002_AUL_005 Dataset003_AUL_010 Dataset004_AUL_020 Dataset005_AUL_040 Dataset006_AUL_080 Dataset007_AUL_160 Dataset008_AUL_320 Dataset001_AUL)
SIZES=(5 10 20 40 80 160 320 625)
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
        echo "Usage: bash $0 [--dry-run]"
        echo "Environment: DEVICE=mps|cpu (default mps), PYTHON=python3, nnUNet_raw, nnUNet_preprocessed, nnUNet_results"
        exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "Too many arguments" >&2; exit 2; }
case "$DEVICE" in mps|cpu) ;; *) echo "DEVICE must be mps or cpu" >&2; exit 2 ;; esac

"$PYTHON" - "$INPUT_DIR" "${DATASETS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

images = Path(sys.argv[1])
inputs = sorted(images.glob('*.png'))
if len(inputs) != 110 or any(not p.name.endswith('_0000.png') for p in inputs):
    raise SystemExit(f'Expected 110 single-channel *_0000.png test images in {images}')
results = Path(os.environ['nnUNet_results'])
expected = {
    results / dataset / f'nnUNetTrainerDataScaling_seed{seed}__nnUNetPlans__2d' / 'fold_0' / 'checkpoint_best.pth'
    for dataset in sys.argv[2:] for seed in (42, 43, 44)
}
found = set(results.glob('Dataset00*_AUL*/nnUNetTrainerDataScaling_seed*__nnUNetPlans__2d/fold_0/checkpoint_best.pth'))
if found != expected:
    raise SystemExit(f'Expected exactly 24 checkpoints. Missing: {sorted(map(str, expected - found))}; unexpected: {sorted(map(str, found - expected))}')
for checkpoint in sorted(expected):
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SystemExit(f'Missing or empty checkpoint: {checkpoint}')
    model = checkpoint.parent.parent
    metadata = json.loads((model / 'dataset.json').read_text())
    plans = json.loads((model / 'plans.json').read_text())
    if metadata['file_ending'] != '.png' or set(metadata['channel_names']) != {'0'}:
        raise SystemExit(f'Unexpected input format: {model}')
    if '2d' not in plans['configurations']:
        raise SystemExit(f'Missing 2d configuration: {model}')
print('Preflight: 24 checkpoints, model metadata, and 110 test images found.')
PY

if [ "$DRY_RUN" -eq 0 ]; then
    "$PYTHON" - "$DEVICE" <<'PY'
import os
import sys
from importlib.metadata import version

import torch
from nnunetv2.inference.predict_from_raw_data import predict_entry_point
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

if version('nnunetv2') != '2.8.1':
    raise SystemExit('Use nnunetv2==2.8.1, as pinned in requirements.txt')
for seed in (42, 43, 44):
    name = f'nnUNetTrainerDataScaling_seed{seed}'
    trainer = recursive_find_trainer_class_by_name(name)
    if trainer is None:
        raise SystemExit(f'Cannot discover custom trainer: {name}')
if sys.argv[1] == 'mps' and not torch.backends.mps.is_available():
    raise SystemExit('MPS is unavailable. Use native arm64 Python/PyTorch, or set DEVICE=cpu.')
print(f'Runtime: torch {torch.__version__}, nnunetv2 {version("nnunetv2")}, device {sys.argv[1]}')
PY
    mkdir -p "$LOG_DIR"
fi

verify_masks() {
    "$PYTHON" - "$INPUT_DIR" "$1" <<'PY'
import sys
from pathlib import Path
from PIL import Image

inputs, outputs = map(Path, sys.argv[1:])
expected = {p.name.replace('_0000.png', '.png') for p in inputs.glob('*_0000.png')}
actual = {p.name for p in outputs.glob('*.png')}
if actual != expected:
    raise SystemExit(f'Mask names differ in {outputs}: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
for name in sorted(expected):
    with Image.open(inputs / name.replace('.png', '_0000.png')) as image:
        size = image.size
    with Image.open(outputs / name) as mask:
        if mask.size != size:
            raise SystemExit(f'Incorrect mask dimensions: {outputs / name}')
        mask.verify()
print(f'Verified {len(expected)} masks in {outputs}')
PY
}

for INDEX in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$INDEX]}"
    SIZE="${SIZES[$INDEX]}"
    for SEED in 42 43 44; do
        TRAINER="nnUNetTrainerDataScaling_seed${SEED}"
        OUTPUT_DIR="${nnUNet_results}/predictions_data_scaling_${SIZE}images_seed${SEED}_best"
        MARKER="${OUTPUT_DIR}/.prediction_complete"
        COMMAND=("$PYTHON" -c 'from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()'
            -i "$INPUT_DIR" -o "$OUTPUT_DIR" -d "$DATASET" -c 2d -p nnUNetPlans
            -f 0 -tr "$TRAINER" -chk checkpoint_best.pth -device "$DEVICE"
            -npp 1 -nps 1 -step_size 0.5)
        if [ "$DRY_RUN" -eq 1 ]; then
            printf '%q ' "${COMMAND[@]}"
            printf '\n'
            continue
        fi
        if [ -f "$MARKER" ]; then
            verify_masks "$OUTPUT_DIR"
            echo "SKIP: size ${SIZE}, seed ${SEED} already complete"
            continue
        fi
        echo "Predicting size ${SIZE}, seed ${SEED}, device ${DEVICE}"
        START_TIME=$(date +%s)
        "${COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/size${SIZE}_seed${SEED}_${DEVICE}.log"
        verify_masks "$OUTPUT_DIR"
        printf '%s\n' "$DEVICE" > "$MARKER"
        END_TIME=$(date +%s)
        echo "Completed size ${SIZE}, seed ${SEED}: $((END_TIME - START_TIME)) seconds"
    done
done

echo "Done: all 24 checkpoint jobs processed."
