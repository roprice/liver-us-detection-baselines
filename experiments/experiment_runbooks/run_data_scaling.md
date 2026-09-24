# Setup and training: data-scaling experiment

This runbook reproduces the main data-scaling experiment: train PlainConvUNet 2D models on nested, pathology-stratified AUL training pools of 5, 10, 20, 40, 80, 160, 320, and 625 cases. Each size is trained for the 150-epoch budget selected by the milestones pilot, across initialization seeds 42, 43, and 44, then evaluated on the same 110-image held-out test set.

The runner creates all seven reduced datasets from `Dataset001_AUL`; it never modifies the source dataset. It records the exact selected cases in `experiment_logs/data_scaling/subsets_manifest.json` and verifies the selection before reusing an existing generated subset.

Run the commands step-by-step to isolate any environment issue.

Design rationale (subset construction, evaluation checkpoint, split repair) is in [`experiment_design.md`](../experiment_design.md#data-scaling-experiment).

## Fixed conditions

| Setting | Value |
|---|---|
| Source dataset | AUL / `Dataset001_AUL` |
| Dataset-pool sizes | 5, 10, 20, 40, 80, 160, 320, 625 |
| Subset construction | Nested, stratified by malignant, benign, and mass-negative pathology; selection seed 42 |
| Test images | 110; copied unchanged from `Dataset001_AUL` to each reduced dataset |
| Fold | 0 |
| Initialization seeds | 42, 43, 44 |
| Epoch budget | 150 |
| Trainer | `nnUNetTrainerDataScaling_seed{42,43,44}` |
| Architecture | PlainConvUNet 2D |
| Evaluation checkpoint | `checkpoint_final.pth` |

## Dataset IDs

The full source pool remains `Dataset001_AUL`. The runner creates the following seven additional nnU-Net datasets; do not use IDs 2–8 for a different dataset in the same `nnUNet_raw` directory.

| Pool size | Dataset ID | Dataset name |
|---:|---:|---|
| 5 | 2 | `Dataset002_AUL_005` |
| 10 | 3 | `Dataset003_AUL_010` |
| 20 | 4 | `Dataset004_AUL_020` |
| 40 | 5 | `Dataset005_AUL_040` |
| 80 | 6 | `Dataset006_AUL_080` |
| 160 | 7 | `Dataset007_AUL_160` |
| 320 | 8 | `Dataset008_AUL_320` |
| 625 | 1 | `Dataset001_AUL` |

## Estimated cost as of September 2026

Roughly 33 GPU-hours of training on Verda's RTX 6000 Ada: 24 models × 150 epochs × ~33 s/epoch (from the [GPU training benchmark](../gpu_training_benchmark/gpu_training_benchmark.md)). That is about $37 on demand at $1.12/hr, or about half on spot. Preprocessing, predictions, and setup add to this. nnU-Net epochs are a fixed 250 iterations, so epoch time does not shrink with pool size.

## Server setup

All commands below assume a fresh Ubuntu GPU instance. The prior pilot used Verda's RTX PRO 6000 configuration.

### 1. Configure prompt (optional)

```sh
cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
PROMPTEOF
source ~/.bashrc
```

### 2. Install system dependencies

```sh
apt update
apt install python3-pip python3-dev unzip python-is-python3 tmux -y
```

### 3. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/liver-us-detection-baselines.git
cd liver-us-detection-baselines
```

Confirm the checkout contains these files before continuing:

```sh
ls experiments/run_data_scaling.sh
ls experiments/custom_trainers/nnUNetTrainerDataScaling.py
```

### 4. Resolve system Python package conflicts

```sh
rm -f /usr/lib/python3/dist-packages/typing_extensions.py
rm -rf /usr/lib/python3/dist-packages/typing_extensions-*.dist-info
rm -rf /usr/lib/python3/dist-packages/idna*
rm -rf /usr/lib/python3/dist-packages/click /usr/lib/python3/dist-packages/click-*.dist-info
```

### 5. Install Python dependencies

```sh
pip install -r requirements.txt --break-system-packages
```

The `nnunetv2==2.8.1` pin is the reproducibility anchor. `python3-dev` supplies `Python.h`, which PyTorch/Triton requires when nnU-Net enables `torch.compile` on current PyTorch releases. The runner also records the installed nnU-Net version, PyTorch version, source revision where available, and repository revision.

### 6. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/experiments/custom_trainers"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/experiments/custom_trainers"
ENVEOF
```

### 7. Download AUL from Zenodo

Dataset: Annotated Ultrasound Liver (AUL) images
DOI: [10.5281/zenodo.7272660](https://doi.org/10.5281/zenodo.7272660)
Citation: Xu, Y., Zheng, B., Liu, X., Wu, T., Ju, J., Wang, S., Lian, Y., Zhang, H., Liang, T., Sang, Y., Jiang, R., Wang, G., Ren, J., & Chen, T. (2022). Annotated Ultrasound Liver images [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7272660

This is a versioned Zenodo record, not the concept DOI (`10.5281/zenodo.7272659`). `zenodo_get 7272660` always resolves to the same archive files, even if new versions are published. Expected checksums:

| File | MD5 |
|---|---|
| `Benign.zip` | `c37fef0cb2730236a79ef57e5315995e` |
| `Malignant.zip` | `63894a9e5654a69c3b94bda84071dfb0` |
| `Normal.zip` | `a7e16299b2cf12ca4a6c3468d2e4978f` |

```sh
mkdir -p data/source
cd data/source
zenodo_get 7272660
md5sum *.zip
unzip Benign.zip -d AUL
unzip Malignant.zip -d AUL
unzip Normal.zip -d AUL
rm -rf AUL/__MACOSX
cd ../..
```

### 8. Convert AUL to nnU-Net format

```sh
python experiments/convert_aul.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL"
```

The conversion passes the images through without modification, renders liver and mass segmentation labels from the source polygons, and creates the fixed stratified 85/15 train/test split.

### 9. Verify source data

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 625
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 110
```

## Training and predictions

### 10. Run the data-scaling experiment

The runner creates or verifies the nested subsets, preprocesses each dataset, trains 24 models (8 sizes × 3 seeds), predicts all 24 final checkpoints on the fixed test set, and runs a per-image GPU inference benchmark for the final checkpoint at each size.

```sh
mkdir -p experiment_logs/data_scaling

# Start a persistent session so the long experiment survives an SSH disconnect
tmux new -s data-scaling
```

```sh
# Inside tmux
bash experiments/run_data_scaling.sh 2>&1 | tee experiment_logs/data_scaling/data_scaling.log
```

Detach without stopping the run with `Ctrl+b`, then `d`. Reattach after reconnecting with:

```sh
tmux attach -t data-scaling
```

### Optionally monitor progress

From a separate SSH shell:

```sh
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader

tail -f ~/liver-us-detection-baselines/experiment_logs/data_scaling/data_scaling.log
```

Use `Ctrl+C` to close `tail`.

## Output layout

The runner writes its experiment evidence below `experiment_logs/data_scaling/`:

```text
experiment_logs/data_scaling/
  data_scaling.log
  subsets_manifest.json
  logs/
    training_times.csv
    prediction_times.csv
    preprocessing/size{5,10,20,40,80,160,320,625}/
      nnUNetPlans.json
      dataset_fingerprint.json
      time_preprocess.txt
    inference/size{5,10,20,40,80,160,320,625}/
      inference_per_image_cuda.csv
      inference_summary_cuda.csv
      inference_settings_cuda.json
      time_gpu_inference_benchmark.txt
    gpu_monitor_size{SIZE}_s{SEED}.csv
    time_train_size{SIZE}_s{SEED}.txt
    time_predict_size{SIZE}_s{SEED}_final.txt
```

The runner writes the generated raw datasets to `nnUNet_raw/Dataset002_AUL_005` through `nnUNet_raw/Dataset008_AUL_320`.

## Checkpoints predicted per model

| Checkpoint file | Prediction directory |
|---|---|
| `checkpoint_final.pth` | `$nnUNet_results/predictions_data_scaling_{SIZE}images_seed{SEED}_final/` |

One directory per size and seed: 24 in total, e.g. `predictions_data_scaling_20images_seed43_final`. Each model's effective training/validation split is saved in its `fold_0/data_scaling_split.json`.

The custom trainer logs model parameter counts, post-initialization GPU memory, and final current/peak GPU memory in each nnU-Net `training_log_*.txt` under its result folder.

## Resume behavior

The runner is safe to re-run after an interruption:

- Existing subsets must exactly match the deterministic selection or the runner stops rather than overwrite them.
- Completed preprocessing stages are marked under `experiment_logs/data_scaling/logs/preprocessing/` and skipped.
- A training run with `checkpoint_final.pth` already present is skipped; incomplete training is resumed through nnU-Net's `--c` option.
- Complete 110-mask prediction directories and completed inference benchmark summaries are skipped. Incomplete prediction directories are removed and regenerated.

If subset creation is interrupted, delete only the partial generated directory (`Dataset002_AUL_005` through `Dataset008_AUL_320`) and rerun. Never delete `Dataset001_AUL`.

Checkpoints from before the split-repair policy are rejected. To restart such a run, move its model folders, predictions, inference results, and run metrics out of the active result and log locations. Keep the raw datasets, preprocessed data, `splits_final.json` files, and preprocessing markers so preprocessing is not repeated.

## Verification

### 11. Verify completion

```sh
# The runner prints this final line:
tail -20 experiment_logs/data_scaling/data_scaling.log
# Look for: "Data-scaling experiment complete"

# Exact nested selection and class balance for every dataset pool
python -m json.tool experiment_logs/data_scaling/subsets_manifest.json

# One row per trained model: header + 24 rows
wc -l experiment_logs/data_scaling/logs/training_times.csv

# One row per prediction: header + 24 rows
wc -l experiment_logs/data_scaling/logs/prediction_times.csv

# Three prediction directories per training-pool size
for SIZE in 5 10 20 40 80 160 320 625; do
  COUNT=$(find "$nnUNet_results" -maxdepth 1 -type d \
    -name "predictions_data_scaling_${SIZE}images_seed*_final" | wc -l)
  echo "${SIZE}: ${COUNT} prediction directories"  # expect 3
done

# Each completed prediction directory contains the unchanged 110-image test set
for DIR in "$nnUNet_results"/predictions_data_scaling_*images_seed*_final; do
  COUNT=$(find "$DIR" -maxdepth 1 -type f -name '*.png' | wc -l)
  echo "${DIR}: ${COUNT} masks"  # expect 110
done

# One inference benchmark summary per data size
find experiment_logs/data_scaling/logs/inference -name inference_summary_cuda.csv | wc -l  # expect 8
```

## Download results

Archive the repository evidence and the nnU-Net working directories before releasing the GPU instance:

```sh
cd ~
tar czf data_scaling_full.tar.gz \
  liver-us-detection-baselines/experiment_logs/data_scaling/ \
  nnUNet_raw/ \
  nnUNet_preprocessed/ \
  nnUNet_results/
```

Then, on the local computer:

```sh
scp root@<server-ip>:~/data_scaling_full.tar.gz /tmp/
tar xzf /tmp/data_scaling_full.tar.gz
```
