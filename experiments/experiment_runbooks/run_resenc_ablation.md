# Setup and training: ResEnc architecture ablation

This planned experiment compares nnU-Net's ResEnc M preset with the size-matched PlainConvUNet result from the data-scaling experiment. It trains one AUL dataset size for 150 epochs with initialization seeds 42, 43, and 44, predicts each `checkpoint_final.pth` on the same 110-image held-out test set, and runs the same per-image GPU inference benchmark.

The comparison size is **TBD pending analysis of the data-scaling experiment**. The runner currently defaults to 625 cases as a placeholder. Set `ABLATION_DATASET_SIZE` explicitly when the final size is selected.

Design rationale (why ResEnc M, preset effects, trainer behavior, comparison checklist) is in [`experiment_design.md`](../experiment_design.md#resenc-architecture-ablation).

## Fixed and pending conditions

| Setting | Value |
|---|---|
| Dataset size | **TBD; default placeholder 625** |
| Supported sizes | 5, 10, 20, 40, 80, 160, 320, 625 |
| Dataset selection | Reuse the exact size-corresponding data-scaling dataset |
| Test images | Same 110 held-out AUL images |
| Fold | 0 |
| Initialization seeds | 42, 43, 44 |
| Epoch budget | 150 |
| Trainer | `nnUNetTrainerResEncAblation_seed{42,43,44}` |
| Planner | `nnUNetPlannerResEncM` |
| Plans | `nnUNetResEncUNetMPlans` |
| Architecture | ResidualEncoderUNet 2D, ResEnc M preset |
| Evaluation checkpoint | `checkpoint_final.pth` |
| nnU-Net | 2.8.1 |

## Estimated cost as of September 2026

Not yet measured. The PlainConvUNet equivalent is about 4 GPU-hours of training for three seeds on Verda's RTX 6000 Ada (150 epochs × ~33 s/epoch, from the [GPU training benchmark](../gpu_training_benchmark/gpu_training_benchmark.md)). ResEnc M will likely take somewhat longer per epoch. Update this section after the first run.

## Dataset selection

The runner maps `ABLATION_DATASET_SIZE` to the same datasets used by the data-scaling experiment:

| Size | Dataset |
|---:|---|
| 5 | `Dataset002_AUL_005` |
| 10 | `Dataset003_AUL_010` |
| 20 | `Dataset004_AUL_020` |
| 40 | `Dataset005_AUL_040` |
| 80 | `Dataset006_AUL_080` |
| 160 | `Dataset007_AUL_160` |
| 320 | `Dataset008_AUL_320` |
| 625 | `Dataset001_AUL` |

The ablation runner never creates or changes a dataset. The source dataset is AUL ([10.5281/zenodo.7272660](https://doi.org/10.5281/zenodo.7272660)); see the data-scaling runbook for download, citation, and checksums. Restore the selected raw dataset and `experiment_logs/data_scaling/subsets_manifest.json` from the completed data-scaling experiment. The runner compares the exact training case IDs, source `dataset.json` hash, and held-out test filenames against that evidence before planning.

## Fresh server setup

### 1. Install system dependencies

```sh
apt update
apt install python3-pip python3-dev unzip python-is-python3 tmux -y
```

### 2. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/liver-us-detection-baselines.git
cd liver-us-detection-baselines

ls experiments/run_resenc_ablation.sh
ls experiments/custom_trainers/nnUNetTrainerResEncAblation.py
```

### 3. Resolve system Python package conflicts

```sh
rm -f /usr/lib/python3/dist-packages/typing_extensions.py
rm -rf /usr/lib/python3/dist-packages/typing_extensions-*.dist-info
rm -rf /usr/lib/python3/dist-packages/idna*
rm -rf /usr/lib/python3/dist-packages/click /usr/lib/python3/dist-packages/click-*.dist-info
```

### 4. Install Python dependencies and record the artifacts

Run this in a fresh environment so the pip report contains the installed nnU-Net artifact rather than an empty “already satisfied” result.

```sh
python -m pip install \
  --report "$HOME/pip-install-report.json" \
  -r requirements.txt \
  --break-system-packages
```

The runner requires this report before starting expensive work. It records the exact nn-U-Net artifact URL and SHA256, the verified upstream `v2.8.1` tag SHA, the installed version, and the study repository SHA and dirty state.

### 5. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/experiments/custom_trainers"
export PIP_INSTALL_REPORT="$HOME/pip-install-report.json"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/experiments/custom_trainers"
export PIP_INSTALL_REPORT="$HOME/pip-install-report.json"
ENVEOF
```

### 6. Restore the data-scaling dataset and evidence

Restore these paths from the completed data-scaling experiment before starting the ablation:

```text
nnUNet_raw/Dataset001_AUL/
experiment_logs/data_scaling/subsets_manifest.json
```

If a reduced size is selected, also restore its corresponding `Dataset002_AUL_005` through `Dataset008_AUL_320` directory. Do not reconstruct a reduced subset independently.

Verify the placeholder 625-case input with:

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 625
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 110
python -m json.tool experiment_logs/data_scaling/subsets_manifest.json >/dev/null
```

The runner stops if the selected training case IDs, source dataset metadata, or test filenames differ from the data-scaling evidence.

## Run the ablation

Set the selected size explicitly even when using the current 625-case placeholder:

```sh
cd ~/liver-us-detection-baselines
mkdir -p experiment_logs/resenc_ablation

tmux new -s resenc-ablation
```

Inside tmux:

```sh
ABLATION_DATASET_SIZE=625 \
  bash experiments/run_resenc_ablation.sh 2>&1 | \
  tee experiment_logs/resenc_ablation/resenc_ablation.log
```

Detach with `Ctrl+b`, then `d`. Reattach with:

```sh
tmux attach -t resenc-ablation
```

The runner plans and preprocesses with:

```sh
nnUNetv2_plan_and_preprocess -d DATASET_ID \
  -pl nnUNetPlannerResEncM \
  --verify_dataset_integrity
```

It passes `-p nnUNetResEncUNetMPlans` to both training and prediction and passes the same plans identifier to the inference benchmark.

## Outputs

```text
experiment_logs/resenc_ablation/
  resenc_ablation.log
  logs/
    environment.json
    pip-install-report.json
    predict_defaults.txt
    training_times.csv
    prediction_times.csv
    preprocessing/size{SIZE}/
      nnUNetResEncUNetMPlans.json
      dataset_fingerprint.json
      time_preprocess.txt
    inference/size{SIZE}/
      inference_per_image_cuda.csv
      inference_summary_cuda.csv
      inference_settings_cuda.json
      time_gpu_inference_benchmark.txt
    gpu_monitor_size{SIZE}_s{42,43,44}.csv
    time_train_size{SIZE}_s{42,43,44}.txt
    time_predict_size{SIZE}_s{42,43,44}_final.txt
```

Model results are written below:

```text
$nnUNet_results/Dataset.../
  nnUNetTrainerResEncAblation_seed{SEED}__nnUNetResEncUNetMPlans__2d/fold_0/
```

## Checkpoints predicted per seed

| Checkpoint file | Prediction directory |
|---|---|
| `checkpoint_final.pth` | `$nnUNet_results/predictions_resenc_ablation_{SIZE}images_seed{SEED}_final/` |

Three directories in total, e.g. `predictions_resenc_ablation_625images_seed42_final`. These names cannot collide with PlainConvUNet data-scaling outputs.

## Resume behavior

The runner is safe to restart:

- Existing valid ResEnc plans and completed preprocessing are skipped.
- An incomplete training run resumes through nnU-Net's `--c` option.
- A completed final checkpoint is reused only if its split-policy marker, trainer name, plans identifier, and completed epoch match this experiment.
- A complete 110-mask prediction directory is skipped; an incomplete directory is removed and regenerated.
- A completed inference summary is skipped.

The pip installation report remains mandatory on every invocation so the provenance record cannot silently disappear during a resumed run.

## Verify completion

```sh
tail -20 experiment_logs/resenc_ablation/resenc_ablation.log
# Expect: ResEnc architecture ablation complete

python -m json.tool experiment_logs/resenc_ablation/logs/environment.json

wc -l experiment_logs/resenc_ablation/logs/training_times.csv
# Expect 4: header plus three seeds

wc -l experiment_logs/resenc_ablation/logs/prediction_times.csv
# Expect 4: header plus three seeds

find "$nnUNet_results" -maxdepth 1 -type d \
  -name 'predictions_resenc_ablation_*images_seed*_final' | wc -l
# Expect 3

find experiment_logs/resenc_ablation/logs/inference \
  -name inference_summary_cuda.csv | wc -l
# Expect 1
```

## Download results

```sh
cd ~
tar czf resenc_ablation_full.tar.gz \
  liver-us-detection-baselines/experiment_logs/resenc_ablation/ \
  nnUNet_raw/ \
  nnUNet_preprocessed/ \
  nnUNet_results/
```

Then, on the local computer:

```sh
scp root@<server-ip>:~/resenc_ablation_full.tar.gz /tmp/
tar xzf /tmp/resenc_ablation_full.tar.gz
```
