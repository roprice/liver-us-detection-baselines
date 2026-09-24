# Setup and training: manually run preliminary milestones test

This runbook reproduces the first test in this study, which saves checkpoints across a sweep of intervals.

Early checkpoints reflect a learning rate schedule tuned for 1000 epochs, so they're not comparable to independent shorter training runs. Still, the sweep reveals where Dice gains plateau under this schedule, which is sufficient to set an epoch budget for subsequent experiments.

Run the commands step-by-step to isolate any environment issue.

## Fixed conditions

| Setting | Value |
|---|---|
| Dataset | AUL / `Dataset001_AUL` |
| Training images | 625 |
| Test images | 110 |
| Fold | 0 |
| Initialization seeds | 42, 43, 44 |
| Epoch budget | 1,000 |
| Trainer | `nnUNetTrainerMilestones_seed{42,43,44}` |
| Architecture | PlainConvUNet 2D |

Each seed saves multiple milestone checkpoints (epochs 50 through 1000), a best-mean-Dice checkpoint, and a best-mass-Dice checkpoint (both EMA-smoothed). After training, the runner triggers prediction on all 27 checkpoints on the 110-image test set; it runs a per-image GPU inference benchmark (`experiments/benchmark_gpu_inference.py`).

## Estimated cost as of September 2026

On Verda.com's RTX PRO 6000 $0.95/hr spot pricing, roughly $10 and 11 GPU-hours for all three seeds including predictions. That's mostly training time but factors in setup and download and deletion times.

## Server setup

All commands run on a fresh Verda GPU instance (NVIDIA RTX PRO 6000, 96 GiB VRAM) running Ubuntu. 

Once you have provisioned the instance and connected to it by SSH, run the following commands from a single shell. During the training in step 10, you can open a new shell if you wish to monitor progress; that won't be logged.

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
apt install python3-pip unzip python-is-python3 tmux -y
```

### 3. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/liver-us-detection-baselines.git
cd liver-us-detection-baselines
```

Confirm the checkout contains these files before continuing:

```sh
ls experiments/run_milestones_pilot.sh
ls experiments/custom_trainers/nnUNetTrainerMilestones.py
ls experiments/benchmark_gpu_inference.py
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

The explicit nnU-Net version pin (nnunetv2==2.8.1) is the reproducibility anchor. The runner logs the installed nnU-Net version, the PyTorch version, and the repo’s git SHA during its environment block; the remaining dependencies are pinned in requirements.txt.

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

This is a versioned Zenodo record, not the floating concept DOI (`10.5281/zenodo.7272659`), so `zenodo_get 7272660` always resolves to the same fixed archive files regardless of any future dataset versions published under the same concept DOI. File checksums as of this record:

| File | MD5 |
|---|---|
| `Benign.zip` | `c37fef0cb2730236a79ef57e5315995e` |
| `Malignant.zip` | `63894a9e5654a69c3b94bda84071dfb0` |
| `Normal.zip` | `a7e16299b2cf12ca4a6c3468d2e4978f` |

```sh
mkdir -p data/source
cd data/source
zenodo_get 7272660
md5sum *.zip  # verify against the checksums above
unzip '*.zip' -d AUL
rm -rf AUL/__MACOSX
cd ../..
```

### 8. Convert to nnU-Net format

```sh
python experiments/convert_aul.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL"
```

Images pass through without modification; only the segmentation labels are rendered from the annotated polygons.

### 9. Verify input data

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 625
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 110
```

## Training and predictions

### 10. Run the preliminary experiment

The runner preprocesses once, trains all three seeds, predicts all nine checkpoints per seed, and runs the per-image GPU inference benchmark. It writes all timing, memory, and configuration evidence to `logs/` (see "Output layout" below).

```sh
# Start a persistent session so training survives an SSH disconnect
tmux new -s training
```

```sh
# Inside tmux
bash experiments/run_milestones_pilot.sh 2>&1 | tee preliminary_milestones_test.log
```

Detach without stopping the run with `Ctrl+b`, then `d`. Reattach after reconnecting with:

```sh
tmux attach -t training
```

### Optionally monitor progress

From a separate SSH shell:

```sh
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader

tail -f ~/liver-us-detection-baselines/preliminary_milestones_test.log
```

Use `Ctrl+C` to close `tail`.

## Output layout

```text
preliminary_milestones_test.log
logs/
  training/
    training_times.csv
    prediction_times.csv
    predict_defaults.txt
    nnUNetPlans.json
    dataset_fingerprint.json
    time_preprocess.txt
    time_gpu_inference_benchmark.txt
    gpu_monitor_s{SEED}.csv
    time_train_s{SEED}.txt
    time_predict_s{SEED}_{label}.txt
  inference/
    inference_per_image_cuda.csv
    inference_summary_cuda.csv
    inference_settings_cuda.json
```

The log also records the environment, checkpoint file sizes, and model footprint.

## Checkpoints saved per seed

| Checkpoint file | Prediction label |
|---|---|
| `checkpoint_epoch50.pth` | `epoch50` |
| `checkpoint_epoch100.pth` | `epoch100` |
| `checkpoint_epoch150.pth` | `epoch150` |
| `checkpoint_epoch300.pth` | `epoch300` |
| `checkpoint_epoch500.pth` | `epoch500` |
| `checkpoint_epoch750.pth` | `epoch750` |
| `checkpoint_best.pth` | `best` |
| `checkpoint_best_mass.pth` | `best_mass` |
| `checkpoint_final.pth` | `final` |

Prediction directories use the naming `predictions_milestones_625images_seed{SEED}_{label}`, e.g. `predictions_milestones_625images_seed42_epoch500`, repeated for each seed (27 directories total).

## Verification

### 11. Verify completion

```sh
# The runner prints this final line:
tail -20 preliminary_milestones_test.log
# Look for: "Preliminary experiment complete"

# 27 prediction directories (9 checkpoints × 3 seeds)
for SEED in 42 43 44; do
  echo "Seed $SEED:"
  ls -d "$nnUNet_results"/predictions_milestones_625images_seed${SEED}_* | wc -l  # expect 9
done

# Each prediction directory has 110 PNG masks
for DIR in "$nnUNet_results"/predictions_milestones_625images_seed*_*; do
  COUNT=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
  echo "$DIR: $COUNT files"  # expect 110 each
done

# Per-seed training wall clock
cat logs/training/training_times.csv  # header + 3 rows

# GPU inference benchmark outputs
ls logs/inference/  # per-image, summary, settings CSVs/JSON
```

## Download results

Archive the repository, logs, and nnU-Net working directories before releasing the GPU instance:

```sh
cd ~
tar czf preliminary_milestones_full.tar.gz \
  liver-us-detection-baselines/ \
  nnUNet_raw/ \
  nnUNet_preprocessed/ \
  nnUNet_results/
```

Then, on the local computer:

```sh
cd ~/Projects/liver-us-detection-baselines

scp root@<server-ip>:~/preliminary_milestones_full.tar.gz /tmp/
tar xzf /tmp/preliminary_milestones_full.tar.gz
```

## Manual cleanup

Unlike later experiments, this runner writes to `logs/` rather than `experiment_logs/<experiment_name>/`. Move the outputs so the analysis scripts find them:

```sh
mkdir -p experiment_logs/milestones_pilot
mv logs preliminary_milestones_test.log experiment_logs/milestones_pilot/
```

Resulting layout:

```text
experiment_logs/milestones_pilot/
  logs/
    training/
    inference/
  preliminary_milestones_test.log
```
