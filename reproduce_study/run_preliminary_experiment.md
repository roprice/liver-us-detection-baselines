# Setup and training: 1,000-epoch preliminary experiment

This experiment determine the right epoch budget for liver mass detection using this data set.

## Fixed conditions

| Setting | Value |
|---|---|
| Dataset | AUL / `Dataset001_LiverUS` |
| Training images | 625 |
| Test images | 110 |
| Fold | 0 |
| Initialization seeds | 42, 43, 44 |
| Epoch budget | 1,000 |
| Trainer | `nnUNetTrainer1000Milestones_s{42,43,44}` |
| Architecture | PlainConvUNet 2D |

Each of the three training runs saves milestone checkpoints at epochs 150, 300, 500, 750, and 1,000, plus the overall best-joint, best-mass, and final checkpoints. After training, the runner predicts from all seven checkpoints per seed on the 110-image test set (7 checkpoints × 3 seeds × 110 images = 2,310 prediction runs), then runs a per-image GPU inference benchmark.

## Estimated cost

Roughly $13 and 6–12 GPU-hours for all three seeds including predictions. Roughly: 95%+ of that time is spent on training, 4% on setup and downloading, 1% on predictions.

## Server setup

All commands run on a fresh Verda GPU instance (NVIDIA RTX PRO 6000, 96 GiB VRAM) running Ubuntu.

### 1. Install system dependencies

```sh
apt update
apt install python3-pip unzip python-is-python3 -y
```

### 2. Clone the repository

```sh
cd ~
git clone https://github.com/roprice/liver-us-detection-baselines.git
cd liver-us-detection-baselines
```

Confirm this checkout contains `training/run_preliminary_1000_epochs.sh`, the `training/custom_trainers/` directory, and `training/benchmark_gpu_inference.py` before continuing.

### 3. Resolve system Python package conflicts

```sh
rm -f /usr/lib/python3/dist-packages/typing_extensions.py
rm -rf /usr/lib/python3/dist-packages/typing_extensions-*.dist-info
rm -rf /usr/lib/python3/dist-packages/idna*
```

### 4. Install Python dependencies

```sh
pip install -r requirements.txt --break-system-packages
pip install "nnunetv2==2.8.1" idna --break-system-packages
```

The explicit nnU-Net version pin is required for reproducibility. The runner captures the installed nnU-Net version, source revision, and dependency versions during its environment block, so no manual version logging is needed here.

### 5. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/training/custom_trainers"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Persist for SSH reconnects
cat >> ~/.bashrc << 'ENVEOF'
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/training/custom_trainers"
ENVEOF
```

### 6. Download AUL from Zenodo

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
pip install zenodo-get --break-system-packages
zenodo_get 7272660
md5sum *.zip  # verify against the checksums above
unzip '*.zip' -d AUL
rm -rf AUL/__MACOSX
cd ../..
```

### 7. Convert to nnU-Net format

```sh
python training/convert_aul.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_LiverUS"
```

Images pass through without modification; only the segmentation labels are rendered from the annotated polygons.

### 8. Verify input data

```sh
ls "$nnUNet_raw/Dataset001_LiverUS/imagesTr" | wc -l  # expect 625
ls "$nnUNet_raw/Dataset001_LiverUS/imagesTs" | wc -l  # expect 110
```

## Training

### 9. Run the preliminary experiment

```sh
nohup bash training/run_preliminary_1000_epochs.sh \
  > preliminary_1000ep.log 2>&1 &

tail -f preliminary_1000ep.log
```

The runner preprocesses once, trains all three seeds, predicts from all seven checkpoints per seed, and then runs the per-image GPU inference benchmark. It captures GPU samples (`nvidia-smi`), process memory and CPU time (`/usr/bin/time -v`), per-checkpoint prediction timing, checkpoint file sizes, model footprint, and nnU-Net auto-configuration. All of this is written to `logs/`; there are no manual logging steps to run separately.

### Checkpoints saved per seed

| Checkpoint file | Prediction label |
|---|---|
| `checkpoint_ep150.pth` | `ep150` |
| `checkpoint_ep300.pth` | `ep300` |
| `checkpoint_ep500.pth` | `ep500` |
| `checkpoint_ep750.pth` | `ep750` |
| `checkpoint_best_joint.pth` | `joint` |
| `checkpoint_best_mass.pth` | `mass` |
| `checkpoint_final.pth` | `final` |

Prediction directories use the naming `predictions_625_s{SEED}_1000ep_{label}`, e.g. `predictions_625_s42_1000ep_ep150`, repeated for each seed (21 directories total).

## Verification

### 10. Verify completion

```sh
tail -20 preliminary_1000ep.log
# Look for: "Preliminary experiment complete"

# 21 prediction directories (7 checkpoints × 3 seeds)
for SEED in 42 43 44; do
  echo "Seed $SEED:"
  ls -d "$nnUNet_results"/predictions_625_s${SEED}_1000ep_* | wc -l  # expect 7
done

# Each prediction directory has 110 PNG masks
for DIR in "$nnUNet_results"/predictions_625_s*_1000ep_*; do
  COUNT=$(ls "$DIR"/*.png 2>/dev/null | wc -l)
  echo "$DIR: $COUNT files"  # expect 110 each
done

# Per-seed training wall clock
cat logs/training/training_times.csv  # header + 3 rows

# GPU inference benchmark outputs
ls logs/inference/  # per-image, summary, settings CSVs/JSON
```

## Download results

Do not selectively download. Download the full set of evidence needed to reconstruct and audit the run: the repo (code plus the logs and terminal output it accumulated during the run), all three nnU-Net working directories, and the shell history of every command actually executed. SSH key material is deliberately excluded.

```sh
# On the GPU server
cd ~

# Flush this session's in-memory history to ~/.bash_history before archiving.
# Bash only writes history to disk on shell exit by default, and nohup'd work
# in a still-open session may not have been flushed yet.
history -a

tar czf preliminary_experiment_full.tar.gz \
  liver-us-detection-baselines/ \
  nnUNet_raw/ \
  nnUNet_preprocessed/ \
  nnUNet_results/ \
  .bash_history
```

Then, on the Mac:

```sh
cd ~/Projects/liver-us-detection-baselines

scp root@<server-ip>:~/preliminary_experiment_full.tar.gz /tmp/
tar xzf /tmp/preliminary_experiment_full.tar.gz
```

## What happens next

Evaluate the predictions locally. Compare Dice, centroid-based detection, IoU-based detection, and triage across epochs 150, 300, 500, 750, 1000 for all three seeds. The results determine:

1. **Epoch budget for the full sweep.** If performance plateaus by epoch 150, use 150. If the learning curve shifts under the current preprocessing, adopt the epoch where detection and triage stabilize.
2. **Primary detection definition.** Evaluate both centroid-based and IoU > 0 detection on the same predictions. Choose one as the primary metric; the other remains a secondary analysis for benchmark comparability.

The runner already produced the GPU inference benchmark. To record comparable CPU inference numbers, rerun the benchmark script later on the Mac with the same checkpoints, images, and settings, changing only `--device` and `--output-dir`:

```sh
python training/benchmark_gpu_inference.py \
  --nnunet-raw "$nnUNet_raw" \
  --dataset-name Dataset001_LiverUS \
  --dataset-id 1 \
  --seeds 42 43 44 \
  --trainer-prefix nnUNetTrainer1000Milestones_s \
  --checkpoint checkpoint_final.pth \
  --device cpu \
  --output-dir logs/inference_cpu
```

Do NOT proceed to the full 8-size sweep until both decisions are made.

## Environment variable notes

nnU-Net environment variables do not persist between SSH sessions unless written to `.bashrc` (step 5). If you reconnect and did not run that step, re-export before running anything:

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/training/custom_trainers"
```

## Spot/preemptible instance notes

The runner passes `--c` to `nnUNetv2_train`, so training resumes from the last checkpoint if preempted. Re-export environment variables (or rely on `.bashrc`) and rerun the same command. Preprocessing is skipped if already done. Use the same GPU type across all three seeds for consistency.
