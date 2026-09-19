# Setup and training: manually run preliminary milestones test

This document lets you manually reproduce the first test in this study, which saves checkpoints across a sweep of intervals.

Early checkpoints reflect a learning rate schedule tuned for 1000 epochs; they unlike independent shorter training runs. Still, the sweep reveals where Dice gains plateau under this schedule, which is sufficient to set an epoch budget for subsequent experiments.

Because your environment may deviate in unpredictable ways, run these commands step-by-step to pinpoint and resolve issues if they come up.

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

Each seed saves multiple milestone checkpoints (epochs 50 through 1000), a best-mean-Dice checkpoint, and a best-mass-Dice checkpoint (both EMA-smoothed). After training, the runner triggers prediction on all 27 checkpoints on the 110-image test set; it runs a per-image GPU inference benchmark (`training/benchmark_gpu_inference.py`).

## Estimated cost as of September 2026

On Verda.com's RTX PRO 6000 $0.95/hr spot pricing, roughly $10 and 11 GPU-hours for all three seeds including predictions. That's mostly training time but factors in setup and download and deletion times.

## Server setup

All commands run on a fresh Verda GPU instance (NVIDIA RTX PRO 6000, 96 GiB VRAM) running Ubuntu. 

Once you have provisioned the instance and connected to it by SSH, run the following commands from a single shell. During the training in step 10, you can open a new shell if you wish to monitor progress; that won't be logged.

### 1. Configure prompt and history

```sh
# Capture every subsequent command in ~/.bash_history 
export PROMPT_COMMAND='history -a'

# More visible prompt with a timestamp, save each command to disk 
cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
export HISTTIMEFORMAT='%F %T '
export PROMPT_COMMAND='history -a'
PROMPTEOF

# Apply to current shell (new shells pick it up from ~/.bashrc)
source ~/.bashrc
```

Run this before any other command so command history is captured from the very start of the session.

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

Confirm this checkout contains `training/run_preliminary_milestones_test.sh`, the `training/custom_trainers/` directory, and `training/benchmark_gpu_inference.py` before continuing.

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
pip install "nnunetv2==2.8.1" idna --break-system-packages
```

The explicit nnU-Net version pin (nnunetv2==2.8.1) is the reproducibility anchor. The runner logs the installed nnU-Net version, the PyTorch version, and the repo’s git SHA during its environment block; the remaining dependencies are pinned in requirements.txt.

### 6. Configure nnU-Net directories

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
pip install zenodo-get --break-system-packages
zenodo_get 7272660
md5sum *.zip  # verify against the checksums above
unzip '*.zip' -d AUL
rm -rf AUL/__MACOSX
cd ../..
```

### 8. Convert to nnU-Net format

```sh
python training/convert_aul.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL"
```

Images pass through without modification; only the segmentation labels are rendered from the annotated polygons.

### 9. Verify input data

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 625
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 110
```

## Training & Predictions

### 10. Run the preliminary experiment

```sh
# Start a persistent session so training survives an SSH disconnect
tmux new -s training

# Inside tmux, run the runner, teeing output to a file you can tail later
bash training/run_preliminary_milestones_test.sh 2>&1 | tee preliminary_milestones_test.log
```

Detach from tmux without stopping training with `Ctrl+b` then `d`. Reattach after a reconnect (or from any other SSH session) with `tmux attach -t training`. The runner keeps running inside tmux regardless of the SSH connection.

#### Optionally monitor progress

From a separate SSH session (so these commands do not interfere with the training shell):

```sh
tail -f preliminary_milestones_test.log

nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
```

The runner preprocesses once, trains all three seeds, predicts from all nine checkpoints per seed, and then runs the per-image GPU inference benchmark. It captures GPU samples (`nvidia-smi`), process memory and CPU time (`/usr/bin/time -v`), per-checkpoint prediction timing, checkpoint file sizes, model footprint, and nnU-Net auto-configuration. All of this is written to `logs/`; there are no manual logging steps to run separately.

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
# Reattach to the tmux session, or tail the log file
tmux attach -t training
# or, if detached:
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

Rather than selectively downloading, download the full set of evidence needed to reconstruct and audit the run: the repo (code plus the logs and terminal output it accumulated during the run), all three nnU-Net working directories, and the shell history of every command actually executed. SSH key material is deliberately excluded.

```sh
# On the GPU server
cd ~

# Flush this session's in-memory history to ~/.bash_history before archiving.
# Bash only writes history to disk on shell exit by default, and work running
# in a still-open tmux session may not have been flushed yet.
history -a

tar czf preliminary_milestones_full.tar.gz \
  liver-us-detection-baselines/ \
  nnUNet_raw/ \
  nnUNet_preprocessed/ \
  nnUNet_results/ \
  .bash_history
```

Then, on the Mac:

```sh
cd ~/Projects/liver-us-detection-baselines

scp root@<server-ip>:~/preliminary_milestones_full.tar.gz /tmp/
tar xzf /tmp/preliminary_milestones_full.tar.gz
```

## What happens next

Evaluate the predictions locally. The runner already produced the GPU inference benchmark. To record comparable CPU inference numbers, rerun the benchmark script later on the Mac with the same checkpoints, images, and settings, changing only `--device` and `--output-dir`:

```sh
python training/benchmark_gpu_inference.py \
  --nnunet-raw "$nnUNet_raw" \
  --dataset-name Dataset001_AUL \
  --dataset-id 1 \
  --seeds 42 43 44 \
  --trainer-prefix nnUNetTrainerMilestones_seed \
  --checkpoint checkpoint_final.pth \
  --device cpu \
  --output-dir logs/inference_cpu
```

## Environment variable notes

nnU-Net environment variables do not persist between SSH sessions unless written to `.bashrc` (step 6). If you reconnect and did not run that step, re-export before running anything:

```sh
export nnUNet_raw="$HOME/nnUNet_raw"
export nnUNet_preprocessed="$HOME/nnUNet_preprocessed"
export nnUNet_results="$HOME/nnUNet_results"
export nnUNet_extTrainer="$HOME/liver-us-detection-baselines/training/custom_trainers"
```

## Spot/preemptible instance notes for Verda.com

To train on a cheaper spot instance on Verda.com, as of September 2026, assumes the risk of the instance being taken. The runner passes `--c` to `nnUNetv2_train`, so training resumes from the last checkpoint if preempted. Re-export environment variables (or rely on `.bashrc`) and rerun the same command. Preprocessing is skipped if already done. Use the same GPU type across all three seeds for consistency.


## Data deletion

Final step - be sure to delete not just the instance but the module that the data is saved on.
