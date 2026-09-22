# Prediction-only: manually run preliminary milestones validation

This document lets you reproduce the held out validation-side predictions for the preliminary milestones experiment: predicting the 150, 300, and 750 epoch checkpoints on the validation split for each seeded run.

The validation images are the fold-0 split from `splits_final.json`. They were held out from gradient updates during training; nnU-Net used them only for pseudo-Dice logging and best-checkpoint selection. Predicting with these checkpoints on the held-out cases purports to provide additional epoch-convergence insight.

Because your environment may deviate in unpredictable ways, run these commands step-by-step to pinpoint and resolve issues if they come up.

## Fixed conditions

| Setting | Value |
|---|---|
| Dataset | AUL / `Dataset001_AUL` |
| Training images | 625 |
| Validation images | fold-0 split (see `splits_final.json`) |
| Fold | 0 |
| Initialization seeds | 42, 43, 44 |
| Trainer | `nnUNetTrainerMilestones_seed{42,43,44}` |
| Architecture | PlainConvUNet 2D |

## Prerequisites

- The `preliminary_milestones_test_vals.zip` archive uploaded from the Mac, as prepared below.
- The nnU-Net environment variables set (see "Configure nnU-Net directories" below).

## Prepare the upload archive on the Mac

From the directory containing the three nnU-Net directories, create a staging copy. Trim only the copied `nnUNet_results/` directory; keep the 150, 300, and 750 epoch checkpoints for seeds 42, 43, and 44, along with each trainer's `plans.json` and `dataset.json`.

```sh
rm -rf preliminary_milestones_test_vals
mkdir preliminary_milestones_test_vals
cp -R nnUNet_raw nnUNet_preprocessed nnUNet_results preliminary_milestones_test_vals/

find preliminary_milestones_test_vals/nnUNet_results -type f -name '*.pth' \
  ! -name checkpoint_epoch150.pth \
  ! -name checkpoint_epoch300.pth \
  ! -name checkpoint_epoch750.pth \
  -delete

(
  cd preliminary_milestones_test_vals
  zip -r ../preliminary_milestones_test_vals.zip \
    nnUNet_raw nnUNet_preprocessed nnUNet_results
)
```

The ZIP must contain `nnUNet_raw/`, `nnUNet_preprocessed/`, and `nnUNet_results/` at its root, not inside a `preliminary_milestones_test_vals/` directory. 

This should create a ZIP file named `preliminary_milestones_test_vals.zip` of ~3.4 GB.

## Server setup

All commands run on Verda.com's RTX 6000 Ada.

### 1. Configure prompt (optional)

```sh
# More visible prompt with a timestamp so no manual steps are missed
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

Confirm this checkout contains `training/run_preliminary_milestones_test_vals.sh` and the `training/custom_trainers/` directory before continuing.

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

The explicit nnU-Net version pin (nnunetv2==2.8.1) is the reproducibility anchor. The runner logs the installed nnU-Net version, the PyTorch version, and the repo's git SHA during its environment block; the remaining dependencies are pinned in requirements.txt.

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

### 7. Upload the archive

From the Mac, upload the ZIP to the server home directory:

```sh
scp preliminary_milestones_test_vals.zip root@<server-ip>:~/
```

### 8. Confirm the archive contents

```sh
unzip -l ~/preliminary_milestones_test_vals.zip
```

Confirm it lists `nnUNet_raw/`, `nnUNet_preprocessed/`, and `nnUNet_results/` at the archive root. The runner restores these directories before building the validation image set and running predictions.

## Validation predictions

### 9. Run the validation predictions

The script builds a temporary directory of validation images (symlinks into `tmp/val_images_fold0/`), then predicts the three milestone checkpoints (150, 300, and 750) for each seed.

```sh
# Start a persistent session so predictions survive an SSH disconnect
tmux new -s val

# Inside tmux, run the runner, teeing output to a file you can tail later
bash training/run_preliminary_milestones_test_vals.sh 2>&1 | tee preliminary_milestones_val.log
```

Detach from tmux without stopping the run with `Ctrl+b` then `d`. Reattach after a reconnect with `tmux attach -t val`.

#### Optionally monitor progress

From a separate SSH session:

```sh
tail -f ~/liver-us-detection-baselines/preliminary_milestones_val.log
```

Ctl+C to close `tail`.

The runner logs an environment block (host, GPU, CPU, RAM, PyTorch, nnU-Net version and SHA, repo SHA), samples GPU utilization/memory/power/temperature once per second per seed, records per-checkpoint prediction timing, checkpoint file sizes, model footprint, and runs a per-image GPU inference benchmark on the val split. All of this is written under `experiment_logs/milestones_pilot_vals/`; there are no manual logging steps to run separately.

## Checkpoints predicted per seed

| Checkpoint file | Prediction label |
|---|---|
| `checkpoint_epoch150.pth` | `epoch150` |
| `checkpoint_epoch300.pth` | `epoch300` |
| `checkpoint_epoch750.pth` | `epoch750` |

`checkpoint_best` and `checkpoint_best_mass` are excluded because they were selected using val-set pseudo-Dice and would be biased on this same surface. The remaining milestones (50, 100, 500) and `final` are left unpredicted.

Prediction directories use the naming `predictions_milestones_val_625images_seed{SEED}_{label}`, e.g. `predictions_milestones_val_625images_seed42_epoch300`, repeated for each seed (9 directories total).

## Verification

### 10. Verify completion

```sh
# Reattach to the tmux session, or tail the log file
tmux attach -t val
# or, if detached:
tail -20 preliminary_milestones_val.log
# Look for: "Validation predictions complete"

# 9 prediction directories (3 checkpoints × 3 seeds)
for SEED in 42 43 44; do
  echo "Seed $SEED:"
  ls -d "$nnUNet_results"/predictions_milestones_val_625images_seed${SEED}_* | wc -l  # expect 3
done
```

Each prediction directory's PNG count equals the number of fold-0 validation cases from `splits_final.json`, not the full test-set count.

## Download results

Download the evidence needed to reconstruct and audit the run.

```sh
cd ~
tar czf preliminary_milestones_val.tar.gz \
  liver-us-detection-baselines/experiment_logs/milestones_pilot_vals/ \
  nnUNet_results/predictions_milestones_val_625images_seed*_*/
```

Then, on your local computer:

```sh
scp root@<server-ip>:~/preliminary_milestones_val.tar.gz /tmp/
tar xzf /tmp/preliminary_milestones_val.tar.gz
```

## Archive provenance

The runner logs the archive path and the repository git SHA in its output, linking the validation predictions to the uploaded `preliminary_milestones_test_vals.zip` snapshot.
