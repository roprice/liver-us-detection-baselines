# Prediction-only: manually run preliminary milestones validation

This document lets you reproduce the validation-side predictions for the preliminary milestones experiment. Unlike the setup-and-training runbook, this script only predicts — it assumes training has already completed.

The validation images are the fold-0 split from `splits_final.json`. They were held out from gradient updates during training; nnU-Net used them only for pseudo-Dice logging and best-checkpoint selection. Predicting the milestone checkpoints on this independent surface confirms the epoch-convergence plateau observed on the test set.

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

Each seed predicts seven milestone checkpoints (epochs 50, 100, 150, 300, 500, 750, and final) on the validation split. `checkpoint_best` and `checkpoint_best_mass` are excluded because they were selected on validation pseudo-Dice and would look artificially optimistic on this same image set.

## Prerequisites

This runbook reuses a server where the setup-and-training runbook has already completed. The following must be in place before you start:

- Completed training from `training/run_preliminary_milestones_test.sh`.
- `splits_final.json` in `$nnUNet_preprocessed/Dataset001_AUL/`.
- All milestone `.pth` files present in `$nnUNet_results/Dataset001_AUL/`.
- The nnU-Net environment variables set (see "Configure nnU-Net directories" below).

## Server setup

All commands run on a fresh Verda GPU instance (NVIDIA RTX PRO 6000, 96 GiB VRAM) running Ubuntu, or on any instance where the prerequisites above already hold.

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

### 7. Confirm prerequisites are present

```sh
# Training must already have produced the fold-0 split
ls "$nnUNet_preprocessed/Dataset001_AUL/splits_final.json"

# And the milestone checkpoints for each seed
for SEED in 42 43 44; do
  ls "$nnUNet_results/Dataset001_AUL/nnUNetTrainerMilestones_seed${SEED}__nnUNetPlans__2d/fold_0/"*.pth
done
```

## Validation predictions

### 8. Run the validation predictions

The script builds a temporary directory of validation images (symlinks into `tmp/val_images_fold0/`), then predicts the seven milestone checkpoints for each seed.

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
| `checkpoint_epoch50.pth` | `epoch50` |
| `checkpoint_epoch100.pth` | `epoch100` |
| `checkpoint_epoch150.pth` | `epoch150` |
| `checkpoint_epoch300.pth` | `epoch300` |
| `checkpoint_epoch500.pth` | `epoch500` |
| `checkpoint_epoch750.pth` | `epoch750` |
| `checkpoint_final.pth` | `final` |

`checkpoint_best` and `checkpoint_best_mass` are excluded because they were selected using val-set pseudo-Dice and would be biased on this same surface.

Prediction directories use the naming `predictions_milestones_val_625images_seed{SEED}_{label}`, e.g. `predictions_milestones_val_625images_seed42_epoch500`, repeated for each seed (21 directories total).

## Verification

### 9. Verify completion

```sh
# Reattach to the tmux session, or tail the log file
tmux attach -t val
# or, if detached:
tail -20 preliminary_milestones_val.log
# Look for: "Validation predictions complete"

# 21 prediction directories (7 checkpoints × 3 seeds)
for SEED in 42 43 44; do
  echo "Seed $SEED:"
  ls -d "$nnUNet_results"/predictions_milestones_val_625images_seed${SEED}_* | wc -l  # expect 7
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
