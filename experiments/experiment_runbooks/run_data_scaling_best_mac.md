# Predict all 24 data-scaling best checkpoints on an M1 Mac

Runner: `experiments/experiment_runners/run_data_scaling_best_mac.sh`

This is prediction only: no training, planning, evaluation, or GPU benchmarking.
It uses fold 0 of `checkpoint_best.pth` for seeds 42, 43, and 44:

| Dataset | Training images | Checkpoints |
| --- | ---: | ---: |
| Dataset002_AUL_005 | 5 | 3 |
| Dataset003_AUL_010 | 10 | 3 |
| Dataset004_AUL_020 | 20 | 3 |
| Dataset005_AUL_040 | 40 | 3 |
| Dataset006_AUL_080 | 80 | 3 |
| Dataset007_AUL_160 | 160 | 3 |
| Dataset008_AUL_320 | 320 | 3 |
| Dataset001_AUL | 625 | 3 |

The supplied `Dataset00*_AUL/` glob matches only Dataset001. The inventory
pattern including the subset suffixes is:

```text
nnUNet_results/Dataset00*_AUL*/nnUNetTrainerDataScaling_seed*__nnUNetPlans__2d/fold_0/checkpoint_best.pth
```

All 24 checkpoints and their `plans.json` and `dataset.json` files were present
when this runner was prepared. Every model predicts the same 110 held-out images
from `nnUNet_raw/Dataset001_AUL/imagesTs`, as in the data-scaling experiment.
Training images and preprocessed arrays are not needed for inference.

## 1. Python environment

Run these commands from the repository root. Use an existing native Apple Silicon
Python environment with the project requirements installed, or create one using
a native arm64 Python 3.11 or 3.12 installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -c 'import platform, torch; print(platform.machine(), torch.__version__); print("MPS available:", torch.backends.mps.is_available())'
```

Expect `arm64` and `MPS available: True`. Do not use macOS's system Python for
installation. The runner requires the project's pinned `nnunetv2==2.8.1` and
uses the selected Python's nnU-Net entry point, avoiding a mismatched CLI on PATH.
Set `PYTHON=/absolute/path/to/python3` if needed.

The runner sets `nnUNet_extTrainer` to `experiments/custom_trainers` automatically;
there is no need to copy trainer files into site-packages.

## 2. Check the inventory and commands

```bash
bash experiments/experiment_runners/run_data_scaling_best_mac.sh --dry-run
```

This checks the exact 24 expected size/seed checkpoints, model JSON metadata,
and the 110 input filenames, then prints all 24 prediction commands. It writes
nothing and needs only Python's standard library; it does not test PyTorch,
MPS, checkpoint loading, or image decoding.

By default, the runner resolves all paths relative to its repository, even when
launched from another directory. Existing `nnUNet_raw`, `nnUNet_preprocessed`,
and `nnUNet_results` environment variables override those defaults. Unset stale
values before running if they point to another machine or study:

```bash
unset nnUNet_raw nnUNet_preprocessed nnUNet_results
```

## 3. Run predictions

```bash
caffeinate -i bash experiments/experiment_runners/run_data_scaling_best_mac.sh
```

Keep the Mac plugged in and the lid open. `caffeinate` prevents idle sleep while
the runner is active. Run only one copy of the runner at a time.

Settings for the 32 GB M1:

- MPS device; one model per process, with all 24 jobs run sequentially.
- One preprocessing worker and one segmentation-export worker (`-npp 1 -nps 1`).
- `torch.compile` disabled; CPU fallback enabled for supported MPS fallback operations.
- nnU-Net's standard mirroring/test-time augmentation retained, with sliding-window
  step size 0.5. No probability arrays are saved.
- On MPS, nnU-Net keeps sliding-window accumulation on CPU. Its message about
  `perform_everything_on_device` being disabled is expected, not a switch of the
  network itself to CPU.

These settings reduce simultaneous memory use; runtime and peak memory have not
been benchmarked on this machine. CPU/MPS results need not be bit-identical to
previous CUDA predictions.

## 4. Outputs and completion

Each model writes 110 PNG segmentation masks to:

```text
nnUNet_results/predictions_data_scaling_<SIZE>images_seed<SEED>_best/
```

For example, `nnUNet_results/predictions_data_scaling_5images_seed42_best/`.
There should be 24 output directories and 2,640 masks total. Labels remain
background=0, liver=1, mass=2. nnU-Net also writes prediction arguments and model
metadata. Existing `_final` prediction directories are not touched.

Per-model logs are written to:

```text
experiment_logs/data_scaling_best_mac/size<SIZE>_seed<SEED>_<DEVICE>.log
```

The runner stops on the first failure. After each successful prediction it checks
all expected mask names, PNG integrity, and image dimensions before writing
`.prediction_complete` inside that output directory. The marker records the device.
The terminal reports each completed job's elapsed seconds.

## 5. Resume or use CPU

Rerun the same command after an interruption. Completed model directories with a
marker are verified and skipped. A model without a marker is predicted again in
full, overwriting that model's existing masks; this avoids trusting partially
written outputs. Its per-model log is also overwritten on retry.

If MPS is unavailable, runs out of memory, or encounters an unsupported operation
that CPU fallback cannot handle:

```bash
DEVICE=cpu caffeinate -i bash experiments/experiment_runners/run_data_scaling_best_mac.sh
```

Completed MPS jobs remain skipped; unfinished jobs run on CPU. No checkpoint or
input files are modified. For a uniform all-CPU rerun, remove the completion
markers from the `_best` output directories first.

Markers assume checkpoints, test inputs, and inference settings are unchanged.
If any change, remove the affected output directory's `.prediction_complete`
marker before rerunning. Likewise, if verification rejects a completed folder,
remove its marker to regenerate its masks. Unexpected extra PNGs must be moved
out of that folder before verification can pass.
