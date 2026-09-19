# Compare GPU rental costs for AUL training

This experiment compares three GPU rental instances with the built-in nnU-Net training benchmark. It supports hardware selection for the study.

The runner uses `nnUNetTrainerBenchmark_5epochs` from nnU-Net 2.8.1. It includes data loading, augmentation, training, and validation batches. It skips model checkpoints and final predictions. It does not call `benchmark_gpu_inference.py` or a custom trainer.

An epoch is one fixed interval of training and validation. Each benchmark epoch contains 250 training batches and 50 validation batches. Each instance completes three separate runs of five epochs.

The primary measurement is the median of three fastest-epoch times, one from each run. This follows the built-in benchmark metric. It estimates favorable sustained performance rather than total rental time. The runner also saves all five epoch times and elapsed process time for each run.

## Files and fixed conditions

Place `run_gpu_training_benchmark.sh` in `training/`. Place this guide in `reproduce_study/`.

| Condition | Value |
|---|---|
| Dataset | AUL, Zenodo record 7272660 |
| Development images | 625 in `Dataset001_AUL` |
| Held-out images | 110, unused by this benchmark |
| Network | PlainConvUNet 2D |
| Plans | `nnUNetPlans`, default 8 GB target |
| Fold | 0 |
| Trainer | `nnUNetTrainerBenchmark_5epochs` |
| nnU-Net version | 2.8.1 |
| Repeats | 3 separate processes |
| Augmentation workers | 8 on every instance |
| Compilation | Enabled on every instance |
| GPUs per run | 1 |

The repeats measure timing variation. They are not the study initialization seeds 42, 43, and 44. This benchmark does not promise identical random batches or bitwise results. Its short training schedule is not an accuracy experiment.

## 1. Record the rental choices

Choose the three eligible GPU instance types before measuring performance. Record their advertised prices and availability on the selection date. Use the same currency and rental category, either spot or on-demand.

Record the provider, region, instance type, image identifier, CPU allocation, RAM, storage type, and price basis. Save the price evidence with the study records. Include mandatory instance charges in the hourly price. Record separate charges, taxes, discounts, and spot interruptions independently.

Use one GPU per instance and local SSD storage for the prepared data. Select at least eight available CPU threads on each instance. Use comparable CPU and storage allocations where available. Differences in these resources make this a comparison of rental instances, rather than isolated GPU hardware.

Choose an Ubuntu image with CUDA-enabled PyTorch that supports all three GPUs. Reuse its exact image identifier and Python version across instances. Follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) if the image lacks a compatible installation.

## 2. Install shell dependencies

Connect to the first instance through SSH. Run these commands as root, or prefix the package commands with `sudo`:

```bash
apt update
apt install -y git python3-venv unzip tmux
```

## 3. Fetch the study revision

Commit both new files to the study repository before this step. Replace `STUDY_COMMIT` with that full commit identifier. If the repository URL changes, replace the URL too.

```bash
cd "$HOME"
git clone https://github.com/roprice/liver-us-detection-baselines.git
cd liver-us-detection-baselines
git checkout STUDY_COMMIT
```

Use the same commit on every instance. Make sure that this checkout contains both new files and `training/convert_aul.py`.

## 4. Install Python dependencies

Create a separate environment that can access the image's installed PyTorch:

```bash
python3 -m venv --system-site-packages .venv-gpu-benchmark
source .venv-gpu-benchmark/bin/activate
python -m pip install -r requirements.txt "nnunetv2==2.8.1" zenodo-get
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
python -m pip freeze > benchmark-environment.txt
```

The supplied `requirements.txt` uses minimum versions. `benchmark-environment.txt` records the resolved versions, including inherited packages. Preserve the exact cloud image too, because a package list does not capture drivers or system libraries.

On subsequent instances, copy `benchmark-environment.txt` from the first instance. Create and activate the same environment.
Install from that file:

```bash
python -m pip install -r benchmark-environment.txt
python -m pip check
```

If a recorded package uses an unavailable local path, preserve its wheel or reuse an image snapshot before continuing. Do not silently substitute another version.

## 5. Set the benchmark directories

Run these commands from the repository root in the active Python environment:

```bash
export nnUNet_raw="$HOME/aul-gpu-benchmark/raw"
export nnUNet_preprocessed="$HOME/aul-gpu-benchmark/preprocessed"
export nnUNet_results="$HOME/aul-gpu-benchmark/results"
export CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=8
export nnUNet_compile=true
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

The runner creates separate result directories under `logs/gpu_training_benchmark/`. It overrides `nnUNet_results` for each repeat. It leaves the milestone experiment results alone.

These variables apply to the current shell. After reconnecting, activate the environment and repeat this step. Start the benchmark without another training process on the same GPU.

## 6. Download the source data

Complete steps 6 through 9 on the first instance only. On subsequent instances, restore the prepared archive described below and continue at step 10.

```bash
mkdir -p data/source
cd data/source
zenodo_get 7272660
cat > aul-checksums.md5 <<'EOF'
c37fef0cb2730236a79ef57e5315995e  Benign.zip
63894a9e5654a69c3b94bda84071dfb0  Malignant.zip
a7e16299b2cf12ca4a6c3468d2e4978f  Normal.zip
EOF
md5sum -c aul-checksums.md5
```

Make sure that all three files report `OK` before extraction:

```bash
unzip 'Benign.zip' -d AUL
unzip 'Malignant.zip' -d AUL
unzip 'Normal.zip' -d AUL
cd ../..
```

The source is [AUL versioned record 7272660](https://doi.org/10.5281/zenodo.7272660). The expected archive hashes come from the supplied study conversion script.

## 7. Convert AUL

Run the existing conversion script:

```bash
python training/convert_aul.py \
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL"
```

The script converts images to grayscale PNG and renders polygon annotations as masks. It uses seed 42 for its category-stratified development/test split. It does not establish patient-level independence. This hardware benchmark uses only the development partition.

## 8. Make sure that counts match

```bash
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['nnUNet_raw']) / 'Dataset001_AUL'
assert len(list((root / 'imagesTr').glob('*_0000.png'))) == 625
assert len(list((root / 'labelsTr').glob('*.png'))) == 625
assert len(list((root / 'imagesTs').glob('*_0000.png'))) == 110
print('Dataset counts match.')
PY
```

## 9. Prepare the shared benchmark input

Run preprocessing once:

```bash
bash training/run_gpu_training_benchmark.sh --prepare-only
```

The runner prepares only the 2D configuration. It retains an existing `splits_final.json`, or creates the standard five-fold split with seed 12345. A SHA-256 manifest is a list of file hashes. The runner saves this manifest and the conversion case mapping with the prepared dataset.

Archive the prepared data and resolved package versions:

```bash
cp benchmark-environment.txt "$HOME/aul-gpu-benchmark/"
tar -czf "$HOME/aul-gpu-benchmark-input.tar.gz" \
  -C "$HOME/aul-gpu-benchmark" preprocessed benchmark-environment.txt
sha256sum "$HOME/aul-gpu-benchmark-input.tar.gz"
```

Save this archive and its hash before deleting the first instance. Copy the same archive to each subsequent instance. From your local machine, replace the SSH addresses and transfer it:

```bash
scp root@FIRST_SERVER:~/aul-gpu-benchmark-input.tar.gz .
scp aul-gpu-benchmark-input.tar.gz root@NEXT_SERVER:~/
```

On each subsequent instance, compare the archive hash with the first instance's recorded hash. Restore it into the directories from step 5:

```bash
sha256sum "$HOME/aul-gpu-benchmark-input.tar.gz"
tar -xzf "$HOME/aul-gpu-benchmark-input.tar.gz" -C "$HOME/aul-gpu-benchmark"
cp "$HOME/aul-gpu-benchmark/benchmark-environment.txt" .
```

Use the restored package file for step 4 on subsequent instances. Do not repeat conversion, planning, or preprocessing there. The training runner requires the saved manifest and refuses changed or missing files.

## 10. Run the GPU comparison

Start a persistent shell from the repository root:

```bash
tmux new -s gpu-benchmark
```

Inside tmux, activate the environment and repeat step 5. Enter the actual rental details at the prompts:

```bash
source .venv-gpu-benchmark/bin/activate
read -r -p 'Provider: ' PROVIDER
read -r -p 'Instance type: ' INSTANCE_LABEL
read -r -p 'Region: ' REGION
read -r -p 'Exact cloud image identifier: ' IMAGE_REFERENCE
read -r -p 'Hourly instance price, number only: ' HOURLY_PRICE
read -r -p 'Currency, for example USD: ' CURRENCY
read -r -p 'Rental category, spot or on-demand: ' RATE_TYPE
export PROVIDER INSTANCE_LABEL REGION IMAGE_REFERENCE HOURLY_PRICE CURRENCY RATE_TYPE
bash training/run_gpu_training_benchmark.sh
```

The runner completes three repeats automatically. Detach with `Ctrl+b`, then `d`. Reattach with `tmux attach -t gpu-benchmark`.

If the instance interrupts a run, retain its incomplete directory. Start a new invocation after recovery. The benchmark does not resume because its trainer saves no checkpoints. Record interruptions and their charges separately.

## Read and preserve the results

Each invocation creates a unique directory under `logs/gpu_training_benchmark/`. The runner prints its path at startup and completion.

| File | Contents |
|---|---|
| `status.txt` | `COMPLETE` only after every repeat succeeds |
| `summary.json` | Median, range, and estimated cost per epoch |
| `repeats.csv` | Timing and estimated cost for each repeat |
| `environment.json` | Rental details, hardware, software, plan dimensions, and Git revision |
| `requirements-frozen.txt` | Installed package versions |
| `nnUNetPlans.json` | Fixed network and batch configuration |
| `splits_final.json` | Development folds |
| `benchmark_manifest.json` | Prepared input hashes |
| `repeat_N/console.log` | Complete output for that repeat |
| `repeat_N/epoch_times.json` | Five epoch times, rounded by nnU-Net |
| `repeat_N/results/.../benchmark_result.json` | Original nnU-Net result, including precise fastest time |

The runner treats missing results, invalid timings, and fewer than five completed epochs as failures. Some upstream training errors return a successful process status, so the runner also inspects the result contents.

Compare only completed invocations with matching input hashes, package versions, worker counts, and compilation configuration. Repeated processes can reuse compiler and filesystem caches. The process timings therefore do not represent identical cold starts.

Calculate the selection metric as follows:

```text
Estimated cost per epoch =
median fastest epoch seconds Ã— hourly instance price Ã· 3600
```

Report all three repeat measurements and their range. Prefer the lowest estimated cost among the tested eligible instances. If the differences overlap normal timing variation, collect more repeats before declaring a winner.

The cost estimate excludes setup, preprocessing, idle rental time, interruptions, and separate charges. The elapsed process estimate includes startup and teardown for that process. Neither estimate is an invoice total. Report actual billed costs separately when available.

Describe the winner as the lowest-cost tested instance for this workload and price date. The benchmark does not measure model footprint, inference latency, energy consumption, or final accuracy. Keep those measurements in the existing study experiments.

Archive the evidence from the repository root:

```bash
tar -czf "$HOME/gpu-training-benchmark-evidence.tar.gz" \
  logs/gpu_training_benchmark \
  training/run_gpu_training_benchmark.sh \
  training/convert_aul.py \
  reproduce_study/run_gpu_training_benchmark.md \
  requirements.txt benchmark-environment.txt
```

Download the archive from your local machine before terminating the instance:

```bash
scp root@SERVER:~/gpu-training-benchmark-evidence.tar.gz ./gpu-training-benchmark-INSTANCE.tar.gz
```

Replace `SERVER` and `INSTANCE` for each rental. Retain the prepared input archive once for the whole comparison. Retain the selected Git commit and cloud image identifier with the evidence.

## Upstream reference

The [nnU-Net benchmark documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/benchmarking.md) describes the built-in trainer. This runner targets the implementation distributed in `nnunetv2==2.8.1`.
