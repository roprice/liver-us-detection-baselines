# Compare GPU rental costs for AUL training

This experiment compares GPU rental instances with the built-in nnU-Net training benchmark. It supports hardware selection for the study.

The benchmark uses `nnUNetTrainerBenchmark_5epochs` from nnU-Net 2.8.1. It includes data loading, augmentation, training, and validation batches. It skips model checkpoints and final predictions. It does not call `benchmark_gpu_inference.py` or a custom trainer.

An epoch is one fixed interval of training and validation. Each benchmark epoch contains 250 training batches and 50 validation batches. Each GPU instance runs one five-epoch benchmark.

The primary measurement is `fastest_epoch`, which nnU-Net writes to `benchmark_result.json`. The benchmark also records the GPU name, PyTorch version, and cuDNN version.

## Files and fixed conditions

This guide uses `experiments/convert_aul.py` from the study repository.

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
| GPUs per run | 1 |
| Compilation | Disabled on every instance |

We selected Verda.com based on its EU ownership and location, pricing flexibility for short runs, and ease of use. These instructions may be useful for other cloud GPU providers too.

## First instance

Use the first instance to download AUL, convert it, create the shared 2D preprocessing output, and run the first benchmark. Download the prepared input archive before you delete this instance.

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

# Apply to current shell. New shells read ~/.bashrc.
source ~/.bashrc
```

Run this before any other command so command history is captured from the start of the session.

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

Make sure that this checkout contains `experiments/convert_aul.py` before continuing.

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

The explicit `nnunetv2==2.8.1` pin is the reproducibility anchor. The benchmark result records the installed PyTorch and cuDNN versions.

### 6. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/aul-gpu-benchmark/raw"
export nnUNet_preprocessed="$HOME/aul-gpu-benchmark/preprocessed"
export nnUNet_results="$HOME/aul-gpu-benchmark/results"
export nnUNet_compile=false

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

`nnUNet_compile=false` avoids PyTorch compilation. Use this setting for every benchmark instance.

If you lose your connection, re-run this step as these won't persist.


### 7. Download AUL from Zenodo

Dataset: Annotated Ultrasound Liver images

DOI: [10.5281/zenodo.7272660](https://doi.org/10.5281/zenodo.7272660)

```sh
mkdir -p data/source
cd data/source
pip install zenodo-get --break-system-packages
zenodo_get 7272660
md5sum *.zip
unzip '*.zip' -d AUL
rm -rf AUL/__MACOSX
cd ../..
```

Make sure that the archive hashes match these values:

| File | MD5 |
|---|---|
| `Benign.zip` | `c37fef0cb2730236a79ef57e5315995e` |
| `Malignant.zip` | `63894a9e5654a69c3b94bda84071dfb0` |
| `Normal.zip` | `a7e16299b2cf12ca4a6c3468d2e4978f` |

### 8. Convert AUL to nnU-Net format

```sh
python experiments/convert_aul.py \\
  --raw-data-dir data/source/AUL \
  --output-dir "$nnUNet_raw/Dataset001_AUL"
```

Images pass through without modification. The script renders segmentation labels from the annotated polygons.

### 9. Make sure that the input counts match

```sh
ls "$nnUNet_raw/Dataset001_AUL/imagesTr" | wc -l  # expect 625
ls "$nnUNet_raw/Dataset001_AUL/imagesTs" | wc -l  # expect 110
```

### 10. Prepare the shared benchmark input

```sh
nnUNetv2_plan_and_preprocess -d 1 -c 2d --verify_dataset_integrity
```

This command creates the fixed 2D preprocessing output and `nnUNetPlans.json`. Run it only on this first instance.

Archive the prepared data:

```sh
tar -czf "$HOME/aul-gpu-benchmark-input.tar.gz" \
  -C "$HOME/aul-gpu-benchmark" preprocessed
sha256sum "$HOME/aul-gpu-benchmark-input.tar.gz"
```

Save the archive and its hash. You copy this archive from your Mac to each later GPU instance.

### 11. Run the benchmark

```sh
nnUNetv2_train 1 2d 0 -tr nnUNetTrainerBenchmark_5epochs
```

The command takes about three minutes. The first epoch includes startup work. nnU-Net reports `fastest_epoch` after all five epochs finish.

### 12. Print copy-ready Markdown results

Set the actual hourly price for this instance. Then run the command below.

```sh
export HOURLY_PRICE=1.10
export CURRENCY=USD
export BENCHMARK_JSON="$nnUNet_results/Dataset001_AUL/nnUNetTrainerBenchmark_5epochs__nnUNetPlans__2d/fold_0/benchmark_result.json"

python - "$BENCHMARK_JSON" <<'PY'
import json
import os
import subprocess
import sys

record = next(iter(json.load(open(sys.argv[1])).values()))
seconds = float(record['fastest_epoch'])
price = float(os.environ['HOURLY_PRICE'])
vram_mib = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
    text=True,
).strip().splitlines()[0]
vram_gib = int(vram_mib) // 1024
cost = seconds * price / 3600
hours = seconds * 1000 / 3600

print(f'''## {record['gpu_name']}

GPU: {record['gpu_name']}
VRAM: {vram_gib} GiB
Fastest epoch: {seconds:.4f} seconds
PyTorch: {record['torch_version']}
Hourly price: ${price:.3f}
Currency: {os.environ['CURRENCY']}
Cost per benchmark epoch: {seconds:.4f} × {price:.3f} ÷ 3600 = ${cost:.5f}
Estimated 1,000-epoch training time: {hours:.2f} hours
Estimated cost for 1,000 epochs: ${cost * 1000:.2f}''')
PY
```

Copy the printed Markdown into your benchmark record.

## Subsequent instance(s)

Use this section for every later GPU instance. Do not download AUL, convert AUL, or run preprocessing again.

### 1. Configure prompt (optional)

```sh

# More visible prompt with a timestamp so no manual steps are missed
cat >> ~/.bashrc << 'PROMPTEOF'
PS1='\[\e[38;5;208m\]\u@\h:\w \t \[\e[0m\]\$ '
PROMPTEOF
source ~/.bashrc
```

Optionally run this to help you keep track of what step you're on

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

### 6. Configure nnU-Net directories

```sh
export nnUNet_raw="$HOME/aul-gpu-benchmark/raw"
export nnUNet_preprocessed="$HOME/aul-gpu-benchmark/preprocessed"
export nnUNet_results="$HOME/aul-gpu-benchmark/results"
export nnUNet_compile=false

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

### 7. Upload the prepared input archive

Run this command on your Mac. Replace `NEXT_SERVER` with the IP address of this instance.

```sh
scp -i ~/.ssh/id_ed25519 ~/aul-gpu-benchmark-input.tar.gz root@NEXT_SERVER:~/
```

### 8. Restore the prepared input archive

Run these commands on the GPU instance:

```sh
sha256sum "$HOME/aul-gpu-benchmark-input.tar.gz"
tar -xzf "$HOME/aul-gpu-benchmark-input.tar.gz" -C "$HOME/aul-gpu-benchmark"
```

Make sure that the archive hash matches the hash from the first instance. This archive supplies the same prepared AUL data and 2D nnU-Net plan.

### 9. Run the benchmark

```sh
nnUNetv2_train 1 2d 0 -tr nnUNetTrainerBenchmark_5epochs
```

### 10. Print copy-ready Markdown results

Set the actual hourly price for this instance. Then run the command below.

```sh
export HOURLY_PRICE=1.453
export CURRENCY=USD
export BENCHMARK_JSON="$nnUNet_results/Dataset001_AUL/nnUNetTrainerBenchmark_5epochs__nnUNetPlans__2d/fold_0/benchmark_result.json"

python - "$BENCHMARK_JSON" <<'PY'
import json
import os
import subprocess
import sys

record = next(iter(json.load(open(sys.argv[1])).values()))
seconds = float(record['fastest_epoch'])
price = float(os.environ['HOURLY_PRICE'])
vram_mib = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
    text=True,
).strip().splitlines()[0]
vram_gib = int(vram_mib) // 1024
cost = seconds * price / 3600
hours = seconds * 1000 / 3600

print(f'''## {record['gpu_name']}

GPU: {record['gpu_name']}
VRAM: {vram_gib} GiB
Fastest epoch: {seconds:.4f} seconds
PyTorch: {record['torch_version']}
Hourly price: ${price:.3f}
Currency: {os.environ['CURRENCY']}
Cost per benchmark epoch: {seconds:.4f} × {price:.3f} ÷ 3600 = ${cost:.5f}
Estimated 1,000-epoch training time: {hours:.2f} hours
Estimated cost for 1,000 epochs: ${cost * 1000:.2f}''')
PY
```

Copy the printed Markdown into your benchmark record. Repeat the subsequent-instance section for each remaining GPU.

## Upstream reference

The [nnU-Net benchmark documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/benchmarking.md) describes the built-in trainer. This guide targets `nnunetv2==2.8.1`.
