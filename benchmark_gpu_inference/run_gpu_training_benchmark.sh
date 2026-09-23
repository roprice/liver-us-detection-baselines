#!/usr/bin/env bash
# Run from the repository root. No custom trainer is required.
set -euo pipefail
export GPU_BENCHMARK_RUNNER="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
python - "$@" <<'PY'
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


def fail(message):
    raise SystemExit(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command, log, env=None):
    print('+ ' + ' '.join(command), flush=True)
    start = time.monotonic()
    with log.open('w') as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
    if code:
        fail(f'Command failed with exit status {code}. Read {log}.')
    return time.monotonic() - start


parser = argparse.ArgumentParser(description='Compare rental costs with the built-in nnU-Net training benchmark.')
parser.add_argument('--prepare-only', action='store_true', help='Prepare Dataset001_AUL once before GPU comparisons.')
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    fail('Use at least one repeat.')
if metadata.version('nnunetv2') != '2.8.1':
    fail('Install nnunetv2==2.8.1 before this experiment.')
for name in ('nnUNet_raw', 'nnUNet_preprocessed'):
    if not os.environ.get(name):
        fail(f'Set {name} before this experiment.')
raw = Path(os.environ['nnUNet_raw']).resolve() / 'Dataset001_AUL'
prepared = Path(os.environ['nnUNet_preprocessed']).resolve() / 'Dataset001_AUL'
runner = Path(os.environ['GPU_BENCHMARK_RUNNER']).resolve()
repo = runner.parent.parent
logs = repo / 'logs' / 'gpu_training_benchmark'
logs.mkdir(parents=True, exist_ok=True)
manifest_path = prepared / 'benchmark_manifest.json'

if args.prepare_only:
    if manifest_path.exists():
        fail('Prepared benchmark data already exist. Reuse this dataset or select a new preprocessing directory.')
    if len(list((raw / 'imagesTr').glob('*_0000.png'))) != 625:
        fail('Expected 625 training images. Run training/convert_aul.py first.')
    if len(list((raw / 'labelsTr').glob('*.png'))) != 625:
        fail('Expected 625 training labels. Inspect the converted dataset.')
    prep_log = logs / ('prepare_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.log')
    run(['nnUNetv2_plan_and_preprocess', '-d', '1', '-c', '2d', '-pl', 'ExperimentPlanner',
         '-gpu_memory_target', '8', '-npfp', '4', '-np', '4', '--verify_dataset_integrity'], prep_log)
    from nnunetv2.utilities.crossval_split import generate_crossval_split
    split_path = prepared / 'splits_final.json'
    if not split_path.exists():
        identifiers = sorted(p.stem for p in (raw / 'labelsTr').glob('*.png'))
        save(split_path, generate_crossval_split(identifiers, seed=12345, n_splits=5))
    if (raw / 'case_mapping.json').is_file():
        shutil.copy2(raw / 'case_mapping.json', prepared / 'case_mapping.json')
    manifest = {str(p.relative_to(prepared)): sha(p)
                for p in sorted(prepared.rglob('*')) if p.is_file()}
    save(manifest_path, manifest)
    print(f'Preparation complete. Reuse the entire directory: {prepared}')
    sys.exit(0)

required = ('PROVIDER', 'INSTANCE_LABEL', 'REGION', 'IMAGE_REFERENCE', 'HOURLY_PRICE', 'CURRENCY', 'RATE_TYPE')
for name in required:
    if not os.environ.get(name, '').strip():
        fail(f'Set {name} before this experiment.')
try:
    price = float(os.environ['HOURLY_PRICE'])
except ValueError:
    fail('HOURLY_PRICE must be a positive number without a currency symbol.')
if not math.isfinite(price) or price <= 0:
    fail('HOURLY_PRICE must be a finite positive number.')
if os.environ['RATE_TYPE'] not in ('spot', 'on-demand'):
    fail('Set RATE_TYPE to spot or on-demand.')
if not manifest_path.is_file():
    fail('Run this script with --prepare-only, or copy the prepared benchmark dataset from the first instance.')
manifest = json.loads(manifest_path.read_text())
for relative, expected in manifest.items():
    path = prepared / relative
    if not path.is_file() or sha(path) != expected:
        fail(f'Prepared data differ from the saved manifest: {relative}')
plans = json.loads((prepared / 'nnUNetPlans.json').read_text())
configuration = plans['configurations']['2d']
if not configuration['architecture']['network_class_name'].endswith('.PlainConvUNet'):
    fail('This experiment requires PlainConvUNet.')
if json.loads((prepared / 'dataset.json').read_text())['numTraining'] != 625:
    fail('This experiment requires 625 development images.')
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    fail('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES before this experiment.')

# Fix these values across instances. They can be changed together before the experiment.
env = os.environ.copy()
env.pop('nnUNet_extTrainer', None)
env.setdefault('nnUNet_n_proc_DA', '8')
env.setdefault('nnUNet_compile', 'true')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS'):
    env[key] = '1'
try:
    workers = int(env['nnUNet_n_proc_DA'])
except ValueError:
    fail('nnUNet_n_proc_DA must be a positive integer.')
if workers < 1 or env['nnUNet_compile'].lower() not in ('true', 'false'):
    fail('Use positive augmentation workers and nnUNet_compile=true or false.')

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = Path(tempfile.mkdtemp(prefix=stamp + '_', dir=logs))
print(f'Results directory: {out}', flush=True)
(out / 'status.txt').write_text('INCOMPLETE\n')
shutil.copy2(runner, out / runner.name)
for source in (repo / 'training' / 'convert_aul.py', repo / 'requirements.txt',
               repo / 'reproduce_study' / 'run_gpu_training_benchmark.md'):
    if source.is_file():
        shutil.copy2(source, out / source.name)
for filename in ('nnUNetPlans.json', 'dataset.json', 'dataset_fingerprint.json', 'splits_final.json', 'benchmark_manifest.json'):
    shutil.copy2(prepared / filename, out / filename)

def capture(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

context = {
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'rental': {k: os.environ[k] for k in required},
    'python': sys.version, 'platform': platform.platform(),
    'nnunetv2': metadata.version('nnunetv2'), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
    'gpu_name': torch.cuda.get_device_name(0), 'cpu_count': os.cpu_count(),
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'environment': {k: env.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'nnUNet_compile', 'nnUNet_n_proc_DA',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS')},
    'plan_sha256': sha(prepared / 'nnUNetPlans.json'), 'manifest_sha256': sha(manifest_path),
    'batch_size': configuration['batch_size'], 'patch_size': configuration['patch_size'],
    'repeats': args.repeats, 'fold': 0,
}
for label, command in {
    'gpu': ['nvidia-smi', '-q'], 'cpu': ['lscpu'], 'memory': ['free', '-b'],
    'storage': ['df', '-T', str(prepared)],
    'git_commit': ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
    'git_status': ['git', '-C', str(repo), 'status', '--porcelain'],
}.items():
    context[label] = capture(command)
save(out / 'environment.json', context)
freeze = capture([sys.executable, '-m', 'pip', 'freeze'])
if freeze['exit_code']:
    fail('Package recording failed. Inspect the Python environment.')
(out / 'requirements-frozen.txt').write_text(freeze['stdout'])
trainer = 'nnUNetTrainerBenchmark_5epochs'
rows = []
for repeat in range(1, args.repeats + 1):
    repeat_dir = out / f'repeat_{repeat}'
    repeat_dir.mkdir()
    env['nnUNet_results'] = str(repeat_dir / 'results')
    seconds = run(['nnUNetv2_train', '1', '2d', '0', '-p', 'nnUNetPlans', '-tr', trainer],
                  repeat_dir / 'console.log', env)
    result_file = Path(env['nnUNet_results']) / 'Dataset001_AUL' / f'{trainer}__nnUNetPlans__2d' / 'fold_0' / 'benchmark_result.json'
    if not result_file.is_file():
        fail(f'Benchmark result is missing. Read {repeat_dir / "console.log"}.')
    result = json.loads(result_file.read_text())
    if len(result) != 1:
        fail(f'Expected one benchmark record in {result_file}.')
    record = next(iter(result.values()))
    fastest = record.get('fastest_epoch')
    if isinstance(fastest, bool) or not isinstance(fastest, (int, float)) or not math.isfinite(fastest) or fastest <= 0:
        fail(f'Benchmark failed: {fastest}. Read {repeat_dir / "console.log"}.')
    epoch_times = [float(x) for x in re.findall(r'Epoch time:\s*([0-9.]+)\s*s', (repeat_dir / 'console.log').read_text())]
    if len(epoch_times) != 5 or record.get('num_gpus') != 1:
        fail('Expected five completed epochs on one GPU. Read the repeat console log.')
    save(repeat_dir / 'epoch_times.json', epoch_times)
    rows.append({'repeat': repeat, 'gpu_name': record['gpu_name'], 'fastest_epoch_seconds': fastest,
                 'process_seconds': seconds, 'estimated_cost_per_epoch': fastest * price / 3600,
                 'estimated_process_cost': seconds * price / 3600, 'currency': os.environ['CURRENCY']})
    with (out / 'repeats.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
median = statistics.median(row['fastest_epoch_seconds'] for row in rows)
save(out / 'summary.json', {
    'metric': 'median of the fastest epoch from each independent process',
    'repeats': len(rows), 'median_fastest_epoch_seconds': median,
    'min_fastest_epoch_seconds': min(row['fastest_epoch_seconds'] for row in rows),
    'max_fastest_epoch_seconds': max(row['fastest_epoch_seconds'] for row in rows),
    'estimated_cost_per_epoch': median * price / 3600,
    'hourly_price': price, 'currency': os.environ['CURRENCY'],
    'plan_sha256': context['plan_sha256'], 'manifest_sha256': context['manifest_sha256'],
})
(out / 'status.txt').write_text('COMPLETE\n')
print((out / 'summary.json').read_text())
print(f'GPU training benchmark complete. Results: {out}')
PY
