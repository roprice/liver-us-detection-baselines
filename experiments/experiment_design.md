# Experiment design notes

Methods rationale for the experiments in `experiment_runbooks/`. The runbooks cover procedure only.

## Data-scaling experiment

Runbook: [`run_data_scaling.md`](experiment_runbooks/run_data_scaling.md)

### Subset construction

The source training pool contains 370 malignant, 170 benign, and 85 mass-negative cases. The runner shuffles these three groups independently once with NumPy seed 42. It first assigns the mass-positive versus mass-negative allocation proportionally. It then divides each mass-positive allocation proportionally into malignant and benign cases. Each pool takes a prefix from every shuffled stratum, so every smaller pool is contained in every larger pool. The 625-case pool is the unmodified source training set.

| Size | Malignant | Benign | Mass-negative |
|---:|---:|---:|---:|
| 5 | 3 | 1 | 1 |
| 10 | 6 | 3 | 1 |
| 20 | 12 | 5 | 3 |
| 40 | 24 | 11 | 5 |
| 80 | 47 | 22 | 11 |
| 160 | 95 | 43 | 22 |
| 320 | 189 | 87 | 44 |
| 625 | 370 | 170 | 85 |

### Evaluation checkpoint

Every run is evaluated at `checkpoint_final.pth`. This is the fixed 150-epoch budget selected by the milestones pilot. It avoids letting a size-specific internal-validation checkpoint decide when each model is evaluated.

### Pathology-preserving split repair

Pool sizes describe the raw datasets, not the number of gradient-training images. The trainer starts from nnU-Net's fold-0 split and checks that the training partition contains malignant, benign, and normal examples.

If a pathology is absent, the trainer swaps one validation case of that pathology with a training case from a category holding at least two training cases. Missing categories are processed in malignant, benign, normal order. Candidates are chosen by sorted case ID, independent of the initialization seed. Splits that already contain all three categories are unchanged. Partition sizes and disjointness are preserved.

This guarantees representation, not proportionality within the training partition. Training partitions are also not guaranteed to be nested across sizes.

For the five-case pool, the repaired split has two malignant, one benign, and one normal training image, plus one malignant validation image. All three seeds use the same effective split.

The original `splits_final.json` is left unchanged. The authoritative effective assignments and counts are saved in each model's `fold_0/data_scaling_split.json`. The test set is never used for preprocessing decisions, gradient updates, checkpoint selection, or subset construction.

## ResEnc architecture ablation

Runbook: [`run_resenc_ablation.md`](experiment_runbooks/run_resenc_ablation.md)

### Why ResEnc M

nnU-Net describes ResEnc M as the residual-encoder preset with a GPU budget closest to the standard PlainConvUNet. ResEnc L and XL would add substantially larger compute and memory budgets.

### Preset, not a one-variable substitution

The ablation evaluates the ResEnc M preset as a whole. Its planner may choose a different patch size or batch size from `nnUNetPlans`. Both plans files are preserved so these differences can be accounted for when interpreting results.

### Trainer

`nnUNetTrainerResEncAblation` inherits the data-scaling trainer. That includes the 150-epoch budget, seed handling, pathology-preserving split repair, checkpoint policy, and memory and parameter instrumentation. It additionally rejects any initialized network that is not a `ResidualEncoderUNet`.

### Comparison checklist

Before comparing against PlainConvUNet, confirm the matched data-scaling run used the same dataset name, fold, seeds, epoch budget, held-out test images, and `checkpoint_final.pth`. Compare the archived `nnUNetPlans.json` and `nnUNetResEncUNetMPlans.json` when discussing architecture, patch-size, or batch-size effects.
