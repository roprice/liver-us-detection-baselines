# Case-level triage-based detection by saved milestone epoch on fold-0 validation data

Three preliminary milestones validation runs: seeds 42/43/44, 125 fold-0 validation images each, triage = any retained mass (no overlap required), noise floor 0.03% of image area.

A mass-present case is detected if the model predicts any retained mass anywhere on the image.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

**Saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 150 | 0.885 ± 0.039 | 0.204 ± 0.140 |
| 300 | 0.907 ± 0.043 | 0.093 ± 0.032 |
| 750 | 0.922 ± 0.024 | 0.056 ± 0.000 |

## Malignant masses

**Saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 150 | 0.953 ± 0.015 | 0.204 ± 0.140 |
| 300 | 0.966 ± 0.020 | 0.093 ± 0.032 |
| 750 | 0.974 ± 0.013 | 0.056 ± 0.000 |

## Benign masses

**Saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 150 | 0.701 ± 0.105 | 0.204 ± 0.140 |
| 300 | 0.747 ± 0.111 | 0.093 ± 0.032 |
| 750 | 0.782 ± 0.080 | 0.056 ± 0.000 |
