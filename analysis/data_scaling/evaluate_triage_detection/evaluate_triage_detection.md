# Case-level triage-based detection by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Triage = any retained mass (no overlap required), noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass-present case is detected if the model predicts any retained mass anywhere on the image. No overlap or IoU threshold is required.

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.944 ± 0.052 | 0.778 ± 0.038 |
| 10 | 0.881 ± 0.026 | 0.622 ± 0.102 |
| 20 | 0.782 ± 0.043 | 0.244 ± 0.038 |
| 40 | 0.821 ± 0.011 | 0.378 ± 0.038 |
| 80 | 0.821 ± 0.021 | 0.178 ± 0.038 |
| 160 | 0.846 ± 0.016 | 0.289 ± 0.038 |
| 320 | 0.891 ± 0.016 | 0.222 ± 0.077 |
| 625 | 0.909 ± 0.032 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.979 ± 0.024 | 0.778 ± 0.038 |
| 10 | 0.938 ± 0.027 | 0.622 ± 0.102 |
| 20 | 0.862 ± 0.015 | 0.244 ± 0.038 |
| 40 | 0.933 ± 0.009 | 0.378 ± 0.038 |
| 80 | 0.903 ± 0.009 | 0.178 ± 0.038 |
| 160 | 0.954 ± 0.015 | 0.289 ± 0.038 |
| 320 | 0.969 ± 0.000 | 0.222 ± 0.077 |
| 625 | 0.969 ± 0.015 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.867 ± 0.120 | 0.778 ± 0.038 |
| 10 | 0.756 ± 0.051 | 0.622 ± 0.102 |
| 20 | 0.611 ± 0.117 | 0.244 ± 0.038 |
| 40 | 0.578 ± 0.019 | 0.378 ± 0.038 |
| 80 | 0.644 ± 0.051 | 0.178 ± 0.038 |
| 160 | 0.611 ± 0.077 | 0.289 ± 0.038 |
| 320 | 0.722 ± 0.051 | 0.222 ± 0.077 |
| 625 | 0.778 ± 0.084 | 0.067 ± 0.000 |
