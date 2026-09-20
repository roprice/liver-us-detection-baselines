# Case-level triage-based detection by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, triage = any retained mass (no overlap required), noise floor 0.03% of image area.

A mass-present case is detected if the model predicts any retained mass anywhere on the image.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.818 ± 0.098 | 0.089 ± 0.102 |
| 100 | 0.818 ± 0.034 | 0.044 ± 0.038 |
| 150 | 0.853 ± 0.064 | 0.133 ± 0.000 |
| 300 | 0.874 ± 0.028 | 0.133 ± 0.067 |
| 500 | 0.874 ± 0.053 | 0.111 ± 0.038 |
| 750 | 0.881 ± 0.024 | 0.044 ± 0.038 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.923 ± 0.027 | 0.089 ± 0.102 |
| 100 | 0.938 ± 0.015 | 0.044 ± 0.038 |
| 150 | 0.959 ± 0.032 | 0.133 ± 0.000 |
| 300 | 0.959 ± 0.032 | 0.133 ± 0.067 |
| 500 | 0.959 ± 0.009 | 0.111 ± 0.038 |
| 750 | 0.969 ± 0.015 | 0.044 ± 0.038 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.589 ± 0.255 | 0.089 ± 0.102 |
| 100 | 0.556 ± 0.117 | 0.044 ± 0.038 |
| 150 | 0.622 ± 0.135 | 0.133 ± 0.000 |
| 300 | 0.689 ± 0.038 | 0.133 ± 0.067 |
| 500 | 0.689 ± 0.150 | 0.111 ± 0.038 |
| 750 | 0.689 ± 0.051 | 0.044 ± 0.038 |
