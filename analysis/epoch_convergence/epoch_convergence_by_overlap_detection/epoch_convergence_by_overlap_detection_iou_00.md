# Case-level overlap-based detection (IoU>0) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, detection = IoU > 0, noise floor 0.03% of image area.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.761 ± 0.054 | 0.089 ± 0.102 |
| 100 | 0.768 ± 0.021 | 0.044 ± 0.038 |
| 150 | 0.807 ± 0.044 | 0.133 ± 0.000 |
| 300 | 0.807 ± 0.043 | 0.133 ± 0.067 |
| 500 | 0.821 ± 0.038 | 0.111 ± 0.038 |
| 750 | 0.835 ± 0.022 | 0.044 ± 0.038 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.903 ± 0.032 | 0.089 ± 0.102 |
| 100 | 0.923 ± 0.015 | 0.044 ± 0.038 |
| 150 | 0.944 ± 0.032 | 0.133 ± 0.000 |
| 300 | 0.928 ± 0.058 | 0.133 ± 0.067 |
| 500 | 0.938 ± 0.015 | 0.111 ± 0.038 |
| 750 | 0.954 ± 0.027 | 0.044 ± 0.038 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.456 ± 0.117 | 0.089 ± 0.102 |
| 100 | 0.433 ± 0.058 | 0.044 ± 0.038 |
| 150 | 0.511 ± 0.069 | 0.133 ± 0.000 |
| 300 | 0.544 ± 0.019 | 0.133 ± 0.067 |
| 500 | 0.567 ± 0.088 | 0.111 ± 0.038 |
| 750 | 0.578 ± 0.019 | 0.044 ± 0.038 |
