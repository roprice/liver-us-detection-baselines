# Case-level overlap-based detection (IoU>0.5) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, detection = IoU > 0.5, noise floor 0.03% of image area.

A mass case is detected when the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds the threshold. Off-target blobs are ignored, so this is comparable to centroid detection.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.558 ± 0.048 | 0.089 ± 0.102 |
| 100 | 0.593 ± 0.034 | 0.044 ± 0.038 |
| 150 | 0.625 ± 0.022 | 0.133 ± 0.000 |
| 300 | 0.646 ± 0.044 | 0.133 ± 0.067 |
| 500 | 0.660 ± 0.047 | 0.111 ± 0.038 |
| 750 | 0.702 ± 0.022 | 0.044 ± 0.038 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.697 ± 0.058 | 0.089 ± 0.102 |
| 100 | 0.738 ± 0.031 | 0.044 ± 0.038 |
| 150 | 0.759 ± 0.039 | 0.133 ± 0.000 |
| 300 | 0.795 ± 0.039 | 0.133 ± 0.067 |
| 500 | 0.785 ± 0.027 | 0.111 ± 0.038 |
| 750 | 0.821 ± 0.049 | 0.044 ± 0.038 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.256 ± 0.069 | 0.089 ± 0.102 |
| 100 | 0.278 ± 0.051 | 0.044 ± 0.038 |
| 150 | 0.333 ± 0.033 | 0.133 ± 0.000 |
| 300 | 0.322 ± 0.069 | 0.133 ± 0.067 |
| 500 | 0.389 ± 0.107 | 0.111 ± 0.038 |
| 750 | 0.444 ± 0.051 | 0.044 ± 0.038 |
