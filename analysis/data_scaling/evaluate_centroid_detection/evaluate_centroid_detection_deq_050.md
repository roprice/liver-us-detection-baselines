# Case-level centroid-based detection (deq=0.5) by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Centroid detection = predicted centroid within 0.5x GT equivalent diameter, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass case is detected when any ground-truth mass component has a retained predicted centroid within 0.5x its equivalent circular diameter (D = 2 × sqrt(area / π)).

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.351 ± 0.030 | 0.778 ± 0.038 |
| 10 | 0.474 ± 0.011 | 0.622 ± 0.102 |
| 20 | 0.418 ± 0.070 | 0.244 ± 0.038 |
| 40 | 0.596 ± 0.022 | 0.378 ± 0.038 |
| 80 | 0.646 ± 0.016 | 0.178 ± 0.038 |
| 160 | 0.765 ± 0.026 | 0.289 ± 0.038 |
| 320 | 0.782 ± 0.006 | 0.222 ± 0.077 |
| 625 | 0.825 ± 0.006 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.405 ± 0.032 | 0.778 ± 0.038 |
| 10 | 0.574 ± 0.039 | 0.622 ± 0.102 |
| 20 | 0.513 ± 0.073 | 0.244 ± 0.038 |
| 40 | 0.754 ± 0.027 | 0.378 ± 0.038 |
| 80 | 0.810 ± 0.024 | 0.178 ± 0.038 |
| 160 | 0.897 ± 0.024 | 0.289 ± 0.038 |
| 320 | 0.903 ± 0.009 | 0.222 ± 0.077 |
| 625 | 0.938 ± 0.015 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.233 ± 0.033 | 0.778 ± 0.038 |
| 10 | 0.256 ± 0.051 | 0.622 ± 0.102 |
| 20 | 0.211 ± 0.069 | 0.244 ± 0.038 |
| 40 | 0.256 ± 0.019 | 0.378 ± 0.038 |
| 80 | 0.289 ± 0.019 | 0.178 ± 0.038 |
| 160 | 0.478 ± 0.038 | 0.289 ± 0.038 |
| 320 | 0.522 ± 0.019 | 0.222 ± 0.077 |
| 625 | 0.578 ± 0.019 | 0.067 ± 0.000 |
