# Case-level centroid-based detection (deq=1) by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Centroid detection = predicted centroid within 1x GT equivalent diameter, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass case is detected when any ground-truth mass component has a retained predicted centroid within 1x its equivalent circular diameter (D = 2 × sqrt(area / π)).

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.579 ± 0.036 | 0.778 ± 0.038 |
| 10 | 0.656 ± 0.022 | 0.622 ± 0.102 |
| 20 | 0.572 ± 0.050 | 0.244 ± 0.038 |
| 40 | 0.709 ± 0.006 | 0.378 ± 0.038 |
| 80 | 0.723 ± 0.016 | 0.178 ± 0.038 |
| 160 | 0.782 ± 0.022 | 0.289 ± 0.038 |
| 320 | 0.828 ± 0.006 | 0.222 ± 0.077 |
| 625 | 0.839 ± 0.006 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.662 ± 0.041 | 0.778 ± 0.038 |
| 10 | 0.810 ± 0.009 | 0.622 ± 0.102 |
| 20 | 0.692 ± 0.055 | 0.244 ± 0.038 |
| 40 | 0.892 ± 0.015 | 0.378 ± 0.038 |
| 80 | 0.867 ± 0.024 | 0.178 ± 0.038 |
| 160 | 0.908 ± 0.027 | 0.289 ± 0.038 |
| 320 | 0.944 ± 0.009 | 0.222 ± 0.077 |
| 625 | 0.954 ± 0.015 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.400 ± 0.033 | 0.778 ± 0.038 |
| 10 | 0.322 ± 0.051 | 0.622 ± 0.102 |
| 20 | 0.311 ± 0.077 | 0.244 ± 0.038 |
| 40 | 0.311 ± 0.038 | 0.378 ± 0.038 |
| 80 | 0.411 ± 0.019 | 0.178 ± 0.038 |
| 160 | 0.511 ± 0.051 | 0.289 ± 0.038 |
| 320 | 0.578 ± 0.038 | 0.222 ± 0.077 |
| 625 | 0.589 ± 0.019 | 0.067 ± 0.000 |
