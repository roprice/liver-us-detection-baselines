# Case-level centroid-based detection (deq=0.25) by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Centroid detection = predicted centroid within 0.25x GT equivalent diameter, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass case is detected when any ground-truth mass component has a retained predicted centroid within 0.25x its equivalent circular diameter (D = 2 × sqrt(area / π)).

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.147 ± 0.038 | 0.778 ± 0.038 |
| 10 | 0.232 ± 0.011 | 0.622 ± 0.102 |
| 20 | 0.179 ± 0.063 | 0.244 ± 0.038 |
| 40 | 0.326 ± 0.011 | 0.378 ± 0.038 |
| 80 | 0.470 ± 0.044 | 0.178 ± 0.038 |
| 160 | 0.600 ± 0.018 | 0.289 ± 0.038 |
| 320 | 0.635 ± 0.006 | 0.222 ± 0.077 |
| 625 | 0.744 ± 0.012 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.159 ± 0.032 | 0.778 ± 0.038 |
| 10 | 0.267 ± 0.018 | 0.622 ± 0.102 |
| 20 | 0.221 ± 0.085 | 0.244 ± 0.038 |
| 40 | 0.400 ± 0.027 | 0.378 ± 0.038 |
| 80 | 0.585 ± 0.046 | 0.178 ± 0.038 |
| 160 | 0.728 ± 0.024 | 0.289 ± 0.038 |
| 320 | 0.754 ± 0.015 | 0.222 ± 0.077 |
| 625 | 0.856 ± 0.009 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.122 ± 0.051 | 0.778 ± 0.038 |
| 10 | 0.156 ± 0.019 | 0.622 ± 0.102 |
| 20 | 0.089 ± 0.019 | 0.244 ± 0.038 |
| 40 | 0.167 ± 0.033 | 0.378 ± 0.038 |
| 80 | 0.222 ± 0.051 | 0.178 ± 0.038 |
| 160 | 0.322 ± 0.019 | 0.289 ± 0.038 |
| 320 | 0.378 ± 0.038 | 0.222 ± 0.077 |
| 625 | 0.500 ± 0.033 | 0.067 ± 0.000 |
