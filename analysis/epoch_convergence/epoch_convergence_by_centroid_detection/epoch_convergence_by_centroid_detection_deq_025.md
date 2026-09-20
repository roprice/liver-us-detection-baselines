# Case-level centroid-based detection (deq=0.25) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, centroid detection = predicted centroid within 0.25x GT equivalent diameter, noise floor 0.03% of image area.

A mass case is detected when the closest retained predicted centroid lies within 0.25x the ground-truth mass's equivalent circular diameter.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.642 ± 0.046 | 0.089 ± 0.102 |
| 100 | 0.691 ± 0.040 | 0.044 ± 0.038 |
| 150 | 0.719 ± 0.006 | 0.133 ± 0.000 |
| 300 | 0.740 ± 0.022 | 0.133 ± 0.067 |
| 500 | 0.733 ± 0.062 | 0.111 ± 0.038 |
| 750 | 0.747 ± 0.018 | 0.044 ± 0.038 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.774 ± 0.047 | 0.089 ± 0.102 |
| 100 | 0.826 ± 0.039 | 0.044 ± 0.038 |
| 150 | 0.846 ± 0.015 | 0.133 ± 0.000 |
| 300 | 0.862 ± 0.055 | 0.133 ± 0.067 |
| 500 | 0.856 ± 0.036 | 0.111 ± 0.038 |
| 750 | 0.856 ± 0.032 | 0.044 ± 0.038 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.356 ± 0.051 | 0.089 ± 0.102 |
| 100 | 0.400 ± 0.067 | 0.044 ± 0.038 |
| 150 | 0.444 ± 0.038 | 0.133 ± 0.000 |
| 300 | 0.478 ± 0.051 | 0.133 ± 0.067 |
| 500 | 0.467 ± 0.120 | 0.111 ± 0.038 |
| 750 | 0.511 ± 0.019 | 0.044 ± 0.038 |
