# Case-level overlap-based detection (IoU>0) by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Detection = IoU > 0, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass-present case is detected when `overlap_detection_iou_00_flag` is True: the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds 0. Off-target blobs are ignored.

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.533 ± 0.026 | 0.778 ± 0.038 |
| 10 | 0.642 ± 0.032 | 0.622 ± 0.102 |
| 20 | 0.547 ± 0.036 | 0.244 ± 0.038 |
| 40 | 0.691 ± 0.006 | 0.378 ± 0.038 |
| 80 | 0.733 ± 0.016 | 0.178 ± 0.038 |
| 160 | 0.786 ± 0.026 | 0.289 ± 0.038 |
| 320 | 0.832 ± 0.011 | 0.222 ± 0.077 |
| 625 | 0.839 ± 0.006 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.615 ± 0.046 | 0.778 ± 0.038 |
| 10 | 0.805 ± 0.032 | 0.622 ± 0.102 |
| 20 | 0.677 ± 0.041 | 0.244 ± 0.038 |
| 40 | 0.882 ± 0.009 | 0.378 ± 0.038 |
| 80 | 0.867 ± 0.024 | 0.178 ± 0.038 |
| 160 | 0.908 ± 0.027 | 0.289 ± 0.038 |
| 320 | 0.949 ± 0.009 | 0.222 ± 0.077 |
| 625 | 0.954 ± 0.015 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.356 ± 0.019 | 0.778 ± 0.038 |
| 10 | 0.289 ± 0.038 | 0.622 ± 0.102 |
| 20 | 0.267 ± 0.033 | 0.244 ± 0.038 |
| 40 | 0.278 ± 0.038 | 0.378 ± 0.038 |
| 80 | 0.444 ± 0.019 | 0.178 ± 0.038 |
| 160 | 0.522 ± 0.051 | 0.289 ± 0.038 |
| 320 | 0.578 ± 0.038 | 0.222 ± 0.077 |
| 625 | 0.589 ± 0.019 | 0.067 ± 0.000 |
