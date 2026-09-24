# Case-level overlap-based detection (IoU>0.5) by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Detection = IoU > 0.5, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass-present case is detected when `overlap_detection_iou_05_flag` is True: the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds 0.5. Off-target blobs are ignored.

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.018 ± 0.006 | 0.778 ± 0.038 |
| 10 | 0.105 ± 0.011 | 0.622 ± 0.102 |
| 20 | 0.060 ± 0.016 | 0.244 ± 0.038 |
| 40 | 0.182 ± 0.022 | 0.378 ± 0.038 |
| 80 | 0.337 ± 0.028 | 0.178 ± 0.038 |
| 160 | 0.474 ± 0.021 | 0.289 ± 0.038 |
| 320 | 0.558 ± 0.028 | 0.222 ± 0.077 |
| 625 | 0.653 ± 0.011 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.021 ± 0.018 | 0.778 ± 0.038 |
| 10 | 0.113 ± 0.009 | 0.622 ± 0.102 |
| 20 | 0.077 ± 0.027 | 0.244 ± 0.038 |
| 40 | 0.215 ± 0.041 | 0.378 ± 0.038 |
| 80 | 0.441 ± 0.039 | 0.178 ± 0.038 |
| 160 | 0.605 ± 0.024 | 0.289 ± 0.038 |
| 320 | 0.703 ± 0.039 | 0.222 ± 0.077 |
| 625 | 0.805 ± 0.024 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.011 ± 0.019 | 0.778 ± 0.038 |
| 10 | 0.089 ± 0.019 | 0.622 ± 0.102 |
| 20 | 0.022 ± 0.038 | 0.244 ± 0.038 |
| 40 | 0.111 ± 0.019 | 0.378 ± 0.038 |
| 80 | 0.111 ± 0.038 | 0.178 ± 0.038 |
| 160 | 0.189 ± 0.019 | 0.289 ± 0.038 |
| 320 | 0.244 ± 0.038 | 0.222 ± 0.077 |
| 625 | 0.322 ± 0.019 | 0.067 ± 0.000 |
