# Case-level overlap-based detection (IoU>0.2) by training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625. Detection = IoU > 0.2, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A mass-present case is detected when `overlap_detection_iou_02_flag` is True: the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds 0.2. Off-target blobs are ignored.

Each table reports case-level false positives computed on normal cases only. A normal case with any retained predicted mass component is a false alarm.

Values are mean ± sample standard deviation across the three seeds.

## All masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.182 ± 0.012 | 0.778 ± 0.038 |
| 10 | 0.305 ± 0.011 | 0.622 ± 0.102 |
| 20 | 0.235 ± 0.030 | 0.244 ± 0.038 |
| 40 | 0.418 ± 0.006 | 0.378 ± 0.038 |
| 80 | 0.568 ± 0.032 | 0.178 ± 0.038 |
| 160 | 0.702 ± 0.022 | 0.289 ± 0.038 |
| 320 | 0.772 ± 0.006 | 0.222 ± 0.077 |
| 625 | 0.804 ± 0.012 | 0.067 ± 0.000 |

## Malignant masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.195 ± 0.018 | 0.778 ± 0.038 |
| 10 | 0.374 ± 0.018 | 0.622 ± 0.102 |
| 20 | 0.287 ± 0.036 | 0.244 ± 0.038 |
| 40 | 0.538 ± 0.000 | 0.378 ± 0.038 |
| 80 | 0.718 ± 0.039 | 0.178 ± 0.038 |
| 160 | 0.821 ± 0.024 | 0.289 ± 0.038 |
| 320 | 0.918 ± 0.009 | 0.222 ± 0.077 |
| 625 | 0.944 ± 0.009 | 0.067 ± 0.000 |

## Benign masses

| Training size | Detection | Normal cases FP rate |
|--------------:|----------:|---------------------:|
| 5 | 0.156 ± 0.038 | 0.778 ± 0.038 |
| 10 | 0.156 ± 0.019 | 0.622 ± 0.102 |
| 20 | 0.122 ± 0.019 | 0.244 ± 0.038 |
| 40 | 0.156 ± 0.019 | 0.378 ± 0.038 |
| 80 | 0.244 ± 0.019 | 0.178 ± 0.038 |
| 160 | 0.444 ± 0.038 | 0.289 ± 0.038 |
| 320 | 0.456 ± 0.038 | 0.222 ± 0.077 |
| 625 | 0.500 ± 0.058 | 0.067 ± 0.000 |
