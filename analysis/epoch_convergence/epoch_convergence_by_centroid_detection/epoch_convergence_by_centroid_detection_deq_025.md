# Case-level centroid-based detection (deq=0.25) by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, centroid detection = predicted centroid within 0.25x GT equivalent diameter, noise floor 0.03% of image area.

A mass case is detected when the closest retained predicted centroid lies within 0.25x the ground-truth mass's equivalent circular diameter.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.589 | 0.000 |
| 100 | 0.674 | 0.000 |
| 150 | 0.716 | 0.133 |
| 300 | 0.758 | 0.200 |
| 500 | 0.663 | 0.133 |
| 750 | 0.768 | 0.067 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.723 | 0.000 |
| 100 | 0.831 | 0.000 |
| 150 | 0.831 | 0.133 |
| 300 | 0.908 | 0.200 |
| 500 | 0.815 | 0.133 |
| 750 | 0.892 | 0.067 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.300 | 0.000 |
| 100 | 0.333 | 0.000 |
| 150 | 0.467 | 0.133 |
| 300 | 0.433 | 0.200 |
| 500 | 0.333 | 0.133 |
| 750 | 0.500 | 0.067 |
