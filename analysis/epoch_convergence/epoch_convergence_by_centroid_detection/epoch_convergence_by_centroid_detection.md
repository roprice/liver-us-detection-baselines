# Case-level centroid-based detection by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, centroid detection = predicted centroid within 0.5x GT equivalent diameter, noise floor 100 px.

A mass case is detected when the closest retained predicted centroid lies within half the ground-truth mass's equivalent circular diameter.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.695 | 0.000 |
| 100 | 0.737 | 0.000 |
| 150 | 0.811 | 0.133 |
| 300 | 0.821 | 0.200 |
| 500 | 0.768 | 0.133 |
| 750 | 0.832 | 0.067 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.862 | 0.000 |
| 100 | 0.908 | 0.000 |
| 150 | 0.938 | 0.133 |
| 300 | 0.954 | 0.200 |
| 500 | 0.908 | 0.133 |
| 750 | 0.954 | 0.067 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.333 | 0.000 |
| 100 | 0.367 | 0.000 |
| 150 | 0.533 | 0.133 |
| 300 | 0.533 | 0.200 |
| 500 | 0.467 | 0.133 |
| 750 | 0.567 | 0.067 |
