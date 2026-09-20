# Case-level overlap-based detection (IoU>=0.2) by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, detection = IoU >= 0.2, noise floor 100 px.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.621 | 0.000 |
| 100 | 0.737 | 0.000 |
| 150 | 0.768 | 0.133 |
| 300 | 0.800 | 0.200 |
| 500 | 0.695 | 0.133 |
| 750 | 0.821 | 0.067 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.800 | 0.000 |
| 100 | 0.908 | 0.000 |
| 150 | 0.923 | 0.133 |
| 300 | 0.954 | 0.200 |
| 500 | 0.877 | 0.133 |
| 750 | 0.938 | 0.067 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.233 | 0.000 |
| 100 | 0.367 | 0.000 |
| 150 | 0.433 | 0.133 |
| 300 | 0.467 | 0.200 |
| 500 | 0.300 | 0.133 |
| 750 | 0.567 | 0.067 |
