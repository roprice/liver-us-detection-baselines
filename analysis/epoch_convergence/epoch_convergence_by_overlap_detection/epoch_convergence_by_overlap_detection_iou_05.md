# Case-level overlap-based detection (IoU>=0.5) by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, detection = IoU >= 0.5, noise floor 100 px.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.495 | 0.000 |
| 100 | 0.568 | 0.000 |
| 150 | 0.600 | 0.133 |
| 300 | 0.632 | 0.200 |
| 500 | 0.611 | 0.133 |
| 750 | 0.716 | 0.067 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.615 | 0.000 |
| 100 | 0.723 | 0.000 |
| 150 | 0.723 | 0.133 |
| 300 | 0.800 | 0.200 |
| 500 | 0.769 | 0.133 |
| 750 | 0.862 | 0.067 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.233 | 0.000 |
| 100 | 0.233 | 0.000 |
| 150 | 0.333 | 0.133 |
| 300 | 0.267 | 0.200 |
| 500 | 0.267 | 0.133 |
| 750 | 0.400 | 0.067 |
