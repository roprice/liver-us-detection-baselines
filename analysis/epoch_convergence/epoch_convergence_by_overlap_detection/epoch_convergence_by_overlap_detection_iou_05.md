# Case-level overlap-based detection (IoU>=0.5) by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, detection = IoU >= 0.5, noise floor 100 px.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.505 | 0.000 |
| 100 | 0.579 | 0.000 |
| 150 | 0.632 | 0.133 |
| 300 | 0.632 | 0.200 |
| 500 | 0.611 | 0.133 |
| 750 | 0.726 | 0.067 |

## Malignant masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.631 | 0.000 |
| 100 | 0.738 | 0.000 |
| 150 | 0.754 | 0.133 |
| 300 | 0.800 | 0.200 |
| 500 | 0.769 | 0.133 |
| 750 | 0.877 | 0.067 |

## Benign masses

| Epoch | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.233 | 0.000 |
| 100 | 0.233 | 0.000 |
| 150 | 0.367 | 0.133 |
| 300 | 0.267 | 0.200 |
| 500 | 0.267 | 0.133 |
| 750 | 0.400 | 0.067 |
