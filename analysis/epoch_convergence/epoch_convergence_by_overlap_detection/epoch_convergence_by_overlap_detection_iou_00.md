# Case-level overlap-based detection (IoU>0) by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, detection = overlap (IoU > 0), noise floor 100 px.

No error bars (single seed).

## All masses

| Epoch | Detection | False positive rate |
|------:|----------:|-------------------:|
| 50 | 0.716 | 0.000 |
| 100 | 0.747 | 0.000 |
| 150 | 0.842 | 0.133 |
| 300 | 0.832 | 0.200 |
| 500 | 0.779 | 0.133 |
| 750 | 0.842 | 0.067 |

## Malignant masses

| Epoch | Detection | False positive rate |
|------:|----------:|-------------------:|
| 50 | 0.892 | 0.000 |
| 100 | 0.923 | 0.000 |
| 150 | 0.969 | 0.133 |
| 300 | 0.969 | 0.200 |
| 500 | 0.923 | 0.133 |
| 750 | 0.969 | 0.067 |

## Benign masses

| Epoch | Detection | False positive rate |
|------:|----------:|-------------------:|
| 50 | 0.333 | 0.000 |
| 100 | 0.367 | 0.000 |
| 150 | 0.567 | 0.133 |
| 300 | 0.533 | 0.200 |
| 500 | 0.467 | 0.133 |
| 750 | 0.567 | 0.067 |
