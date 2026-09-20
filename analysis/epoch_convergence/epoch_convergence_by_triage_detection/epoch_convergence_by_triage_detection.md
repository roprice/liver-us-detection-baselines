# Case-level triage-based detection by saved milestone epoch

Single preliminary milestones test run: seed 42, 625 images, triage = any retained mass (no overlap required), noise floor 100 px.

A mass-present case is detected if the model predicts any retained mass anywhere on the image. A Normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Detection | False positive rate |
|------:|----------:|-------------------:|
| 50 | 0.737 | 0.000 |
| 100 | 0.779 | 0.000 |
| 150 | 0.895 | 0.133 |
| 300 | 0.884 | 0.200 |
| 500 | 0.821 | 0.133 |
| 750 | 0.895 | 0.067 |

## Malignant masses

| Epoch | Detection | False positive rate |
|------:|----------:|-------------------:|
| 50 | 0.908 | 0.000 |
| 100 | 0.938 | 0.000 |
| 150 | 0.985 | 0.133 |
| 300 | 0.985 | 0.200 |
| 500 | 0.954 | 0.133 |
| 750 | 0.985 | 0.067 |

## Benign masses

| Epoch | Detection | False positive rate |
|------:|----------:|-------------------:|
| 50 | 0.367 | 0.000 |
| 100 | 0.433 | 0.000 |
| 150 | 0.700 | 0.133 |
| 300 | 0.667 | 0.200 |
| 500 | 0.533 | 0.133 |
| 750 | 0.700 | 0.067 |
