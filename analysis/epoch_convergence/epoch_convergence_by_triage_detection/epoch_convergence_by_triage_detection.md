# Triage detection vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints, triage = any retained mass (no overlap required), noise floor 100 px.

A mass-present case is detected if the model predicts any retained mass anywhere on the image. A Normal case with any prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |
|------:|--------:|----------:|---------:|-------------:|
| 50 | 0.848 | 1.000 | 0.737 | 0.000 |
| 100 | 0.876 | 1.000 | 0.779 | 0.000 |
| 150 | 0.934 | 0.977 | 0.895 | 0.133 |
| 300 | 0.923 | 0.966 | 0.884 | 0.200 |
| 500 | 0.891 | 0.975 | 0.821 | 0.133 |
| 750 | 0.939 | 0.988 | 0.895 | 0.067 |

## Malignant masses

| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |
|------:|--------:|----------:|---------:|-------------:|
| 50 | 0.952 | 1.000 | 0.908 | 0.000 |
| 100 | 0.968 | 1.000 | 0.938 | 0.000 |
| 150 | 0.977 | 0.970 | 0.985 | 0.133 |
| 300 | 0.970 | 0.955 | 0.985 | 0.200 |
| 500 | 0.961 | 0.969 | 0.954 | 0.133 |
| 750 | 0.985 | 0.985 | 0.985 | 0.067 |

## Benign masses

| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |
|------:|--------:|----------:|---------:|-------------:|
| 50 | 0.537 | 1.000 | 0.367 | 0.000 |
| 100 | 0.605 | 1.000 | 0.433 | 0.000 |
| 150 | 0.792 | 0.913 | 0.700 | 0.133 |
| 300 | 0.755 | 0.870 | 0.667 | 0.200 |
| 500 | 0.667 | 0.889 | 0.533 | 0.133 |
| 750 | 0.808 | 0.955 | 0.700 | 0.067 |
