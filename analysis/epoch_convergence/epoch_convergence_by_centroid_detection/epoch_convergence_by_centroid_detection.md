# Centroid detection vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints, centroid detection = predicted centroid within 0.5x GT equivalent diameter, noise floor 100 px.

A mass case is detected when the closest retained predicted centroid lies within half the ground-truth mass's equivalent circular diameter. A Normal case with any retained prediction is a false alarm.

No error bars (single seed).

## All masses

| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |
|------:|--------:|----------:|---------:|-------------:|
| 50 | 0.820 | 1.000 | 0.695 | 0.000 |
| 100 | 0.848 | 1.000 | 0.737 | 0.000 |
| 150 | 0.885 | 0.975 | 0.811 | 0.133 |
| 300 | 0.886 | 0.963 | 0.821 | 0.200 |
| 500 | 0.859 | 0.973 | 0.768 | 0.133 |
| 750 | 0.903 | 0.988 | 0.832 | 0.067 |

## Malignant masses

| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |
|------:|--------:|----------:|---------:|-------------:|
| 50 | 0.926 | 1.000 | 0.862 | 0.000 |
| 100 | 0.952 | 1.000 | 0.908 | 0.000 |
| 150 | 0.953 | 0.968 | 0.938 | 0.133 |
| 300 | 0.954 | 0.954 | 0.954 | 0.200 |
| 500 | 0.937 | 0.967 | 0.908 | 0.133 |
| 750 | 0.969 | 0.984 | 0.954 | 0.067 |

## Benign masses

| Epoch | Case F1 | Case Prec | Case Rec | Case FP rate |
|------:|--------:|----------:|---------:|-------------:|
| 50 | 0.500 | 1.000 | 0.333 | 0.000 |
| 100 | 0.537 | 1.000 | 0.367 | 0.000 |
| 150 | 0.667 | 0.889 | 0.533 | 0.133 |
| 300 | 0.653 | 0.842 | 0.533 | 0.200 |
| 500 | 0.609 | 0.875 | 0.467 | 0.133 |
| 750 | 0.708 | 0.944 | 0.567 | 0.067 |
