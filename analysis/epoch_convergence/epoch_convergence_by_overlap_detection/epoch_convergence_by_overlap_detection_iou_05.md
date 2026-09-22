# Case-level overlap-based detection (IoU>0.5) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, detection = IoU > 0.5, noise floor 0.03% of image area.

A mass case is detected when the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds the threshold. Off-target blobs are ignored, so this is comparable to centroid detection.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.558 ± 0.048 | 0.089 ± 0.102 |
| 100 | 0.593 ± 0.034 | 0.044 ± 0.038 |
| 150 | 0.625 ± 0.022 | 0.133 ± 0.000 |
| 300 | 0.646 ± 0.044 | 0.133 ± 0.067 |
| 500 | 0.663 ± 0.042 | 0.111 ± 0.038 |
| 750 | 0.702 ± 0.022 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.702 ± 0.026 | 0.044 ± 0.038 |
| Best mass | 0.691 ± 0.030 | 0.044 ± 0.038 |
| 1000 (Final) | 0.684 ± 0.018 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Malignant masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.697 ± 0.058 | 0.089 ± 0.102 |
| 100 | 0.738 ± 0.031 | 0.044 ± 0.038 |
| 150 | 0.759 ± 0.039 | 0.133 ± 0.000 |
| 300 | 0.795 ± 0.039 | 0.133 ± 0.067 |
| 500 | 0.790 ± 0.024 | 0.111 ± 0.038 |
| 750 | 0.821 ± 0.049 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.821 ± 0.039 | 0.044 ± 0.038 |
| Best mass | 0.810 ± 0.044 | 0.044 ± 0.038 |
| 1000 (Final) | 0.810 ± 0.044 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Benign masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.256 ± 0.069 | 0.089 ± 0.102 |
| 100 | 0.278 ± 0.051 | 0.044 ± 0.038 |
| 150 | 0.333 ± 0.033 | 0.133 ± 0.000 |
| 300 | 0.322 ± 0.069 | 0.133 ± 0.067 |
| 500 | 0.389 ± 0.107 | 0.111 ± 0.038 |
| 750 | 0.444 ± 0.051 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.444 ± 0.019 | 0.044 ± 0.038 |
| Best mass | 0.433 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.411 ± 0.038 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
