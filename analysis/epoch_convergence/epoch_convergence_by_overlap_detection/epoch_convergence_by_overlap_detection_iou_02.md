# Case-level overlap-based detection (IoU>0.2) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, detection = IoU > 0.2, noise floor 0.03% of image area.

A mass case is detected when the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds the threshold. Off-target blobs are ignored, so this is comparable to centroid detection.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.691 ± 0.068 | 0.089 ± 0.102 |
| 100 | 0.737 ± 0.011 | 0.044 ± 0.038 |
| 150 | 0.758 ± 0.011 | 0.133 ± 0.000 |
| 300 | 0.775 ± 0.052 | 0.133 ± 0.067 |
| 500 | 0.761 ± 0.054 | 0.111 ± 0.038 |
| 750 | 0.807 ± 0.012 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.814 ± 0.026 | 0.044 ± 0.038 |
| Best mass | 0.811 ± 0.032 | 0.044 ± 0.038 |
| 1000 (Final) | 0.796 ± 0.024 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Malignant masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.846 ± 0.067 | 0.089 ± 0.102 |
| 100 | 0.887 ± 0.024 | 0.044 ± 0.038 |
| 150 | 0.908 ± 0.015 | 0.133 ± 0.000 |
| 300 | 0.908 ± 0.067 | 0.133 ± 0.067 |
| 500 | 0.903 ± 0.047 | 0.111 ± 0.038 |
| 750 | 0.923 ± 0.015 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.918 ± 0.024 | 0.044 ± 0.038 |
| Best mass | 0.918 ± 0.024 | 0.044 ± 0.038 |
| 1000 (Final) | 0.908 ± 0.027 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Benign masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.356 ± 0.107 | 0.089 ± 0.102 |
| 100 | 0.411 ± 0.038 | 0.044 ± 0.038 |
| 150 | 0.433 ± 0.033 | 0.133 ± 0.000 |
| 300 | 0.489 ± 0.038 | 0.133 ± 0.067 |
| 500 | 0.456 ± 0.107 | 0.111 ± 0.038 |
| 750 | 0.556 ± 0.019 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.589 ± 0.038 | 0.044 ± 0.038 |
| Best mass | 0.578 ± 0.051 | 0.044 ± 0.038 |
| 1000 (Final) | 0.556 ± 0.019 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
