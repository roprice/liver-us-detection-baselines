# Case-level centroid-based detection (deq=1) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, centroid detection = predicted centroid within 1x GT equivalent diameter, noise floor 0.03% of image area.

A mass case is detected when the closest retained predicted centroid lies within 1x the ground-truth mass's equivalent circular diameter.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.761 ± 0.054 | 0.089 ± 0.102 |
| 100 | 0.768 ± 0.021 | 0.044 ± 0.038 |
| 150 | 0.807 ± 0.044 | 0.133 ± 0.000 |
| 300 | 0.807 ± 0.043 | 0.133 ± 0.067 |
| 500 | 0.821 ± 0.038 | 0.111 ± 0.038 |
| 750 | 0.839 ± 0.024 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.853 ± 0.000 | 0.044 ± 0.038 |
| Best mass | 0.853 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.842 ± 0.011 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Malignant masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.903 ± 0.032 | 0.089 ± 0.102 |
| 100 | 0.923 ± 0.015 | 0.044 ± 0.038 |
| 150 | 0.944 ± 0.032 | 0.133 ± 0.000 |
| 300 | 0.928 ± 0.058 | 0.133 ± 0.067 |
| 500 | 0.938 ± 0.015 | 0.111 ± 0.038 |
| 750 | 0.954 ± 0.027 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.954 ± 0.015 | 0.044 ± 0.038 |
| Best mass | 0.954 ± 0.015 | 0.044 ± 0.038 |
| 1000 (Final) | 0.954 ± 0.015 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Benign masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.456 ± 0.117 | 0.089 ± 0.102 |
| 100 | 0.433 ± 0.058 | 0.044 ± 0.038 |
| 150 | 0.511 ± 0.069 | 0.133 ± 0.000 |
| 300 | 0.544 ± 0.019 | 0.133 ± 0.067 |
| 500 | 0.567 ± 0.088 | 0.111 ± 0.038 |
| 750 | 0.589 ± 0.019 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.633 ± 0.033 | 0.044 ± 0.038 |
| Best mass | 0.633 ± 0.033 | 0.044 ± 0.038 |
| 1000 (Final) | 0.600 ± 0.000 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
