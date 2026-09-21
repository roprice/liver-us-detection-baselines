# Case-level centroid-based detection (deq=0.5) by saved milestone epoch

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, centroid detection = predicted centroid within 0.5x GT equivalent diameter, noise floor 0.03% of image area.

A mass case is detected when the closest retained predicted centroid lies within 0.5x the ground-truth mass's equivalent circular diameter.

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## All masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.719 ± 0.043 | 0.089 ± 0.102 |
| 100 | 0.754 ± 0.030 | 0.044 ± 0.038 |
| 150 | 0.793 ± 0.030 | 0.133 ± 0.000 |
| 300 | 0.793 ± 0.049 | 0.133 ± 0.067 |
| 500 | 0.807 ± 0.037 | 0.111 ± 0.038 |
| 750 | 0.818 ± 0.016 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.842 ± 0.011 | 0.044 ± 0.038 |
| Best mass | 0.835 ± 0.006 | 0.044 ± 0.038 |
| 1000 (Final) | 0.832 ± 0.000 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Malignant masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.867 ± 0.054 | 0.089 ± 0.102 |
| 100 | 0.903 ± 0.039 | 0.044 ± 0.038 |
| 150 | 0.933 ± 0.024 | 0.133 ± 0.000 |
| 300 | 0.908 ± 0.067 | 0.133 ± 0.067 |
| 500 | 0.918 ± 0.018 | 0.111 ± 0.038 |
| 750 | 0.928 ± 0.024 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.938 ± 0.000 | 0.044 ± 0.038 |
| Best mass | 0.928 ± 0.018 | 0.044 ± 0.038 |
| 1000 (Final) | 0.938 ± 0.000 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.

## Benign masses

**Predetermined saved checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.400 ± 0.058 | 0.089 ± 0.102 |
| 100 | 0.433 ± 0.058 | 0.044 ± 0.038 |
| 150 | 0.489 ± 0.051 | 0.133 ± 0.000 |
| 300 | 0.544 ± 0.019 | 0.133 ± 0.067 |
| 500 | 0.567 ± 0.088 | 0.111 ± 0.038 |
| 750 | 0.578 ± 0.019 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.633 ± 0.033 | 0.044 ± 0.038 |
| Best mass | 0.633 ± 0.033 | 0.044 ± 0.038 |
| 1000 (Final) | 0.600 ± 0.000 | 0.044 ± 0.038 |

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
