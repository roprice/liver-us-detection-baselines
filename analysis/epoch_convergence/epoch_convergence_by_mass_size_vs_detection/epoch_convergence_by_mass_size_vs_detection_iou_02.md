# Case-level overlap-based detection (IoU>0.2) by saved milestone epoch and mass size

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, detection = IoU > 0.2, noise floor 0.03% of image area.

A mass case is detected when the highest single-component IoU between a retained predicted component and the ground-truth mass exceeds the threshold. Off-target blobs are ignored, so this is comparable to centroid detection.

Mass into 4 bins according to size and overlap detection is measured against each by epoch
- 0–3,000 px² (n=20)
- >3,000–10,000 px² (n=17)
- >10,000–30,000 px² (n=23)
- >30,000 px² (n=35)

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## Both pathologies, 4 log-sized area bins

### 0–3,000 px² (n=20)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.133 ± 0.058 | 0.089 ± 0.102 |
| 100 | 0.167 ± 0.058 | 0.044 ± 0.038 |
| 150 | 0.250 ± 0.132 | 0.133 ± 0.000 |
| 300 | 0.367 ± 0.058 | 0.133 ± 0.067 |
| 500 | 0.317 ± 0.153 | 0.111 ± 0.038 |
| 750 | 0.433 ± 0.029 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.483 ± 0.058 | 0.044 ± 0.038 |
| Best mass | 0.467 ± 0.029 | 0.044 ± 0.038 |
| 1000 (Final) | 0.467 ± 0.029 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.667 ± 0.090 | 0.089 ± 0.102 |
| 100 | 0.745 ± 0.090 | 0.044 ± 0.038 |
| 150 | 0.804 ± 0.034 | 0.133 ± 0.000 |
| 300 | 0.745 ± 0.090 | 0.133 ± 0.067 |
| 500 | 0.804 ± 0.034 | 0.111 ± 0.038 |
| 750 | 0.784 ± 0.034 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.804 ± 0.122 | 0.044 ± 0.038 |
| Best mass | 0.824 ± 0.102 | 0.044 ± 0.038 |
| 1000 (Final) | 0.765 ± 0.059 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=23)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.812 ± 0.109 | 0.089 ± 0.102 |
| 100 | 0.841 ± 0.025 | 0.044 ± 0.038 |
| 150 | 0.870 ± 0.043 | 0.133 ± 0.000 |
| 300 | 0.841 ± 0.091 | 0.133 ± 0.067 |
| 500 | 0.812 ± 0.091 | 0.111 ± 0.038 |
| 750 | 0.899 ± 0.025 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.884 ± 0.050 | 0.044 ± 0.038 |
| Best mass | 0.884 ± 0.050 | 0.044 ± 0.038 |
| 1000 (Final) | 0.855 ± 0.066 | 0.044 ± 0.038 |

### >30,000 px² (n=35)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.943 ± 0.049 | 0.089 ± 0.102 |
| 100 | 0.990 ± 0.016 | 0.044 ± 0.038 |
| 150 | 0.952 ± 0.044 | 0.133 ± 0.000 |
| 300 | 0.981 ± 0.016 | 0.133 ± 0.067 |
| 500 | 0.962 ± 0.016 | 0.111 ± 0.038 |
| 750 | 0.971 ± 0.029 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.962 ± 0.016 | 0.044 ± 0.038 |
| Best mass | 0.952 ± 0.033 | 0.044 ± 0.038 |
| 1000 (Final) | 0.962 ± 0.016 | 0.044 ± 0.038 |

## Malignant, 4 log-sized area bins

### 0–3,000 px² (n=4)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.083 ± 0.144 | 0.089 ± 0.102 |
| 100 | 0.000 ± 0.000 | 0.044 ± 0.038 |
| 150 | 0.250 ± 0.250 | 0.133 ± 0.000 |
| 300 | 0.333 ± 0.382 | 0.133 ± 0.067 |
| 500 | 0.250 ± 0.250 | 0.111 ± 0.038 |
| 750 | 0.333 ± 0.289 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.417 ± 0.144 | 0.044 ± 0.038 |
| Best mass | 0.417 ± 0.144 | 0.044 ± 0.038 |
| 1000 (Final) | 0.417 ± 0.144 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=12)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.778 ± 0.127 | 0.089 ± 0.102 |
| 100 | 0.889 ± 0.048 | 0.044 ± 0.038 |
| 150 | 0.944 ± 0.048 | 0.133 ± 0.000 |
| 300 | 0.889 ± 0.127 | 0.133 ± 0.067 |
| 500 | 0.972 ± 0.048 | 0.111 ± 0.038 |
| 750 | 0.944 ± 0.048 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.917 ± 0.083 | 0.044 ± 0.038 |
| Best mass | 0.944 ± 0.048 | 0.044 ± 0.038 |
| 1000 (Final) | 0.917 ± 0.083 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.843 ± 0.090 | 0.089 ± 0.102 |
| 100 | 0.882 ± 0.059 | 0.044 ± 0.038 |
| 150 | 0.922 ± 0.034 | 0.133 ± 0.000 |
| 300 | 0.882 ± 0.102 | 0.133 ± 0.067 |
| 500 | 0.843 ± 0.122 | 0.111 ± 0.038 |
| 750 | 0.922 ± 0.034 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.922 ± 0.034 | 0.044 ± 0.038 |
| Best mass | 0.922 ± 0.034 | 0.044 ± 0.038 |
| 1000 (Final) | 0.882 ± 0.059 | 0.044 ± 0.038 |

### >30,000 px² (n=32)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.969 ± 0.031 | 0.089 ± 0.102 |
| 100 | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 150 | 0.969 ± 0.031 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.990 ± 0.018 | 0.111 ± 0.038 |
| 750 | 0.990 ± 0.018 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.979 ± 0.018 | 0.044 ± 0.038 |
| Best mass | 0.969 ± 0.031 | 0.044 ± 0.038 |
| 1000 (Final) | 0.979 ± 0.018 | 0.044 ± 0.038 |

## Benign, 4 log-sized area bins

### 0–3,000 px² (n=16)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.146 ± 0.036 | 0.089 ± 0.102 |
| 100 | 0.208 ± 0.072 | 0.044 ± 0.038 |
| 150 | 0.250 ± 0.108 | 0.133 ± 0.000 |
| 300 | 0.375 ± 0.062 | 0.133 ± 0.067 |
| 500 | 0.333 ± 0.130 | 0.111 ± 0.038 |
| 750 | 0.458 ± 0.036 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.500 ± 0.062 | 0.044 ± 0.038 |
| Best mass | 0.479 ± 0.036 | 0.044 ± 0.038 |
| 1000 (Final) | 0.479 ± 0.036 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=5)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.400 ± 0.200 | 0.089 ± 0.102 |
| 100 | 0.400 ± 0.200 | 0.044 ± 0.038 |
| 150 | 0.467 ± 0.115 | 0.133 ± 0.000 |
| 300 | 0.400 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.400 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.400 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.533 ± 0.231 | 0.044 ± 0.038 |
| Best mass | 0.533 ± 0.231 | 0.044 ± 0.038 |
| 1000 (Final) | 0.400 ± 0.000 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=6)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.722 ± 0.192 | 0.089 ± 0.102 |
| 100 | 0.722 ± 0.096 | 0.044 ± 0.038 |
| 150 | 0.722 ± 0.192 | 0.133 ± 0.000 |
| 300 | 0.722 ± 0.096 | 0.133 ± 0.067 |
| 500 | 0.722 ± 0.192 | 0.111 ± 0.038 |
| 750 | 0.833 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.778 ± 0.096 | 0.044 ± 0.038 |
| Best mass | 0.778 ± 0.096 | 0.044 ± 0.038 |
| 1000 (Final) | 0.778 ± 0.096 | 0.044 ± 0.038 |

### >30,000 px² (n=3)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.667 ± 0.333 | 0.089 ± 0.102 |
| 100 | 0.889 ± 0.192 | 0.044 ± 0.038 |
| 150 | 0.778 ± 0.192 | 0.133 ± 0.000 |
| 300 | 0.778 ± 0.192 | 0.133 ± 0.067 |
| 500 | 0.667 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.778 ± 0.192 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.778 ± 0.192 | 0.044 ± 0.038 |
| Best mass | 0.778 ± 0.192 | 0.044 ± 0.038 |
| 1000 (Final) | 0.778 ± 0.192 | 0.044 ± 0.038 |


`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
