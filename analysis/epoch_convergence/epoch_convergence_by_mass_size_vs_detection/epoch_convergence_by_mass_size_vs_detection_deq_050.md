# Case-level centroid-based detection (deq=0.5) by saved milestone epoch and mass size

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, centroid detection = predicted centroid within 0.5x GT equivalent diameter, noise floor 0.03% of image area.

A mass case is detected when the closest retained predicted centroid lies within 0.5x the ground-truth mass's equivalent circular diameter.

Mass into 4 bins according to size and centroid detection is measured against each by epoch
- 0–3,000 px² (n=20)
- >3,000–10,000 px² (n=17)
- >10,000–30,000 px² (n=23)
- >30,000 px² (n=35)

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## Both pathologies, 4 log-sized area bins

### 0–3,000 px² (n=20)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.150 ± 0.050 | 0.089 ± 0.102 |
| 100 | 0.250 ± 0.087 | 0.044 ± 0.038 |
| 150 | 0.283 ± 0.161 | 0.133 ± 0.000 |
| 300 | 0.383 ± 0.076 | 0.133 ± 0.067 |
| 500 | 0.383 ± 0.161 | 0.111 ± 0.038 |
| 750 | 0.483 ± 0.029 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.517 ± 0.058 | 0.044 ± 0.038 |
| Best mass | 0.517 ± 0.058 | 0.044 ± 0.038 |
| 1000 (Final) | 0.517 ± 0.058 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.706 ± 0.102 | 0.089 ± 0.102 |
| 100 | 0.706 ± 0.118 | 0.044 ± 0.038 |
| 150 | 0.804 ± 0.034 | 0.133 ± 0.000 |
| 300 | 0.745 ± 0.136 | 0.133 ± 0.067 |
| 500 | 0.824 ± 0.059 | 0.111 ± 0.038 |
| 750 | 0.765 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.843 ± 0.068 | 0.044 ± 0.038 |
| Best mass | 0.784 ± 0.090 | 0.044 ± 0.038 |
| 1000 (Final) | 0.784 ± 0.034 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=23)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.826 ± 0.043 | 0.089 ± 0.102 |
| 100 | 0.870 ± 0.043 | 0.044 ± 0.038 |
| 150 | 0.913 ± 0.000 | 0.133 ± 0.000 |
| 300 | 0.870 ± 0.043 | 0.133 ± 0.067 |
| 500 | 0.870 ± 0.043 | 0.111 ± 0.038 |
| 750 | 0.884 ± 0.025 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.913 ± 0.043 | 0.044 ± 0.038 |
| Best mass | 0.899 ± 0.025 | 0.044 ± 0.038 |
| 1000 (Final) | 0.899 ± 0.025 | 0.044 ± 0.038 |

### >30,000 px² (n=35)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.981 ± 0.016 | 0.089 ± 0.102 |
| 100 | 0.990 ± 0.016 | 0.044 ± 0.038 |
| 150 | 1.000 ± 0.000 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 1.000 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.990 ± 0.016 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.981 ± 0.033 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.990 ± 0.016 | 0.044 ± 0.038 |

## Malignant, 4 log-sized area bins

### 0–3,000 px² (n=4)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.083 ± 0.144 | 0.089 ± 0.102 |
| 100 | 0.333 ± 0.144 | 0.044 ± 0.038 |
| 150 | 0.417 ± 0.382 | 0.133 ± 0.000 |
| 300 | 0.417 ± 0.382 | 0.133 ± 0.067 |
| 500 | 0.333 ± 0.289 | 0.111 ± 0.038 |
| 750 | 0.583 ± 0.144 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.583 ± 0.144 | 0.044 ± 0.038 |
| Best mass | 0.583 ± 0.144 | 0.044 ± 0.038 |
| 1000 (Final) | 0.583 ± 0.144 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=12)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.806 ± 0.173 | 0.089 ± 0.102 |
| 100 | 0.833 ± 0.083 | 0.044 ± 0.038 |
| 150 | 0.944 ± 0.048 | 0.133 ± 0.000 |
| 300 | 0.861 ± 0.173 | 0.133 ± 0.067 |
| 500 | 0.944 ± 0.048 | 0.111 ± 0.038 |
| 750 | 0.917 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.944 ± 0.048 | 0.044 ± 0.038 |
| Best mass | 0.889 ± 0.048 | 0.044 ± 0.038 |
| 1000 (Final) | 0.917 ± 0.000 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.863 ± 0.034 | 0.089 ± 0.102 |
| 100 | 0.902 ± 0.068 | 0.044 ± 0.038 |
| 150 | 0.922 ± 0.034 | 0.133 ± 0.000 |
| 300 | 0.882 ± 0.059 | 0.133 ± 0.067 |
| 500 | 0.882 ± 0.059 | 0.111 ± 0.038 |
| 750 | 0.902 ± 0.034 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.922 ± 0.034 | 0.044 ± 0.038 |
| Best mass | 0.902 ± 0.034 | 0.044 ± 0.038 |
| 1000 (Final) | 0.922 ± 0.034 | 0.044 ± 0.038 |

### >30,000 px² (n=32)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.990 ± 0.018 | 0.089 ± 0.102 |
| 100 | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 150 | 1.000 ± 0.000 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 1.000 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.990 ± 0.018 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.990 ± 0.018 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 1.000 ± 0.000 | 0.044 ± 0.038 |

## Benign, 4 log-sized area bins

### 0–3,000 px² (n=16)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.167 ± 0.036 | 0.089 ± 0.102 |
| 100 | 0.229 ± 0.095 | 0.044 ± 0.038 |
| 150 | 0.250 ± 0.108 | 0.133 ± 0.000 |
| 300 | 0.375 ± 0.062 | 0.133 ± 0.067 |
| 500 | 0.396 ± 0.130 | 0.111 ± 0.038 |
| 750 | 0.458 ± 0.036 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.500 ± 0.062 | 0.044 ± 0.038 |
| Best mass | 0.500 ± 0.062 | 0.044 ± 0.038 |
| 1000 (Final) | 0.500 ± 0.062 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=5)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.467 ± 0.115 | 0.089 ± 0.102 |
| 100 | 0.400 ± 0.200 | 0.044 ± 0.038 |
| 150 | 0.467 ± 0.115 | 0.133 ± 0.000 |
| 300 | 0.467 ± 0.115 | 0.133 ± 0.067 |
| 500 | 0.533 ± 0.231 | 0.111 ± 0.038 |
| 750 | 0.400 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.600 ± 0.200 | 0.044 ± 0.038 |
| Best mass | 0.533 ± 0.231 | 0.044 ± 0.038 |
| 1000 (Final) | 0.467 ± 0.115 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=6)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.722 ± 0.192 | 0.089 ± 0.102 |
| 100 | 0.778 ± 0.096 | 0.044 ± 0.038 |
| 150 | 0.889 ± 0.096 | 0.133 ± 0.000 |
| 300 | 0.833 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.833 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.833 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.889 ± 0.096 | 0.044 ± 0.038 |
| Best mass | 0.889 ± 0.096 | 0.044 ± 0.038 |
| 1000 (Final) | 0.833 ± 0.000 | 0.044 ± 0.038 |

### >30,000 px² (n=3)
**Predetermined saved checkpoints**
| Checkpoint | Centroid Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.889 ± 0.192 | 0.089 ± 0.102 |
| 100 | 0.889 ± 0.192 | 0.044 ± 0.038 |
| 150 | 1.000 ± 0.000 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 1.000 ± 0.000 | 0.111 ± 0.038 |
| 750 | 1.000 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.889 ± 0.192 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.889 ± 0.192 | 0.044 ± 0.038 |


`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
