# Case-level overlap-based detection (IoU>0.5) by saved milestone epoch and mass size

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, detection = IoU > 0.5, noise floor 0.03% of image area.

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
| 50 | 0.100 ± 0.000 | 0.089 ± 0.102 |
| 100 | 0.117 ± 0.076 | 0.044 ± 0.038 |
| 150 | 0.167 ± 0.076 | 0.133 ± 0.000 |
| 300 | 0.183 ± 0.144 | 0.133 ± 0.067 |
| 500 | 0.250 ± 0.132 | 0.111 ± 0.038 |
| 750 | 0.367 ± 0.029 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.400 ± 0.087 | 0.044 ± 0.038 |
| Best mass | 0.367 ± 0.029 | 0.044 ± 0.038 |
| 1000 (Final) | 0.367 ± 0.029 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.471 ± 0.102 | 0.089 ± 0.102 |
| 100 | 0.588 ± 0.156 | 0.044 ± 0.038 |
| 150 | 0.686 ± 0.034 | 0.133 ± 0.000 |
| 300 | 0.627 ± 0.090 | 0.133 ± 0.067 |
| 500 | 0.588 ± 0.059 | 0.111 ± 0.038 |
| 750 | 0.647 ± 0.059 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.647 ± 0.059 | 0.044 ± 0.038 |
| Best mass | 0.667 ± 0.034 | 0.044 ± 0.038 |
| 1000 (Final) | 0.608 ± 0.068 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=23)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.609 ± 0.043 | 0.089 ± 0.102 |
| 100 | 0.667 ± 0.133 | 0.044 ± 0.038 |
| 150 | 0.696 ± 0.000 | 0.133 ± 0.000 |
| 300 | 0.739 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.739 ± 0.115 | 0.111 ± 0.038 |
| 750 | 0.725 ± 0.066 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.754 ± 0.100 | 0.044 ± 0.038 |
| Best mass | 0.739 ± 0.115 | 0.044 ± 0.038 |
| 1000 (Final) | 0.725 ± 0.050 | 0.044 ± 0.038 |

### >30,000 px² (n=35)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.829 ± 0.057 | 0.089 ± 0.102 |
| 100 | 0.819 ± 0.044 | 0.044 ± 0.038 |
| 150 | 0.810 ± 0.072 | 0.133 ± 0.000 |
| 300 | 0.857 ± 0.029 | 0.133 ± 0.067 |
| 500 | 0.876 ± 0.044 | 0.111 ± 0.038 |
| 750 | 0.905 ± 0.016 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.867 ± 0.016 | 0.044 ± 0.038 |
| Best mass | 0.857 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.876 ± 0.016 | 0.044 ± 0.038 |

## Malignant, 4 log-sized area bins

### 0–3,000 px² (n=4)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.083 ± 0.144 | 0.089 ± 0.102 |
| 100 | 0.000 ± 0.000 | 0.044 ± 0.038 |
| 150 | 0.083 ± 0.144 | 0.133 ± 0.000 |
| 300 | 0.083 ± 0.144 | 0.133 ± 0.067 |
| 500 | 0.167 ± 0.144 | 0.111 ± 0.038 |
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
| 50 | 0.556 ± 0.127 | 0.089 ± 0.102 |
| 100 | 0.750 ± 0.144 | 0.044 ± 0.038 |
| 150 | 0.806 ± 0.048 | 0.133 ± 0.000 |
| 300 | 0.806 ± 0.127 | 0.133 ± 0.067 |
| 500 | 0.750 ± 0.083 | 0.111 ± 0.038 |
| 750 | 0.778 ± 0.096 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.806 ± 0.048 | 0.044 ± 0.038 |
| Best mass | 0.806 ± 0.048 | 0.044 ± 0.038 |
| 1000 (Final) | 0.750 ± 0.144 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.627 ± 0.034 | 0.089 ± 0.102 |
| 100 | 0.706 ± 0.156 | 0.044 ± 0.038 |
| 150 | 0.765 ± 0.000 | 0.133 ± 0.000 |
| 300 | 0.804 ± 0.034 | 0.133 ± 0.067 |
| 500 | 0.745 ± 0.122 | 0.111 ± 0.038 |
| 750 | 0.765 ± 0.102 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.804 ± 0.122 | 0.044 ± 0.038 |
| Best mass | 0.784 ± 0.136 | 0.044 ± 0.038 |
| 1000 (Final) | 0.784 ± 0.090 | 0.044 ± 0.038 |

### >30,000 px² (n=32)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.865 ± 0.048 | 0.089 ± 0.102 |
| 100 | 0.844 ± 0.062 | 0.044 ± 0.038 |
| 150 | 0.823 ± 0.079 | 0.133 ± 0.000 |
| 300 | 0.875 ± 0.031 | 0.133 ± 0.067 |
| 500 | 0.896 ± 0.048 | 0.111 ± 0.038 |
| 750 | 0.927 ± 0.018 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.885 ± 0.018 | 0.044 ± 0.038 |
| Best mass | 0.875 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.896 ± 0.018 | 0.044 ± 0.038 |

## Benign, 4 log-sized area bins

### 0–3,000 px² (n=16)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.104 ± 0.036 | 0.089 ± 0.102 |
| 100 | 0.146 ± 0.095 | 0.044 ± 0.038 |
| 150 | 0.188 ± 0.062 | 0.133 ± 0.000 |
| 300 | 0.208 ± 0.144 | 0.133 ± 0.067 |
| 500 | 0.271 ± 0.130 | 0.111 ± 0.038 |
| 750 | 0.375 ± 0.062 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.396 ± 0.095 | 0.044 ± 0.038 |
| Best mass | 0.354 ± 0.036 | 0.044 ± 0.038 |
| 1000 (Final) | 0.354 ± 0.036 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=5)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.267 ± 0.115 | 0.089 ± 0.102 |
| 100 | 0.200 ± 0.200 | 0.044 ± 0.038 |
| 150 | 0.400 ± 0.000 | 0.133 ± 0.000 |
| 300 | 0.200 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.200 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.333 ± 0.115 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.267 ± 0.115 | 0.044 ± 0.038 |
| Best mass | 0.333 ± 0.115 | 0.044 ± 0.038 |
| 1000 (Final) | 0.267 ± 0.115 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=6)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.556 ± 0.096 | 0.089 ± 0.102 |
| 100 | 0.556 ± 0.096 | 0.044 ± 0.038 |
| 150 | 0.500 ± 0.000 | 0.133 ± 0.000 |
| 300 | 0.556 ± 0.096 | 0.133 ± 0.067 |
| 500 | 0.722 ± 0.192 | 0.111 ± 0.038 |
| 750 | 0.611 ± 0.192 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.611 ± 0.096 | 0.044 ± 0.038 |
| Best mass | 0.611 ± 0.096 | 0.044 ± 0.038 |
| 1000 (Final) | 0.556 ± 0.096 | 0.044 ± 0.038 |

### >30,000 px² (n=3)
**Predetermined saved checkpoints**
| Checkpoint | Overlap Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.444 ± 0.192 | 0.089 ± 0.102 |
| 100 | 0.556 ± 0.192 | 0.044 ± 0.038 |
| 150 | 0.667 ± 0.000 | 0.133 ± 0.000 |
| 300 | 0.667 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.667 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.667 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.667 ± 0.000 | 0.044 ± 0.038 |
| Best mass | 0.667 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.667 ± 0.000 | 0.044 ± 0.038 |


`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
