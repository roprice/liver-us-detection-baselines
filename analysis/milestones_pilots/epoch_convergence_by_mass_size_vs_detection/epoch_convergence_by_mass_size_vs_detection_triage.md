# Case-level triage-based detection by saved milestone epoch and mass size

Three preliminary milestones test runs: seeds 42/43/44, 625 images each, triage = any retained mass (no overlap required), noise floor 0.03% of image area.

A mass-present case is detected if the model predicts any retained mass anywhere on the image.

Mass into 4 bins according to size and triage detection is measured against each by epoch
- 0–3,000 px² (n=20)
- >3,000–10,000 px² (n=17)
- >10,000–30,000 px² (n=23)
- >30,000 px² (n=35)

Each table reports case-level false positives computed on normal cases only. A normal case with any prediction is a false alarm.

Values are mean ± standard deviation across the three seeds.

## Both pathologies, 4 log-sized area bins

### 0–3,000 px² (n=20)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.400 ± 0.304 | 0.089 ± 0.102 |
| 100 | 0.400 ± 0.180 | 0.044 ± 0.038 |
| 150 | 0.417 ± 0.231 | 0.133 ± 0.000 |
| 300 | 0.567 ± 0.104 | 0.133 ± 0.067 |
| 500 | 0.550 ± 0.180 | 0.111 ± 0.038 |
| 750 | 0.600 ± 0.087 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.667 ± 0.153 | 0.044 ± 0.038 |
| Best mass | 0.650 ± 0.132 | 0.044 ± 0.038 |
| 1000 (Final) | 0.600 ± 0.100 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.804 ± 0.122 | 0.089 ± 0.102 |
| 100 | 0.784 ± 0.068 | 0.044 ± 0.038 |
| 150 | 0.902 ± 0.068 | 0.133 ± 0.000 |
| 300 | 0.863 ± 0.034 | 0.133 ± 0.067 |
| 500 | 0.902 ± 0.090 | 0.111 ± 0.038 |
| 750 | 0.863 ± 0.034 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.902 ± 0.034 | 0.044 ± 0.038 |
| Best mass | 0.902 ± 0.034 | 0.044 ± 0.038 |
| 1000 (Final) | 0.863 ± 0.034 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=23)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.928 ± 0.050 | 0.089 ± 0.102 |
| 100 | 0.942 ± 0.025 | 0.044 ± 0.038 |
| 150 | 0.971 ± 0.025 | 0.133 ± 0.000 |
| 300 | 0.957 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.942 ± 0.025 | 0.111 ± 0.038 |
| 750 | 0.957 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.971 ± 0.025 | 0.044 ± 0.038 |
| Best mass | 0.971 ± 0.025 | 0.044 ± 0.038 |
| 1000 (Final) | 0.957 ± 0.000 | 0.044 ± 0.038 |

### >30,000 px² (n=35)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.990 ± 0.016 | 0.089 ± 0.102 |
| 100 | 0.990 ± 0.016 | 0.044 ± 0.038 |
| 150 | 1.000 ± 0.000 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 1.000 ± 0.000 | 0.111 ± 0.038 |
| 750 | 1.000 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.990 ± 0.016 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.990 ± 0.016 | 0.044 ± 0.038 |

## Malignant, 4 log-sized area bins

### 0–3,000 px² (n=4)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.083 ± 0.144 | 0.089 ± 0.102 |
| 100 | 0.333 ± 0.144 | 0.044 ± 0.038 |
| 150 | 0.417 ± 0.382 | 0.133 ± 0.000 |
| 300 | 0.417 ± 0.382 | 0.133 ± 0.067 |
| 500 | 0.417 ± 0.144 | 0.111 ± 0.038 |
| 750 | 0.583 ± 0.144 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.583 ± 0.144 | 0.044 ± 0.038 |
| Best mass | 0.583 ± 0.144 | 0.044 ± 0.038 |
| 1000 (Final) | 0.583 ± 0.144 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=12)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.889 ± 0.096 | 0.089 ± 0.102 |
| 100 | 0.889 ± 0.048 | 0.044 ± 0.038 |
| 150 | 0.972 ± 0.048 | 0.133 ± 0.000 |
| 300 | 0.972 ± 0.048 | 0.133 ± 0.067 |
| 500 | 1.000 ± 0.000 | 0.111 ± 0.038 |
| 750 | 0.972 ± 0.048 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 1.000 ± 0.000 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 0.972 ± 0.048 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=17)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 1.000 ± 0.000 | 0.089 ± 0.102 |
| 100 | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 150 | 1.000 ± 0.000 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.980 ± 0.034 | 0.111 ± 0.038 |
| 750 | 1.000 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 1.000 ± 0.000 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 1.000 ± 0.000 | 0.044 ± 0.038 |

### >30,000 px² (n=32)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 1.000 ± 0.000 | 0.089 ± 0.102 |
| 100 | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 150 | 1.000 ± 0.000 | 0.133 ± 0.000 |
| 300 | 1.000 ± 0.000 | 0.133 ± 0.067 |
| 500 | 1.000 ± 0.000 | 0.111 ± 0.038 |
| 750 | 1.000 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 1.000 ± 0.000 | 0.044 ± 0.038 |
| Best mass | 1.000 ± 0.000 | 0.044 ± 0.038 |
| 1000 (Final) | 1.000 ± 0.000 | 0.044 ± 0.038 |

## Benign, 4 log-sized area bins

### 0–3,000 px² (n=16)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.479 ± 0.344 | 0.089 ± 0.102 |
| 100 | 0.417 ± 0.219 | 0.044 ± 0.038 |
| 150 | 0.417 ± 0.201 | 0.133 ± 0.000 |
| 300 | 0.604 ± 0.072 | 0.133 ± 0.067 |
| 500 | 0.583 ± 0.191 | 0.111 ± 0.038 |
| 750 | 0.604 ± 0.095 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.688 ± 0.188 | 0.044 ± 0.038 |
| Best mass | 0.667 ± 0.157 | 0.044 ± 0.038 |
| 1000 (Final) | 0.604 ± 0.095 | 0.044 ± 0.038 |

### >3,000–10,000 px² (n=5)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| 50 | 0.600 ± 0.200 | 0.089 ± 0.102 |
| 100 | 0.533 ± 0.115 | 0.044 ± 0.038 |
| 150 | 0.733 ± 0.115 | 0.133 ± 0.000 |
| 300 | 0.600 ± 0.000 | 0.133 ± 0.067 |
| 500 | 0.667 ± 0.306 | 0.111 ± 0.038 |
| 750 | 0.600 ± 0.000 | 0.044 ± 0.038 |

**Selected and final checkpoints**
| Checkpoint | Detection | Normal cases FP rate |
|------:|----------:|-------------------:|
| Best | 0.667 ± 0.115 | 0.044 ± 0.038 |
| Best mass | 0.667 ± 0.115 | 0.044 ± 0.038 |
| 1000 (Final) | 0.600 ± 0.000 | 0.044 ± 0.038 |

### >10,000–30,000 px² (n=6)
**Predetermined saved checkpoints**
| Checkpoint | Triage Detection | Normal cases FP rate |
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
| Checkpoint | Triage Detection | Normal cases FP rate |
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
