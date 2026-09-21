# Segmentation quality vs. epoch

Preliminary milestones run: seeds 42/43/44, 625 images each.

Values are mean ± standard deviation across the three seeds.

**Predetermined saved checkpoints**
| Checkpoint | Liver Dice | Malignant mass Dice | Benign mass Dice | Combined mass Dice |
|------:|-----------:|--------------------:|----------------:|-------------------:|
| 50 | 0.8936 ± 0.0028 | 0.6838 ± 0.0348 | 0.2806 ± 0.0594 | 0.5565 ± 0.0386 |
| 100 | 0.8962 ± 0.0042 | 0.7176 ± 0.0105 | 0.3073 ± 0.0298 | 0.5880 ± 0.0072 |
| 150 | 0.9003 ± 0.0016 | 0.7448 ± 0.0067 | 0.3391 ± 0.0116 | 0.6167 ± 0.0023 |
| 300 | 0.8952 ± 0.0050 | 0.7534 ± 0.0454 | 0.3669 ± 0.0274 | 0.6313 ± 0.0321 |
| 500 | 0.8947 ± 0.0091 | 0.7483 ± 0.0325 | 0.3640 ± 0.0834 | 0.6269 ± 0.0437 |
| 750 | 0.9033 ± 0.0018 | 0.7696 ± 0.0191 | 0.4350 ± 0.0168 | 0.6639 ± 0.0126 |

**Selected and final checkpoints**
| Checkpoint | Liver Dice | Malignant mass Dice | Benign mass Dice | Combined mass Dice |
|------:|-----------:|--------------------:|----------------:|-------------------:|
| Best | 0.9016 ± 0.0054 | 0.7732 ± 0.0200 | 0.4602 ± 0.0173 | 0.6743 ± 0.0183 |
| Best mass | 0.9049 ± 0.0005 | 0.7724 ± 0.0207 | 0.4525 ± 0.0202 | 0.6714 ± 0.0205 |
| 1000 (Final) | 0.9018 ± 0.0033 | 0.7669 ± 0.0179 | 0.4333 ± 0.0076 | 0.6615 ± 0.0115 |

Final malignant Dice: 0.7696. Tail-2-epoch slope: +0.00009/epoch.

`Best` epoch was 838 for seed 42, 695 for seed 43, and 999 for seed 44.
`Best mass` epoch was 838 for seed 42, 708 for seed 43, and 999 for seed 44.
