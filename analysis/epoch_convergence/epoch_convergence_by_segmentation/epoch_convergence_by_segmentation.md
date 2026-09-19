# Segmentation quality vs. epoch

Preliminary milestones run: seed 42, 625 images, joint-selected checkpoints.

No error bars (single seed).

| Epoch | Liver Dice | Malignant mass Dice | Benign mass Dice | Combined mass Dice |
|------:|-----------:|--------------------:|----------------:|-------------------:|
| 50 | 0.8912 | 0.6509 | 0.2143 | 0.5130 |
| 100 | 0.9007 | 0.7265 | 0.2736 | 0.5835 |
| 150 | 0.8994 | 0.7513 | 0.3344 | 0.6196 |
| 300 | 0.8986 | 0.7846 | 0.3483 | 0.6468 |
| 500 | 0.8860 | 0.7250 | 0.2695 | 0.5811 |
| 750 | 0.9014 | 0.7912 | 0.4326 | 0.6779 |
| best | 0.9044 | 0.7964 | 0.4758 | 0.6952 |
| best mass | 0.9044 | 0.7964 | 0.4758 | 0.6952 |
| final | 0.9018 | 0.7868 | 0.4276 | 0.6734 |

Final malignant Dice: 0.7912. Tail-2-epoch slope: +0.00026/epoch.
