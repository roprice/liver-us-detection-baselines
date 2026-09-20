# Malignant detection vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints. No error bars (single seed).

Values are malignant case-level recall (fraction of malignant cases detected) under seven criteria: triage, three overlap IoU tiers, and three centroid tolerance tiers.

| Epoch | Triage | Overlap (IoU>0.0) | Overlap (IoU>0.2) | Overlap (IoU>0.5) | Centroid (1.0xD) | Centroid (0.5xD) | Centroid (0.25xD) |
|------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 50 | 0.9077 | 0.8923 | 0.8000 | 0.6308 | 0.8923 | 0.8615 | 0.7231 |
| 100 | 0.9385 | 0.9231 | 0.9077 | 0.7385 | 0.9231 | 0.9077 | 0.8308 |
| 150 | 0.9846 | 0.9692 | 0.9231 | 0.7538 | 0.9692 | 0.9385 | 0.8308 |
| 300 | 0.9846 | 0.9692 | 0.9538 | 0.8000 | 0.9692 | 0.9538 | 0.9077 |
| 500 | 0.9538 | 0.9231 | 0.8923 | 0.7692 | 0.9231 | 0.9077 | 0.8154 |
| 750 | 0.9846 | 0.9692 | 0.9385 | 0.8769 | 0.9692 | 0.9538 | 0.8923 |
