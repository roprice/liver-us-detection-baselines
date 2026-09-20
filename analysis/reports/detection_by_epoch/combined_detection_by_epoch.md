# Combined detection vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints. No error bars (single seed).

Values are combined case-level recall (fraction of all mass-present cases detected) under seven criteria: triage, three overlap IoU tiers, and three centroid tolerance tiers.

| Epoch | Triage | Overlap (IoU>0.0) | Overlap (IoU>0.2) | Overlap (IoU>0.5) | Centroid (1.0xD) | Centroid (0.5xD) | Centroid (0.25xD) |
|------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 50 | 0.7368 | 0.7158 | 0.6211 | 0.5053 | 0.7158 | 0.6947 | 0.6000 |
| 100 | 0.7789 | 0.7474 | 0.7368 | 0.5789 | 0.7474 | 0.7368 | 0.6737 |
| 150 | 0.8947 | 0.8421 | 0.7579 | 0.6316 | 0.8421 | 0.8105 | 0.7158 |
| 300 | 0.8842 | 0.8316 | 0.8000 | 0.6316 | 0.8316 | 0.8211 | 0.7579 |
| 500 | 0.8211 | 0.7789 | 0.7158 | 0.6105 | 0.7789 | 0.7684 | 0.6632 |
| 750 | 0.8947 | 0.8421 | 0.8211 | 0.7263 | 0.8526 | 0.8316 | 0.7684 |
