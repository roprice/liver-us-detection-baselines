# Benign detection vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints. No error bars (single seed).

Values are benign case-level recall (fraction of benign cases detected) under seven criteria: triage, three overlap IoU tiers, and three centroid tolerance tiers.

| Epoch | Triage | Overlap (IoU>0.0) | Overlap (IoU>0.2) | Overlap (IoU>0.5) | Centroid (1.0xD) | Centroid (0.5xD) | Centroid (0.25xD) |
|------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 50 | 0.3667 | 0.3333 | 0.2333 | 0.2333 | 0.3333 | 0.3333 | 0.3000 |
| 100 | 0.4333 | 0.3667 | 0.3667 | 0.2333 | 0.3667 | 0.3667 | 0.3333 |
| 150 | 0.7000 | 0.5667 | 0.4000 | 0.3667 | 0.5667 | 0.5333 | 0.4667 |
| 300 | 0.6667 | 0.5333 | 0.4667 | 0.2667 | 0.5333 | 0.5333 | 0.4333 |
| 500 | 0.5333 | 0.4667 | 0.3333 | 0.2667 | 0.4667 | 0.4667 | 0.3333 |
| 750 | 0.7000 | 0.5667 | 0.5667 | 0.4000 | 0.6000 | 0.5667 | 0.5000 |
