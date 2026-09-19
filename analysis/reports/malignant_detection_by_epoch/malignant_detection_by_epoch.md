# Malignant detection vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints. No error bars (single seed).

Values are malignant case-level recall (fraction of malignant cases detected) under five detection criteria, ordered from most to least permissive.

| Epoch | Triage | Overlap (IoU > 0) | Centroid | IoU >= 0.2 | IoU >= 0.5 |
|------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 50 | 0.9077 | 0.8923 | 0.8615 | 0.8000 | 0.6154 |
| 100 | 0.9385 | 0.9231 | 0.9077 | 0.9077 | 0.7231 |
| 150 | 0.9846 | 0.9692 | 0.9385 | 0.9231 | 0.7231 |
| 300 | 0.9846 | 0.9692 | 0.9538 | 0.9538 | 0.8000 |
| 500 | 0.9538 | 0.9231 | 0.9077 | 0.8769 | 0.7692 |
| 750 | 0.9846 | 0.9692 | 0.9538 | 0.9385 | 0.8615 |
