# False positives vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints.

A false positive is a Normal (mass-free) case the model flags as containing a mass. Every image is one patient with at most one mass, so this is a single case-level signal.

No error bars (single seed).

Normal test cases: 15.

| Epoch | FP rate | False positives | Seed 42 |
|------:|--------:|----------------:|--------:|
| 50 | 0.0000 | 0 | 0/15 |
| 100 | 0.0000 | 0 | 0/15 |
| 150 | 0.1333 | 2 | 2/15 |
| 300 | 0.2000 | 3 | 3/15 |
| 500 | 0.1333 | 2 | 2/15 |
| 750 | 0.0667 | 1 | 1/15 |
