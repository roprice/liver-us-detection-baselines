# False positives vs. epoch

Single preliminary milestones run: seed 42, 625 images, joint-selected checkpoints.

A false positive is a Normal (mass-free) case the model flags as containing a mass. Every image is one patient with at most one mass, so this is a single case-level signal.

No error bars (single seed).

Normal test cases: 15.

| Epoch | False positives | FP rate |
|------:|----------------:|--------:|
| 50 | 0 | 0.0000 |
| 100 | 0 | 0.0000 |
| 150 | 2 | 0.1333 |
| 300 | 3 | 0.2000 |
| 500 | 2 | 0.1333 |
| 750 | 1 | 0.0667 |
