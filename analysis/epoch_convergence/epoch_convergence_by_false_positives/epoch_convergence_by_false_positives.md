# False positives vs. epoch

Preliminary milestones run: seeds 42/43/44, 625 images each.

A false positive is a Normal (mass-free) case the model flags as containing a mass. Every image is one patient with at most one mass, so this is a single case-level signal.

Values are mean ± standard deviation across the three seeds.

Normal test cases: 15.

| Epoch | FP rate | False positives |
|------:|--------:|----------------:|
| 50 | 0.0889 ± 0.1018 | 1.33 ± 1.53 |
| 100 | 0.0444 ± 0.0385 | 0.67 ± 0.58 |
| 150 | 0.1333 ± 0.0000 | 2.00 ± 0.00 |
| 300 | 0.1333 ± 0.0667 | 2.00 ± 1.00 |
| 500 | 0.1111 ± 0.0385 | 1.67 ± 0.58 |
| 750 | 0.0444 ± 0.0385 | 0.67 ± 0.58 |
