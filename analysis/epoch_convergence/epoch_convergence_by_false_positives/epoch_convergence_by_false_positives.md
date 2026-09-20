# False positives vs. epoch

Preliminary milestones run: seeds 42/43/44, 625 images each.

A false positive is a Normal (mass-free) case the model flags as containing a mass. Every image is one patient with at most one mass, so this is a single case-level signal.

Values are mean ± standard deviation across the three seeds.

Normal test cases: 15.

| Epoch | Seed 42 | Seed 43 | Seed 44 | Mean FP rate | Mean FP count |
|------:|:-------:|:-------:|:-------:|-------------:|--------------:|
| 50 | 0/15 | 1/15 | 3/15 | 0.0889 ± 0.1018 | 1.33 ± 1.53 |
| 100 | 0/15 | 1/15 | 1/15 | 0.0444 ± 0.0385 | 0.67 ± 0.58 |
| 150 | 2/15 | 2/15 | 2/15 | 0.1333 ± 0.0000 | 2.00 ± 0.00 |
| 300 | 3/15 | 1/15 | 2/15 | 0.1333 ± 0.0667 | 2.00 ± 1.00 |
| 500 | 2/15 | 1/15 | 2/15 | 0.1111 ± 0.0385 | 1.67 ± 0.58 |
| 750 | 1/15 | 0/15 | 1/15 | 0.0444 ± 0.0385 | 0.67 ± 0.58 |
