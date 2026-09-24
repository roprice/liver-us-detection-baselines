# False positives vs. training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625, noise floor 0.03% of image area. Values are mean ± sample SD across seeds.

Dataset snapshot: `e9cee1981abb`.

A false positive is a Normal (mass-free) case the model flags as containing any retained mass. Every image is one patient with at most one mass, so this is a single case-level signal, not a count of predicted components.

Values are mean ± sample standard deviation across the three seeds.

Normal test cases: 15.

| Training size | Seed 42 | Seed 43 | Seed 44 | Mean FP rate | Mean FP count |
|--------------:|:-------:|:-------:|:-------:|-------------:|--------------:|
| 5 | 12/15 | 12/15 | 11/15 | 0.7778 ± 0.0385 | 11.67 ± 0.58 |
| 10 | 9/15 | 8/15 | 11/15 | 0.6222 ± 0.1018 | 9.33 ± 1.53 |
| 20 | 4/15 | 4/15 | 3/15 | 0.2444 ± 0.0385 | 3.67 ± 0.58 |
| 40 | 6/15 | 6/15 | 5/15 | 0.3778 ± 0.0385 | 5.67 ± 0.58 |
| 80 | 2/15 | 3/15 | 3/15 | 0.1778 ± 0.0385 | 2.67 ± 0.58 |
| 160 | 5/15 | 4/15 | 4/15 | 0.2889 ± 0.0385 | 4.33 ± 0.58 |
| 320 | 2/15 | 4/15 | 4/15 | 0.2222 ± 0.0770 | 3.33 ± 1.15 |
| 625 | 1/15 | 1/15 | 1/15 | 0.0667 ± 0.0000 | 1.00 ± 0.00 |
