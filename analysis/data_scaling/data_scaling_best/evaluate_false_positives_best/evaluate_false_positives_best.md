# False positives vs. training size

Data-scaling test runs: seeds 42/43/44, best checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625, noise floor 0.03% of image area. Seeds vary initialization only; training subsets are nested. Uncertainty: 95% percentile bootstrap over the 15 normal test cases (10,000 replicates, RNG seed 0), applied to the seed-mean FP rate. Case resamples are shared across seeds and sizes.

Dataset snapshot: `d3c90caf06db`.

A false positive is a Normal (mass-free) case the model flags as containing any retained mass. Every image is one patient with at most one mass, so this is a single case-level signal, not a count of predicted components.

Normal test cases: 15.

| Training size | Seed 42 | Seed 43 | Seed 44 | Mean FP rate ± SD | 95% CI |
|--------------:|:-------:|:-------:|:-------:|------------------:|-------:|
| 5 | 15/15 | 13/15 | 9/15 | 0.822 ± 0.204 | 0.689–0.933 |
| 10 | 9/15 | 9/15 | 11/15 | 0.644 ± 0.077 | 0.422–0.844 |
| 20 | 2/15 | 6/15 | 7/15 | 0.333 ± 0.176 | 0.156–0.533 |
| 40 | 3/15 | 4/15 | 5/15 | 0.267 ± 0.067 | 0.111–0.444 |
| 80 | 2/15 | 3/15 | 3/15 | 0.178 ± 0.038 | 0.022–0.356 |
| 160 | 6/15 | 4/15 | 4/15 | 0.311 ± 0.077 | 0.111–0.556 |
| 320 | 2/15 | 3/15 | 4/15 | 0.200 ± 0.067 | 0.044–0.400 |
| 625 | 1/15 | 1/15 | 1/15 | 0.067 ± 0.000 | 0.000–0.178 |
