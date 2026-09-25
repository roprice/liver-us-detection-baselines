# False positives vs. training size

Data-scaling test runs: seeds 42/43/44, final checkpoints, training sizes 5, 10, 20, 40, 80, 160, 320, 625, noise floor 0.03% of image area. Seeds vary initialization only; training subsets are nested. Uncertainty: 95% percentile bootstrap over the 15 normal test cases (10,000 replicates, RNG seed 0), applied to the seed-mean FP rate. Case resamples are shared across seeds and sizes.

Dataset snapshot: `de6aa8358f5e`.

## Retained-mask normal-case FP rate

A false positive is a Normal (mass-free) case the model flags as containing any retained mass. Every image is one patient with at most one mass, so this is a single case-level signal, not a count of predicted components.

Normal test cases: 15.

| Training size | Seed 42 | Seed 43 | Seed 44 | Mean FP rate ± SD | 95% CI |
|--------------:|:-------:|:-------:|:-------:|------------------:|-------:|
| 5 | 12/15 | 12/15 | 11/15 | 0.778 ± 0.038 | 0.600–0.933 |
| 10 | 9/15 | 8/15 | 11/15 | 0.622 ± 0.102 | 0.400–0.822 |
| 20 | 4/15 | 4/15 | 3/15 | 0.244 ± 0.038 | 0.067–0.444 |
| 40 | 6/15 | 6/15 | 5/15 | 0.378 ± 0.038 | 0.200–0.578 |
| 80 | 2/15 | 3/15 | 3/15 | 0.178 ± 0.038 | 0.022–0.356 |
| 160 | 5/15 | 4/15 | 4/15 | 0.289 ± 0.038 | 0.089–0.533 |
| 320 | 2/15 | 4/15 | 4/15 | 0.222 ± 0.077 | 0.067–0.422 |
| 625 | 1/15 | 1/15 | 1/15 | 0.067 ± 0.000 | 0.000–0.178 |

## Exact binomial intervals per seed

95% exact binomial (Clopper-Pearson) interval for each seed's k/15. Conservative by construction; recommended at low counts, where the bootstrap interval tends to be too narrow.

| Training size | Seed 42 | Seed 43 | Seed 44 |
|--------------:|:-------:|:-------:|:-------:|
| 5 | 12/15: 0.519–0.957 | 12/15: 0.519–0.957 | 11/15: 0.449–0.922 |
| 10 | 9/15: 0.323–0.837 | 8/15: 0.266–0.787 | 11/15: 0.449–0.922 |
| 20 | 4/15: 0.078–0.551 | 4/15: 0.078–0.551 | 3/15: 0.043–0.481 |
| 40 | 6/15: 0.163–0.677 | 6/15: 0.163–0.677 | 5/15: 0.118–0.616 |
| 80 | 2/15: 0.017–0.405 | 3/15: 0.043–0.481 | 3/15: 0.043–0.481 |
| 160 | 5/15: 0.118–0.616 | 4/15: 0.078–0.551 | 4/15: 0.078–0.551 |
| 320 | 2/15: 0.017–0.405 | 4/15: 0.078–0.551 | 4/15: 0.078–0.551 |
| 625 | 1/15: 0.002–0.319 | 1/15: 0.002–0.319 | 1/15: 0.002–0.319 |

## No-overlap FP regions per image

Retained predicted region with no pixel overlap with the ground-truth mass; every retained region on a normal image counts. Reported as FP regions per image across all 110 test images. Seed columns give total FP regions.

95% percentile bootstrap over the 110 test images (10,000 replicates, RNG seed 0), applied to the seed-mean FP regions per image.

| Training size | Seed 42 | Seed 43 | Seed 44 | No-overlap FP regions per image ± SD | 95% CI | Per normal image | Per mass image |
|--------------:|:-------:|:-------:|:-------:|--------------------------:|-------:|-----------------:|---------------:|
| 5 | 163 | 181 | 173 | 1.567 ± 0.082 | 1.333–1.815 | 1.444 | 1.586 |
| 10 | 148 | 118 | 137 | 1.221 ± 0.138 | 0.985–1.479 | 1.022 | 1.253 |
| 20 | 51 | 64 | 75 | 0.576 ± 0.109 | 0.445–0.721 | 0.356 | 0.611 |
| 40 | 42 | 50 | 51 | 0.433 ± 0.045 | 0.330–0.545 | 0.556 | 0.414 |
| 80 | 22 | 29 | 24 | 0.227 ± 0.033 | 0.161–0.300 | 0.200 | 0.232 |
| 160 | 25 | 18 | 27 | 0.212 ± 0.043 | 0.142–0.288 | 0.311 | 0.196 |
| 320 | 19 | 14 | 18 | 0.155 ± 0.024 | 0.091–0.224 | 0.289 | 0.133 |
| 625 | 12 | 16 | 13 | 0.124 ± 0.019 | 0.070–0.188 | 0.067 | 0.133 |

## Criterion-specific FP regions per image

Retained component count minus one if the criterion detects the mass on a mass image, otherwise all retained components; on normal images all retained components count. Averaged over all test images.

95% percentile bootstrap over the 110 test images (10,000 replicates, RNG seed 0), applied to the seed-mean FP regions per image. Image resamples are shared across seeds and sizes.

Each cell shows the seed mean ± sample SD; [95% image-bootstrap CI]. All test images are included.

### Overlap detection

| Training size | IoU > 0 | IoU > 0.2 | IoU > 0.5 |
|--------------:|---------------------------:|---------------------------:|---------------------------:|
| 5 | 1.703 ± 0.098 [1.455–1.967] | 2.006 ± 0.084 [1.733–2.294] | 2.148 ± 0.081 [1.888–2.427] |
| 10 | 1.482 ± 0.140 [1.215–1.767] | 1.773 ± 0.155 [1.479–2.085] | 1.945 ± 0.145 [1.648–2.264] |
| 20 | 0.679 ± 0.136 [0.545–0.824] | 0.948 ± 0.141 [0.785–1.124] | 1.100 ± 0.167 [0.936–1.276] |
| 40 | 0.527 ± 0.040 [0.412–0.652] | 0.764 ± 0.040 [0.612–0.921] | 0.967 ± 0.052 [0.806–1.133] |
| 80 | 0.361 ± 0.043 [0.273–0.455] | 0.503 ± 0.045 [0.388–0.627] | 0.703 ± 0.034 [0.570–0.842] |
| 160 | 0.312 ± 0.064 [0.224–0.403] | 0.385 ± 0.068 [0.282–0.491] | 0.582 ± 0.045 [0.461–0.709] |
| 320 | 0.164 ± 0.024 [0.100–0.233] | 0.215 ± 0.029 [0.142–0.291] | 0.400 ± 0.024 [0.303–0.497] |
| 625 | 0.139 ± 0.029 [0.082–0.203] | 0.170 ± 0.019 [0.106–0.239] | 0.300 ± 0.027 [0.218–0.388] |

### Centroid detection

| Training size | d_eq ≤ 0.25 | d_eq ≤ 0.50 | d_eq ≤ 1.00 |
|--------------:|---------------------------:|---------------------------:|---------------------------:|
| 5 | 2.036 ± 0.105 [1.785–2.309] | 1.861 ± 0.095 [1.615–2.127] | 1.664 ± 0.098 [1.418–1.927] |
| 10 | 1.836 ± 0.146 [1.558–2.130] | 1.627 ± 0.159 [1.358–1.918] | 1.470 ± 0.132 [1.203–1.758] |
| 20 | 0.997 ± 0.116 [0.836–1.167] | 0.791 ± 0.110 [0.645–0.948] | 0.658 ± 0.124 [0.524–0.803] |
| 40 | 0.842 ± 0.045 [0.691–1.003] | 0.609 ± 0.055 [0.488–0.736] | 0.512 ± 0.042 [0.400–0.636] |
| 80 | 0.588 ± 0.053 [0.476–0.706] | 0.436 ± 0.048 [0.339–0.539] | 0.370 ± 0.043 [0.282–0.464] |
| 160 | 0.473 ± 0.069 [0.364–0.588] | 0.330 ± 0.064 [0.239–0.421] | 0.315 ± 0.059 [0.227–0.406] |
| 320 | 0.333 ± 0.029 [0.242–0.424] | 0.206 ± 0.037 [0.136–0.282] | 0.167 ± 0.026 [0.103–0.236] |
| 625 | 0.221 ± 0.014 [0.148–0.300] | 0.152 ± 0.026 [0.091–0.221] | 0.139 ± 0.029 [0.082–0.203] |
