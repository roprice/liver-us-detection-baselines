# Relative noise-floor sweep

The pipeline drops predicted mass components smaller than a noise floor to remove speckle. Because image dimensions vary (both within this dataset and across external validation sets), a fixed pixel floor is inconsistent. This sweep replaces it with a *relative* floor expressed as a fraction of each image's area, and measures how many mass-present cases that are detected under the original fixed floor become missed under each candidate.

Three detection criteria are checked: triage (any retained mass), centroid 0.5, and overlap IoU > 0.2. The loss is counted against the fixed 100 px baseline flags already in the predictions dataset.

The AUL-average column converts each fraction to pixels using the weighted mean image area of the external AUL set (~341,420 px).

| Fraction | AUL average (px) | Triage lost | Centroid 0.5 lost | Overlap 0.2 lost |
|------:|------:|------:|------:|------:|
| 0.00005 | 17 | 0 | 0 | 0 |
| 0.00010 | 34 | 0 | 0 | 0 |
| 0.00020 | 68 | 0 | 0 | 0 |
| 0.00030 | 102 | 0 | 0 | 0 |
| 0.00040 | 136 | 9 | 0 | 0 |
| 0.00050 | 170 | 9 | 0 | 0 |
