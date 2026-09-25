# Relative noise-floor sweep

The pipeline drops predicted mass components smaller than a noise floor to remove speckle. Because image dimensions vary (both within this dataset and across external validation sets), a fixed pixel floor is inconsistent. This sweep replaces it with a *relative* floor expressed as a fraction of each image's area, and measures how many mass-present cases that are detected under the selected relative floor become missed under each candidate.

Seven detection criteria are checked: triage (any retained mass), centroid distances of 0.25, 0.5, and 1.0 equivalent diameters, and overlap IoU thresholds of >0, >0.2, and >0.5. Each is compared with the selected relative-floor baseline flags already in the predictions dataset.

The AUL-average column converts each fraction to pixels using the weighted mean image area of the external AUL set (~341,420 px).

| Fraction | AUL average (px) | Triage | Centroid 0.25 | Centroid 0.5 | Centroid 1.0 | Overlap >0 | Overlap >0.2 | Overlap >0.5 |
|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 0.00005 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00010 | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00020 | 68 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00030 | 102 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.00040 | 136 | 19 | 4 | 16 | 17 | 12 | 0 | 0 |
| 0.00050 | 170 | 35 | 8 | 23 | 27 | 17 | 0 | 0 |

*All numeric detection columns are counts of predictions lost relative to the selected 0.03%-of-image-area baseline.*
