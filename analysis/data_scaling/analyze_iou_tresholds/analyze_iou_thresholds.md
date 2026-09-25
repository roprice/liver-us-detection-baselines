# IoU-threshold sensitivity

Training size: 625 images; experiment: data_scaling; evaluation split: test; checkpoint: final. The predicted-component noise floor is fixed at 0.03% of image area (pixel cutoff = int(fraction × image height × image width)). Values are mean ± sample standard deviation across 3 training seeds (42, 43, 44).

Source: `analysis/predictions_dataset/predictions_dataset.csv` via the canonical loader. Dataset snapshot: `de6aa8358f5e`.

Detection requires max_retained_component_iou > threshold, matching the canonical dataset's strict comparison. At IoU 0.00 this requires any positive overlap. Unlike the previous analysis, equality at a positive threshold is not counted as detection.

## IoU thresholds in prior studies and this study

Detection is combined benign/malignant recall on this study's cases. Literature descriptions were verified against the papers' full texts (September 2026). Attributions carried over from the previous analysis that could not be confirmed (Bellver et al. at IoU 0.3, Bilic et al. at IoU 0.5, Kour & Adilakshmi at IoU 0.9) were removed; Vorontsov et al., which reports detection at IoU greater than 0, 0.25, and 0.5, anchors both the 0.0 and 0.5 rows. Wei et al. and Tiyarattanachai et al. compute IoU on bounding boxes, whereas Vorontsov et al. and this study use mask-based IoU.

| IoU | Detection | Paper | Usage |
|----:|----------:|:------|:------|
| 0.0 | 0.839 ± 0.006 | [Vorontsov et al. (Radiology: AI, 2019)](https://doi.org/10.1148/ryai.2019180014) | Any predicted lesion overlap |
| 0.1 | 0.814 ± 0.006 | [Wei et al. (Nature Communications, 2024)](https://doi.org/10.1038/s41467-024-51260-6) | Low-overlap detection before classification |
| 0.2 | 0.804 ± 0.012 | [Tiyarattanachai et al. (PLOS ONE, 2021)](https://doi.org/10.1371/journal.pone.0252882) | Ultrasound lesion localization |
| 0.5 | 0.653 ± 0.011 | [Vorontsov et al. (Radiology: AI, 2019)](https://doi.org/10.1148/ryai.2019180014) | High-overlap detection; favors PPV over sensitivity |

## Malignant overlap-recall by ground-truth mass-area quartile

Q1 contains the smallest malignant masses and Q4 the largest. Cases are sorted by area, then image ID to break ties, and split as evenly as possible; earlier quartiles receive any extra cases.

| Quartile | n | Ground-truth mass area range (px²) |
|:---------|--:|------------------------------------:|
| Q1 | 17 | 823–11,532 |
| Q2 | 16 | 14,074–27,624 |
| Q3 | 16 | 30,263–48,852 |
| Q4 | 16 | 49,653–194,214 |

| IoU threshold | Q1 recall | Q2 recall | Q3 recall | Q4 recall |
|--------------:|----------:|----------:|----------:|----------:|
| 0.00 | 0.882 ± 0.059 | 0.938 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 0.05 | 0.863 ± 0.034 | 0.938 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 0.10 | 0.843 ± 0.034 | 0.938 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 0.25 | 0.843 ± 0.034 | 0.896 ± 0.036 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 0.50 | 0.725 ± 0.034 | 0.771 ± 0.036 | 0.875 ± 0.062 | 0.854 ± 0.095 |
| 0.75 | 0.373 ± 0.034 | 0.396 ± 0.036 | 0.562 ± 0.062 | 0.625 ± 0.000 |
| 0.95 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Benign overlap-recall by ground-truth mass-area quartile

Q1 contains the smallest benign masses and Q4 the largest. Cases are sorted by area, then image ID to break ties, and split as evenly as possible; earlier quartiles receive any extra cases.

| Quartile | n | Ground-truth mass area range (px²) |
|:---------|--:|------------------------------------:|
| Q1 | 8 | 398–1,230 |
| Q2 | 8 | 1,533–2,792 |
| Q3 | 7 | 3,443–11,368 |
| Q4 | 7 | 11,845–81,560 |

| IoU threshold | Q1 recall | Q2 recall | Q3 recall | Q4 recall |
|--------------:|----------:|----------:|----------:|----------:|
| 0.00 | 0.333 ± 0.072 | 0.500 ± 0.000 | 0.619 ± 0.082 | 0.952 ± 0.082 |
| 0.05 | 0.333 ± 0.072 | 0.500 ± 0.000 | 0.524 ± 0.082 | 0.952 ± 0.082 |
| 0.10 | 0.333 ± 0.072 | 0.500 ± 0.000 | 0.524 ± 0.082 | 0.810 ± 0.082 |
| 0.25 | 0.292 ± 0.072 | 0.458 ± 0.072 | 0.476 ± 0.165 | 0.714 ± 0.000 |
| 0.50 | 0.042 ± 0.072 | 0.292 ± 0.072 | 0.286 ± 0.143 | 0.714 ± 0.000 |
| 0.75 | 0.000 ± 0.000 | 0.042 ± 0.072 | 0.000 ± 0.000 | 0.429 ± 0.000 |
| 0.95 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Overall overlap recall by pathology

Recall across all cases in each pathology group, independent of the pathology-specific quartiles above.

| IoU threshold | Benign recall | Malignant recall |
|--------------:|--------------:|-----------------:|
| 0.00 | 0.589 ± 0.019 | 0.954 ± 0.015 |
| 0.05 | 0.567 ± 0.033 | 0.949 ± 0.009 |
| 0.10 | 0.533 ± 0.033 | 0.944 ± 0.009 |
| 0.25 | 0.478 ± 0.069 | 0.933 ± 0.009 |
| 0.50 | 0.322 ± 0.019 | 0.805 ± 0.024 |
| 0.75 | 0.111 ± 0.019 | 0.487 ± 0.024 |
| 0.95 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Triage flagging at the fixed noise floor

Triage flagging does not use an IoU threshold: it requires any retained predicted mass component. It is reported once rather than repeated across the IoU sweep. The normal-case value is the false-positive rate.

| Case group | n | Triage flagging / false-positive rate |
|:-----------|--:|--------------------------------------:|
| Benign | 30 | 0.778 ± 0.084 |
| Malignant | 65 | 0.969 ± 0.015 |
| Normal (false positive) | 15 | 0.067 ± 0.000 |
