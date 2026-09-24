# Checkpoint: final (aggregated across seeds, mean ± std)

Dataset snapshot: `e9cee1981abb`. Seeds: 42, 43, 44.

False positive rate is the fraction of normal test cases with any raw predicted mass pixels (no component-size filtering), reported as mean ± sample SD across seeds. It uses all normal cases in every table, regardless of the mass-case filter.

## Mass Dice across all mass-present cases

For each training-size and seed run, the mass Dice columns include all mass-present cases, including empty and non-overlapping raw predicted mass masks. Liver Dice is calculated over all 110 test images. Values are mean ± sample SD across seeds.

| Size | Liver Dice | Malignant Dice | Benign Dice | Combined Mass Dice | False Positive Rate |
|-----:|-----------:|---------------:|------------:|-------------------:|--------------------:|
| 5 | 0.729 ± 0.007 | 0.145 ± 0.004 | 0.087 ± 0.007 | 0.127 ± 0.005 | 0.867 ± 0.000 |
| 10 | 0.724 ± 0.006 | 0.259 ± 0.004 | 0.116 ± 0.014 | 0.214 ± 0.002 | 0.711 ± 0.077 |
| 20 | 0.801 ± 0.009 | 0.215 ± 0.024 | 0.085 ± 0.003 | 0.174 ± 0.016 | 0.333 ± 0.067 |
| 40 | 0.837 ± 0.002 | 0.378 ± 0.025 | 0.124 ± 0.012 | 0.297 ± 0.020 | 0.422 ± 0.077 |
| 80 | 0.847 ± 0.003 | 0.519 ± 0.013 | 0.192 ± 0.016 | 0.416 ± 0.012 | 0.244 ± 0.038 |
| 160 | 0.871 ± 0.005 | 0.621 ± 0.014 | 0.286 ± 0.011 | 0.515 ± 0.013 | 0.356 ± 0.077 |
| 320 | 0.890 ± 0.004 | 0.714 ± 0.015 | 0.339 ± 0.015 | 0.596 ± 0.013 | 0.222 ± 0.077 |
| 625 | 0.900 ± 0.001 | 0.769 ± 0.008 | 0.382 ± 0.023 | 0.647 ± 0.005 | 0.089 ± 0.038 |

## Segmentation Dice by mass type and size

At the 625-image training size, Dice is calculated per mass case from raw hard predictions, averaged within each pathology for each seed, then summarized across seeds. Area is the ground-truth mass area.

| Class | n | Dice | Median area (px²) | Area range (px²) |
|:------|--:|-----:|------------------:|-----------------:|
| benign | 30 | 0.382 ± 0.023 | 2,643 | 398–81,560 |
| malignant | 65 | 0.769 ± 0.008 | 27,624 | 823–194,214 |

## Mass Dice excluding empty raw mass predictions

For each training-size and seed run, the mass Dice columns exclude mass-present cases whose raw predicted mass mask contains no mass pixels. Liver Dice remains calculated over all 110 test images. Values are mean ± sample SD across seeds.

| Size | Liver Dice (all cases) | Malignant Dice | Benign Dice | Combined Mass Dice | False Positive Rate |
|-----:|-----------:|---------------:|------------:|-------------------:|--------------------:|
| 5 | 0.729 ± 0.007 | 0.146 ± 0.005 | 0.095 ± 0.010 | 0.131 ± 0.007 | 0.867 ± 0.000 |
| 10 | 0.724 ± 0.006 | 0.273 ± 0.012 | 0.147 ± 0.009 | 0.238 ± 0.008 | 0.711 ± 0.077 |
| 20 | 0.801 ± 0.009 | 0.244 ± 0.027 | 0.135 ± 0.030 | 0.214 ± 0.009 | 0.333 ± 0.067 |
| 40 | 0.837 ± 0.002 | 0.400 ± 0.017 | 0.199 ± 0.010 | 0.353 ± 0.012 | 0.422 ± 0.077 |
| 80 | 0.847 ± 0.003 | 0.556 ± 0.014 | 0.288 ± 0.020 | 0.489 ± 0.011 | 0.244 ± 0.038 |
| 160 | 0.871 ± 0.005 | 0.644 ± 0.009 | 0.424 ± 0.038 | 0.590 ± 0.019 | 0.356 ± 0.077 |
| 320 | 0.890 ± 0.004 | 0.733 ± 0.022 | 0.451 ± 0.034 | 0.658 ± 0.002 | 0.222 ± 0.077 |
| 625 | 0.900 ± 0.001 | 0.785 ± 0.014 | 0.456 ± 0.060 | 0.690 ± 0.013 | 0.089 ± 0.038 |

## Mass Dice with overlapping raw mass predictions

For each training-size and seed run, the mass Dice columns include only mass-present cases whose raw predicted mass mask spatially overlaps the ground-truth mass. Liver Dice remains calculated over all 110 test images. Values are mean ± sample SD across seeds.

| Size | Liver Dice (all cases) | Malignant Dice | Benign Dice | Combined Mass Dice | False Positive Rate |
|-----:|-----------:|---------------:|------------:|-------------------:|--------------------:|
| 5 | 0.729 ± 0.007 | 0.216 ± 0.014 | 0.246 ± 0.023 | 0.222 ± 0.010 | 0.867 ± 0.000 |
| 10 | 0.724 ± 0.006 | 0.316 ± 0.012 | 0.389 ± 0.058 | 0.326 ± 0.013 | 0.711 ± 0.077 |
| 20 | 0.801 ± 0.009 | 0.303 ± 0.024 | 0.303 ± 0.063 | 0.302 ± 0.010 | 0.333 ± 0.067 |
| 40 | 0.837 ± 0.002 | 0.423 ± 0.021 | 0.449 ± 0.037 | 0.426 ± 0.021 | 0.422 ± 0.077 |
| 80 | 0.847 ± 0.003 | 0.585 ± 0.013 | 0.411 ± 0.029 | 0.551 ± 0.004 | 0.244 ± 0.038 |
| 160 | 0.871 ± 0.005 | 0.680 ± 0.005 | 0.527 ± 0.035 | 0.647 ± 0.012 | 0.356 ± 0.077 |
| 320 | 0.890 ± 0.004 | 0.749 ± 0.024 | 0.538 ± 0.034 | 0.699 ± 0.010 | 0.222 ± 0.077 |
| 625 | 0.900 ± 0.001 | 0.802 ± 0.010 | 0.585 ± 0.067 | 0.749 ± 0.012 | 0.089 ± 0.038 |
