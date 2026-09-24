# Pooled malignant triage detection by saved milestone epoch

Malignant triage recall is pooled within each seed across the `milestones_pilot` test cohort and the `milestones_pilot_vals` fold-0 validation cohort. A malignant case is detected if the model predicts any retained mass. Values are mean ± sample standard deviation across the three seeds; noise floor 0.03% of image area.

| Epoch | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |
|------:|--------:|--------:|--------:|----------:|
| 150 | 139/143 | 138/143 | 133/143 | 0.956 ± 0.022 |
| 300 | 141/143 | 134/143 | 138/143 | 0.963 ± 0.025 |
| 750 | 141/143 | 138/143 | 138/143 | 0.972 ± 0.012 |

*Each seed pools 65 malignant test cases and 78 malignant fold-0 validation cases (143 total).*
