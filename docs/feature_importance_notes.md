# Feature Importance Notes

## Overview
- Extracted coefficient-based importances from the tuned logistic regression model (`models/logistic_model.pkl`) using feature names stored in `models/feature_names.txt`.
- Positive coefficients increase predicted default risk (higher log-odds); negative coefficients decrease risk relative to the base category of each one-hot encoded group.
- Top 10 features (by absolute coefficient) are saved to `results/top_features.csv`, and a bar chart is saved to `results/figures/feature_importance.png`.

## Key Drivers Observed
- `credit_type_EQUI` has an exceptionally large positive coefficient (~37.7), indicating much higher default odds than the baseline credit bureau category.
- Other credit bureau categories (`credit_type_EXP`, `credit_type_CIB`, `credit_type_CRIF`) carry large negative coefficients (≈ -11.6), suggesting substantially lower risk relative to the baseline.
- Collateral and structure signals (`construction_type_mh`, `security_type_Indriect`, `secured_by_land`) meaningfully increase risk, while `secured_by_home` and `construction_type_sb` reduce it.
- The `lump_sum_payment_lpsm` option adds risk, implying borrowers selecting lump-sum payments may be more likely to default.
- Magnitudes should be interpreted carefully: they reflect separation against the reference category after preprocessing/one-hot encoding rather than raw numeric scale.

## Generated Artifacts
- `results/top_features.csv`: Ranked table with coefficients, absolute coefficients, effect direction, and odds ratios.
- `results/figures/feature_importance.png`: Bar chart (positive in red, negative in blue) for the top 10 features.
- `notebooks/03_feature_analysis.ipynb`: Notebook showing the calculations and interpretation bullets.
