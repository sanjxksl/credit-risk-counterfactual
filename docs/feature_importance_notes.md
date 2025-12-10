# Feature Importance Analysis

## Methodology

### Logistic Regression Baseline (Interpretable Model)
- Extracted coefficient-based importances from the tuned logistic regression model ([models/logistic_model.pkl](../models/logistic_model.pkl))
- Positive coefficients increase predicted default risk (higher log-odds); negative coefficients decrease risk
- Top 10 features saved to [results/top_features.csv](../results/top_features.csv)
- Visualization: [results/figures/feature_importance.png](../results/figures/feature_importance.png)

### MLP Feature Importance (Production Model)
- Extracted model-agnostic permutation importance from the MLP model ([models/mlp_model.pth](../models/mlp_model.pth))
- Method: Randomly shuffle each feature and measure AUC-ROC degradation (n_repeats=5)
- Higher importance = greater performance drop when feature is randomized
- Baseline Test AUC: 88.79%
- Analysis notebook: [notebooks/mlp_feature_importance.ipynb](../notebooks/mlp_feature_importance.ipynb)

## Key Findings

### Top 10 Most Important Features (MLP Permutation Importance)

**Continuous Features** (highest impact):
1. **LTV (Loan-to-Value Ratio)**: Most critical feature - removing it causes largest AUC drop
2. **DTIR (Debt-to-Income Ratio)**: Second most important - directly measures repayment capacity
3. **Credit Score**: Third most critical - strong predictor despite being immutable
4. **Property Value**: Significant impact on risk assessment
5. **Loan Amount**: Directly affects exposure and default risk

**Categorical Features** (structural risk indicators):
- **Credit Bureau Type** (credit_type_EQUI, credit_type_EXP, etc.): Large variance across bureaus
- **Construction Type** (construction_type_mh vs construction_type_sb): Mobile homes carry higher risk
- **Security Type** (security_type_Indriect vs direct): Indirect security increases risk
- **Secured By** (secured_by_land vs secured_by_home): Land-secured loans riskier than home-secured
- **Lump Sum Payment Option**: Borrowers selecting lump-sum repayment show higher default rates

### Comparison: Logistic vs MLP Importance

**Agreement** (both models rank these highly):
- LTV, DTIR, Credit Score consistently top-ranked
- Credit bureau type (EQUI has ~37.7 coefficient in logistic model)
- Construction and security type features

**MLP Captures Non-Linear Relationships**:
- MLP learns interaction effects (e.g., LTV × DTIR interactions)
- Better handles threshold effects (e.g., credit score <600 vs >750)
- Permutation importance reflects these complex patterns

### Actionable Insights

**For Applicants** (aligned with counterfactual analysis):
1. **Increase down payment** → Reduce LTV (58.7% of counterfactuals modified this)
2. **Lower debt-to-income ratio** → Pay down existing debt (56.5% modification rate)
3. **Optimize loan structure** → Adjust term/property value/loan amount (54.3% modification rate)

**For Lenders**:
- Focus underwriting on LTV + DTIR combination (highest predictive power)
- Credit bureau type matters: EQUI-reported scores associated with 37.7x higher odds
- Construction type signals risk: Mobile homes require stricter criteria

### Statistical Notes

**Logistic Regression Coefficients**:
- `credit_type_EQUI`: +37.7 (exceptionally high risk vs baseline)
- `credit_type_EXP/CIB/CRIF`: -11.6 (substantially lower risk)
- `construction_type_mh`: Positive coefficient (increases risk)
- `secured_by_home`: Negative coefficient (decreases risk vs land)
- Interpret relative to reference category after one-hot encoding

**MLP Permutation Importance**:
- Repeated 5 times per feature for statistical stability
- Reports mean importance ± standard deviation
- Feature interdependencies captured (e.g., property_value influences ltv calculation)

## Generated Artifacts

### Logistic Regression Analysis
- [results/top_features.csv](../results/top_features.csv): Ranked coefficients with odds ratios
- [results/figures/feature_importance.png](../results/figures/feature_importance.png): Bar chart (red=risk increase, blue=risk decrease)
- [notebooks/03_feature_analysis.ipynb](../notebooks/03_feature_analysis.ipynb): Full analysis with interpretations

### MLP Analysis
- [notebooks/mlp_feature_importance.ipynb](../notebooks/mlp_feature_importance.ipynb): Permutation importance calculations
- Findings section with top features ranked by AUC impact
- Statistical stability metrics (mean ± std across 5 repeats)
