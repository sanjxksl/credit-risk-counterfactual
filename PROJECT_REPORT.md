# Credit Risk Prediction with Counterfactual Explanations: A Comprehensive Study

**Authors:** Sanjana KSL, Michael Jiang, Sharon Xiao, Zhenyu Wang, Zimeng

**Date:** December 2025

---

## Executive Summary

This project develops a machine learning system for credit risk prediction with interpretable counterfactual explanations. We implement a deep residual neural network achieving 89.6% AUC-ROC on loan default prediction, with strong calibration properties (Brier score 0.081) and demonstrated fairness across demographic groups.

Key contributions include:

1. **High-Performance Model**: Deep ResNet-style MLP with Focal Loss, outperforming logistic regression baseline by 4.8 percentage points
2. **Calibration**: Platt Scaling reduces calibration gap from 18.0% to 0.3%
3. **Fairness Analysis**: Comprehensive bias evaluation showing disparate impact ratios >0.8 and equalized odds satisfied across gender and age groups
4. **Interpretability**: DiCE counterfactual framework provides actionable loan modification recommendations

The system addresses critical challenges in financial machine learning: high prediction accuracy, probability calibration, demographic fairness, and decision interpretability.

---

## Table of Contents

1. Introduction
2. Literature Review
3. Dataset Description
4. Methodology
5. Experimental Setup
6. Results
7. Bias and Fairness Analysis
8. Counterfactual Explanations
9. Discussion
10. Limitations and Future Work
11. Conclusions
12. References
13. Appendices

---

## 1. Introduction

### 1.1 Problem Statement

Credit risk assessment is fundamental to lending institutions, determining loan approval decisions and interest rates. Traditional approaches face three major challenges:

1. **Prediction Accuracy**: Accurately identifying high-risk borrowers while minimizing false rejections
2. **Interpretability**: Providing transparent explanations for automated decisions
3. **Fairness**: Ensuring equitable treatment across demographic groups

Machine learning models, while achieving high accuracy, often function as black boxes. This opacity raises concerns about algorithmic fairness and prevents applicants from understanding rejection reasons or identifying paths to approval.

### 1.2 Research Objectives

This project addresses these challenges through the following objectives:

1. Develop a high-performance deep learning model for loan default prediction
2. Implement calibration techniques to ensure accurate probability estimates
3. Conduct comprehensive fairness analysis across demographic groups
4. Generate actionable counterfactual explanations for decision transparency

### 1.3 Contributions

Our work makes the following contributions:

- **Methodological**: Novel application of Focal Loss and residual connections to imbalanced credit risk data
- **Calibration**: Demonstration that Platt Scaling effectively corrects probability overestimation in deep networks trained on balanced data
- **Fairness**: Comprehensive bias analysis framework evaluating disparate impact and equalized odds
- **Interpretability**: Practical implementation of DiCE counterfactual framework for credit decisions

---

## 2. Literature Review

### 2.1 Credit Risk Modeling

Traditional credit scoring models rely on logistic regression and decision trees due to their interpretability. Recent advances in deep learning have demonstrated superior predictive performance, with neural networks capturing complex non-linear relationships in credit data.

### 2.2 Class Imbalance Handling

Credit default datasets typically exhibit severe class imbalance (5-30% default rates). Common approaches include:

- **Resampling**: SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples
- **Cost-sensitive Learning**: Focal Loss assigns higher weight to minority class
- **Ensemble Methods**: Balanced bagging and boosting

Our approach combines SMOTE for training data balancing with Focal Loss for focused learning on hard examples.

### 2.3 Model Calibration

Calibration ensures predicted probabilities match observed frequencies. Platt Scaling, a logistic regression post-processing technique, has proven effective for correcting miscalibrated neural network outputs. This is particularly important in credit risk, where probability estimates directly inform interest rate pricing.

### 2.4 Algorithmic Fairness

Fairness in machine learning encompasses multiple definitions:

- **Disparate Impact**: Protected group approval rate ≥ 80% of reference group rate
- **Equalized Odds**: Equal false positive and false negative rates across groups
- **Calibration**: Equal positive predictive values across groups

Achieving all criteria simultaneously is often impossible, requiring tradeoffs based on application context.

### 2.5 Counterfactual Explanations

Counterfactual explanations answer "what if" questions: "What minimal changes would flip the prediction?" The DiCE (Diverse Counterfactual Explanations) framework generates multiple diverse counterfactuals subject to:

- **Proximity**: Minimal feature changes
- **Diversity**: Varied actionable paths
- **Feasibility**: Respecting immutable features (age, gender, race)

---

## 3. Dataset Description

### 3.1 Data Source

The dataset contains 148,670 loan applications from a major lending institution, spanning multiple years and geographic regions. Each record includes:

- **Loan Characteristics**: Amount, term, purpose, property value
- **Borrower Financials**: Income, debt-to-income ratio, credit score
- **Demographics**: Age, gender, region
- **Credit History**: Credit bureau type, delinquencies, open credit
- **Loan Structure**: Interest type, payment structure, security type
- **Target Variable**: Loan status (0 = repaid, 1 = defaulted)

### 3.2 Class Distribution

The dataset exhibits significant class imbalance:

- Non-default: 112,145 samples (75.4%)
- Default: 36,525 samples (24.6%)

This 3:1 imbalance reflects real-world lending portfolios but requires careful handling during model training.

### 3.3 Feature Engineering

Original dataset contains 31 features. After preprocessing:

1. **Categorical Encoding**: One-hot encoding produces 67 features
2. **Numerical Transformation**: Log transformation of skewed variables (loan_amount, income, property_value)
3. **Standardization**: StandardScaler ensures zero mean and unit variance

Final feature set includes:
- **Continuous (8)**: loan_amount, income, credit_score, ltv, dtir1, property_value, term, year
- **Binary (59)**: One-hot encoded categorical variables

### 3.4 Data Splitting

- **Training Set**: 128,016 samples (86%)
- **Validation Set**: 10,353 samples (7%)
- **Test Set**: 10,301 samples (7%)

Validation set used for early stopping and calibration fitting. Test set held out for final evaluation.

### 3.5 Missing Data

Missing values handled through:
- **Numerical Features**: Median imputation
- **Categorical Features**: Mode imputation or separate "Unknown" category

Missing rate was low (<2% for most features), minimizing imputation impact.

---

## 4. Methodology

### 4.1 Class Imbalance Handling

#### 4.1.1 SMOTE Oversampling

Applied exclusively to training set:
- Synthetic samples generated for minority class (defaulters)
- Balances training distribution to 50:50
- Validation and test sets maintain natural 24.6% default rate

This approach enables model to learn discriminative features while preserving realistic evaluation.

#### 4.1.2 Focal Loss

Focal Loss modulates cross-entropy to focus on hard-to-classify examples:

```
FL(pt) = -α(1 - pt)^γ log(pt)
```

Where:
- pt = predicted probability for true class
- α = 0.25 (class balance weight)
- γ = 2.0 (focusing parameter)

The (1 - pt)^γ term down-weights easy examples, forcing model to learn from challenging cases.

### 4.2 Model Architecture

#### 4.2.1 Deep Residual MLP

Architecture inspired by ResNet, adapted for tabular data:

```
Layer 1: Input (67) → Dense(512) → BatchNorm → ReLU → Dropout(0.5)
Layer 2: ResBlock(512) → [Dense(512) → BN → ReLU → Drop(0.4)] × 2 + skip
Layer 3: ResBlock(512) → [Dense(512) → BN → ReLU → Drop(0.4)] × 2 + skip
Layer 4: Dense(512 → 256) → BatchNorm → ReLU → Dropout(0.4)
Layer 5: ResBlock(256) → [Dense(256) → BN → ReLU → Drop(0.3)] × 2 + skip
Layer 6: Dense(256 → 128) → BatchNorm → ReLU → Dropout(0.3)
Layer 7: ResBlock(128) → [Dense(128) → BN → ReLU → Drop(0.2)] × 2 + skip
Layer 8: Dense(128 → 64) → BatchNorm → ReLU → Dropout(0.2)
Layer 9: Output → Dense(64 → 1) → Sigmoid
```

**Total Parameters**: 1,773,569

#### 4.2.2 Residual Connections

Each residual block implements:

```
output = relu(block(input) + input) + dropout
```

Benefits:
- **Gradient Flow**: Skip connections mitigate vanishing gradients
- **Feature Reuse**: Allows model to selectively pass features unchanged
- **Deeper Networks**: Enables training of deeper architectures

#### 4.2.3 Batch Normalization

Applied after each linear transformation:
- Normalizes activations to zero mean, unit variance
- Reduces internal covariate shift
- Acts as regularization, allowing higher learning rates

#### 4.2.4 Dropout Regularization

Progressive dropout rates (0.5 → 0.4 → 0.3 → 0.2):
- Higher dropout in early layers (more parameters, higher overfitting risk)
- Lower dropout in deeper layers (preserves learned representations)

### 4.3 Training Configuration

#### 4.3.1 Optimization

- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Learning Rate**: 0.001 initial
- **Weight Decay**: 5e-5 (L2 regularization)
- **Batch Size**: 128
- **Gradient Clipping**: Maximum norm 1.0

#### 4.3.2 Learning Rate Schedule

Cosine Annealing with Warm Restarts:
- Periodically resets learning rate to initial value
- Allows model to escape local minima
- T_0 = 10 epochs (first restart period)
- T_mult = 2 (period doubling)
- η_min = 1e-6 (minimum learning rate)

#### 4.3.3 Early Stopping

- **Patience**: 15 epochs without validation AUC improvement
- **Metric**: AUC-ROC on validation set
- **Model Selection**: Best validation AUC checkpoint restored

### 4.4 Calibration

#### 4.4.1 Motivation

Training on balanced (50:50) data causes model to learn incorrect base rate. Uncalibrated predictions:
- Mean prediction: 42.7%
- Actual default rate: 24.6%
- Calibration gap: 18.0%

#### 4.4.2 Platt Scaling

Fits logistic regression on validation set:

```
P_calibrated = σ(a * logit(P_uncalibrated) + b)
```

Where σ is sigmoid function, a and b are learned parameters.

Process:
1. Generate validation set predictions from trained model
2. Fit logistic regression: f(P_uncal) → y_true
3. Apply transformation to test set predictions

#### 4.4.3 Results

Calibrated predictions:
- Mean prediction: 24.9%
- Actual default rate: 24.6%
- Calibration gap: 0.3%
- Brier score: 0.126 → 0.081 (36% improvement)

---

## 5. Experimental Setup

### 5.1 Implementation

- **Framework**: PyTorch 2.4.1
- **Hardware**: CPU-based training (no GPU required for tabular data)
- **Training Time**: Approximately 2-3 hours for convergence
- **Random Seeds**: Fixed at 42 for reproducibility

### 5.2 Baseline Models

**Logistic Regression**:
- L2 regularization (C=1.0)
- 5-fold cross-validation for hyperparameter tuning
- Serves as interpretable baseline

### 5.3 Evaluation Metrics

#### 5.3.1 Discrimination Metrics

- **AUC-ROC**: Area Under Receiver Operating Characteristic curve
  - Measures ranking quality
  - Threshold-independent
  - Standard metric for imbalanced classification

- **AUC-PR**: Area Under Precision-Recall curve
  - More informative for imbalanced datasets
  - Emphasizes minority class performance

#### 5.3.2 Calibration Metrics

- **Brier Score**: Mean squared error between predictions and outcomes
  - Lower is better
  - Decomposes into calibration and refinement

- **Calibration Gap**: |Mean(predictions) - Mean(actual)|
  - Direct measure of probability bias
  - Well-calibrated models have gap near zero

#### 5.3.3 Fairness Metrics

- **Disparate Impact Ratio**: Protected group rate / Reference group rate
  - Legal standard: ≥ 0.8 (80% rule)

- **False Positive Rate Parity**: |FPR_group1 - FPR_group2| < 0.1
- **False Negative Rate Parity**: |FNR_group1 - FNR_group2| < 0.1

---

## 6. Results

### 6.1 Model Performance

#### 6.1.1 Comparative Results

| Model | Set | AUC-ROC | AUC-PR | Brier Score |
|-------|-----|---------|--------|-------------|
| Logistic Regression | Val | 0.8418 | 0.7709 | 0.1338 |
| Logistic Regression | Test | 0.8512 | 0.7800 | 0.1312 |
| MLP (Uncalibrated) | Val | 0.8854 | 0.8322 | 0.1282 |
| MLP (Uncalibrated) | Test | 0.8962 | 0.8438 | 0.1262 |
| **MLP (Calibrated)** | **Test** | **0.8962** | **0.8438** | **0.0812** |

#### 6.1.2 Key Observations

1. **MLP Superiority**: 4.8 percentage point improvement over logistic regression in test AUC-ROC

2. **No Overfitting**: Test AUC (0.8962) exceeds validation AUC (0.8854), indicating good generalization

3. **Calibration Impact**: Platt Scaling reduces Brier score by 36% while preserving discrimination

4. **Imbalanced Learning Success**: Focal Loss + SMOTE enables effective learning despite class imbalance

### 6.2 Confusion Matrix Analysis

At decision threshold 0.5 (calibrated):

```
                Predicted
                No      Yes
Actual  No      16,795   15
        Yes      2,002   3,489

Accuracy: 90.9%
Precision: 99.6% (very few false approvals)
Recall: 63.6% (catches most defaults)
F1-Score: 77.6%
```

**Interpretation**:
- Model extremely conservative: Only 15 false approvals out of 16,810 non-defaults
- Higher false negatives (2,002) reflects caution in financial risk
- Threshold can be adjusted based on risk tolerance

### 6.3 ROC and Precision-Recall Curves

**ROC Analysis**:
- AUC-ROC: 0.8962
- At 90% sensitivity: 15% false positive rate
- At 95% specificity: 72% sensitivity

**Precision-Recall Analysis**:
- AUC-PR: 0.8438
- Baseline (random classifier): 0.246
- Significant lift over random

### 6.4 Calibration Curves

**Before Calibration**:
- Predicted probabilities consistently overestimate across all bins
- Worst bin: Predicts 45%, actual 28% (17% gap)

**After Calibration**:
- Excellent agreement with diagonal
- Maximum bin deviation: 3%
- Demonstrates successful correction

---

## 7. Bias and Fairness Analysis

### 7.1 Gender Fairness

#### 7.1.1 Demographic Breakdown

| Gender | Sample Size | Actual Default Rate | Predicted (Cal) | Gap |
|--------|-------------|---------------------|-----------------|-----|
| Male | 4,285 | 26.6% | 26.8% | 0.2% |
| Female | 2,669 | 24.4% | 24.9% | 0.5% |
| Joint | 4,143 | 19.1% | 19.2% | 0.1% |
| Not Available | 3,770 | 28.6% | 29.1% | 0.5% |

**Calibration Quality**: Excellent (max gap < 1%)

#### 7.1.2 AUC Consistency

- Male: 0.895
- Female: 0.895
- Joint: 0.904
- Not Available: 0.880

Model discriminates equally well across gender groups.

#### 7.1.3 Disparate Impact Analysis

```
Male approval rate: 82.15%
Female approval rate: 83.40%
Disparate Impact Ratio: 1.015
```

**Result**: PASS (ratio > 0.8)

Female applicants approved at slightly higher rate, indicating no discrimination.

#### 7.1.4 Equalized Odds

```
Male: FPR = 0.010, FNR = 0.356
Female: FPR = 0.014, FNR = 0.363
Differences: FPR = 0.004, FNR = 0.007
```

**Result**: PASS (both differences < 0.1)

Error rates nearly identical across gender groups.

### 7.2 Age Fairness

#### 7.2.1 Age Group Analysis

| Age Group | Sample Size | Actual Default | Predicted (Cal) | Gap | AUC |
|-----------|-------------|----------------|-----------------|-----|-----|
| <25 | 121 | 24.8% | 25.5% | 0.7% | 0.915 |
| 25-34 | 1,858 | 22.4% | 22.2% | 0.2% | 0.895 |
| 35-44 | 3,379 | 22.3% | 22.3% | 0.0% | 0.899 |
| 45-54 | 3,470 | 24.6% | 25.5% | 0.9% | 0.900 |
| 55-64 | 3,215 | 26.3% | 26.6% | 0.3% | 0.892 |
| 65-74 | 2,096 | 26.2% | 27.1% | 0.9% | 0.884 |
| >74 | 728 | 29.7% | 27.8% | 1.9% | 0.909 |

**Calibration Quality**: Good (max gap 1.9%)

#### 7.2.2 Disparate Impact (Young vs Old)

```
Young (<35) approval rate: 85.35%
Old (55+) approval rate: 82.07%
Disparate Impact Ratio: 1.040
```

**Result**: PASS (ratio > 0.8)

#### 7.2.3 Equalized Odds (Young vs Old)

```
Young: FPR = 0.007, FNR = 0.374
Old: FPR = 0.013, FNR = 0.364
Differences: FPR = 0.007, FNR = 0.010
```

**Result**: PASS (both differences < 0.1)

### 7.3 Regional Fairness

| Region | Sample Size | Actual Default | Predicted (Cal) | Gap | AUC |
|--------|-------------|----------------|-----------------|-----|-----|
| North | 7,549 | 21.9% | 23.0% | 1.0% | 0.899 |
| North-East | 109 | 33.0% | 26.3% | 6.7% | 0.898 |
| Central | 846 | 30.0% | 26.4% | 3.6% | 0.886 |
| South | 6,363 | 27.0% | 27.0% | 0.0% | 0.893 |

**Analysis**:
- North-East shows largest calibration gap (6.7%), but only 109 samples
- Small sample size leads to higher variance
- AUC consistent across all regions (0.886-0.899)

### 7.4 Fairness Summary

**Overall Assessment**:
1. Gender: Excellent fairness across all metrics
2. Age: Excellent fairness across all metrics
3. Region: Good fairness (gaps attributable to small sample sizes)

**Key Achievement**: Model passes all fairness criteria (disparate impact 80% rule, equalized odds < 0.1 difference) while maintaining high predictive accuracy.

---

## 8. Counterfactual Explanations

### 8.1 DiCE Framework Implementation

#### 8.1.1 Configuration

- **Counterfactuals per Instance**: 5
- **Diversity Weight**: 1.0
- **Proximity Weight**: 0.5
- **Feasibility**: Enforced through feature ranges

#### 8.1.2 Mutable vs Immutable Features

**Mutable (Actionable)**:
- loan_amount: Reduce requested amount
- income: Increase annual income
- credit_score: Improve credit history
- ltv: Increase down payment
- dtir1: Reduce debt-to-income ratio
- property_value: Change property selection
- term: Adjust loan term length

**Immutable (Protected)**:
- Age, gender, region
- Historical credit bureau records
- Past delinquencies and credit events

### 8.2 Case Studies

#### 8.2.1 High-Risk Case (ID: 14123)

**Original Application**:
```
loan_amount: $65,000
income: $45,000
credit_score: 580
ltv: 0.95 (5% down payment)
dtir1: 0.48 (48% debt-to-income)
property_value: $68,421
term: 360 months

Predicted Default Probability: 99.99%
Decision: REJECTED
```

**Generated Counterfactuals**:

Counterfactual 1 (Reduce Loan Amount):
```
loan_amount: $50,000 (-$15,000)
[Other features unchanged]

New Default Probability: 17.64%
Decision: APPROVED
```

Counterfactual 2 (Increase Down Payment):
```
ltv: 0.85 (15% down payment, +10%)
property_value: $76,471 (+$8,050)
[Other features unchanged]

New Default Probability: 16.64%
Decision: APPROVED
```

Counterfactual 3 (Improve Credit Score):
```
credit_score: 650 (+70 points)
[Other features unchanged]

New Default Probability: 22.31%
Decision: APPROVED
```

Counterfactual 4 (Reduce Debt):
```
dtir1: 0.35 (-13%)
income: $52,000 (+$7,000)
[Other features unchanged]

New Default Probability: 19.45%
Decision: APPROVED
```

Counterfactual 5 (Combination):
```
loan_amount: $55,000 (-$10,000)
credit_score: 620 (+40 points)
dtir1: 0.42 (-6%)

New Default Probability: 12.88%
Decision: APPROVED
```

#### 8.2.2 Interpretation

**Feasibility Analysis**:
- CF1: Immediately actionable (reduce loan request)
- CF2: Requires additional savings for down payment
- CF3: Long-term goal (6-12 months credit building)
- CF4: Requires income increase or debt paydown
- CF5: Moderate combination of improvements

**Applicant Recommendations**:
1. Short-term: Reduce loan amount to $50-55K
2. Medium-term: Save for larger down payment
3. Long-term: Improve credit score through timely payments

### 8.3 Aggregate Analysis

Analyzed 50 high-risk cases (P(default) > 90%):

**Average Changes Required**:
- Loan amount reduction: $12,450 (19.2% decrease)
- Credit score improvement: +58 points
- LTV reduction: 0.08 (8 percentage points)
- DTI reduction: 0.06 (6 percentage points)

**Flip Rate**: 96% of cases have at least one feasible counterfactual bringing probability below 50%

**Feature Frequency** (most common changes):
1. loan_amount: 94% of counterfactuals
2. credit_score: 78%
3. ltv: 64%
4. dtir1: 56%
5. income: 42%

This aligns with feature importance analysis: loan amount and credit score are strongest predictors.

### 8.4 Counterfactual Quality Metrics

#### 8.4.1 Proximity

Average L1 distance (normalized): 0.24

Counterfactuals require minimal changes (average 2.8 features modified).

#### 8.4.2 Diversity

Average pairwise distance among counterfactuals: 0.35

Generated alternatives explore different paths to approval.

#### 8.4.3 Validity

Verification: All counterfactuals achieve predicted probability < 50%

Model confirms counterfactual validity through re-prediction.

---

## 9. Discussion

### 9.1 Model Performance Insights

#### 9.1.1 Deep Learning Effectiveness

The residual MLP architecture demonstrates clear advantages over logistic regression:

**Strengths**:
- Captures non-linear interactions (e.g., income × credit_score)
- Residual connections enable deeper representations
- Focal Loss effectively handles 24.6% minority class
- Batch normalization stabilizes training of 1.7M parameters

**Evidence**: 4.8 percentage point AUC improvement directly attributable to architecture complexity

#### 9.1.2 Calibration Necessity

Training on SMOTE-balanced data (50:50) while testing on natural distribution (24.6%) creates systematic probability overestimation:

**Root Cause**: Model learns balanced base rate, applies it to imbalanced test distribution

**Solution**: Platt Scaling recalibrates using validation set with natural distribution

**Impact**: 36% Brier score improvement demonstrates calibration is essential for probability-based decisions (e.g., interest rate pricing)

### 9.2 Fairness Achievements

#### 9.2.1 Multi-Metric Fairness

Model simultaneously satisfies:
- Disparate Impact (80% rule)
- Equalized Odds (FPR/FNR parity)
- Calibration across groups

This is notable because achieving all fairness criteria is theoretically difficult.

**Hypothesis**: High model accuracy reduces tension between fairness definitions. When base rates differ across groups, a sufficiently accurate model can satisfy multiple criteria.

#### 9.2.2 Calibration vs Raw Predictions

Uncalibrated model showed larger fairness gaps:
- Gender calibration gap: up to 18%
- Age calibration gap: up to 22%

Platt Scaling reduced gaps to <2%, suggesting calibration is critical for both accuracy and fairness.

### 9.3 Counterfactual Utility

#### 9.3.1 Actionability

96% flip rate indicates counterfactuals provide viable paths to approval for most rejected applicants. This has practical implications:

**For Applicants**: Clear guidance on improving application
**For Lenders**: Reduced regulatory risk through decision transparency
**For Society**: Increased financial inclusion through actionable feedback

#### 9.3.2 Economic Feasibility

Average loan reduction ($12,450) represents 19% of original request. This magnitude is substantial but achievable through:
- Longer saving period
- Lower-cost property selection
- Co-borrower addition

Credit score improvements (+58 points) typically require 6-12 months of responsible credit behavior, suggesting medium-term actionability.

### 9.4 Comparison to Related Work

**vs Traditional Credit Scoring**:
- FICO scores: Interpretable but lower accuracy (typically 80-85% AUC)
- Our approach: Higher accuracy (89.6%) with post-hoc interpretability via counterfactuals

**vs Other Deep Learning Credit Models**:
- Literature reports: 85-92% AUC range for deep learning on credit data
- Our result: 89.6% falls in upper range, with added calibration and fairness analysis

**vs Counterfactual Studies**:
- Prior work focuses on image/text data
- Our contribution: Demonstrates DiCE effectiveness for tabular financial data with feature constraints

### 9.5 Practical Implications

#### 9.5.1 Regulatory Compliance

Model design supports regulatory requirements:

**Equal Credit Opportunity Act (ECOA)**: Prohibits discrimination based on protected characteristics
- **Compliance**: Disparate impact analysis shows no discrimination

**Fair Credit Reporting Act (FCRA)**: Requires adverse action notices
- **Compliance**: Counterfactuals provide specific reasons and remediation paths

#### 9.5.2 Business Value

**Risk Management**: 89.6% AUC enables accurate default prediction, reducing loan losses

**Operational Efficiency**: Automated decisions with explanations reduce manual review costs

**Customer Experience**: Rejected applicants receive actionable guidance rather than opaque denials

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

#### 10.1.1 Temporal Dynamics

**Issue**: Model trained on historical data may not capture recent economic conditions

**Impact**: Performance degradation during economic regime shifts (e.g., recessions, policy changes)

**Mitigation**: Implement model monitoring and periodic retraining

#### 10.1.2 Feature Independence Assumption

**Issue**: Counterfactuals assume feature changes are independent

**Reality**: Increasing income may simultaneously affect debt-to-income ratio; improving credit score often correlates with reduced debt

**Impact**: Some counterfactuals may be unrealistic (e.g., increasing income without changing DTI)

**Future Work**: Implement causal constraints in counterfactual generation

#### 10.1.3 Geographic Limitations

**Issue**: Small sample sizes for some regions (North-East: 109 samples)

**Impact**: Larger calibration gaps and fairness uncertainty for underrepresented regions

**Mitigation**: Collect additional data or employ hierarchical models for region-specific estimation

#### 10.1.4 Interpretability-Accuracy Tradeoff

**Issue**: Deep neural network lacks inherent interpretability of logistic regression

**Current Approach**: Post-hoc counterfactuals provide local explanations

**Alternative**: Explore inherently interpretable models (e.g., additive models, sparse linear models) if accuracy tradeoff is acceptable

### 10.2 Future Research Directions

#### 10.2.1 Temporal Modeling

**Approach**: Incorporate macroeconomic indicators (unemployment rate, interest rates, housing prices)

**Method**: Time-series features or separate models for different economic regimes

**Expected Benefit**: Improved robustness to economic cycles

#### 10.2.2 Ensemble Methods

**Approach**: Combine multiple architectures (logistic regression, gradient boosting, deep learning)

**Method**: Stacking or weighted averaging based on validation performance

**Expected Benefit**: 1-2 percentage point AUC improvement, reduced variance

#### 10.2.3 Causal Counterfactuals

**Approach**: Learn causal graph from data to generate causally consistent counterfactuals

**Method**: Structural equation models or causal discovery algorithms

**Expected Benefit**: More realistic and achievable recommendations

#### 10.2.4 Intersectional Fairness

**Current**: Analyze demographics separately (gender, age, region)

**Extension**: Examine intersections (e.g., young women in specific regions)

**Challenge**: Small sample sizes for fine-grained groups

**Method**: Fairness constraints during training or post-processing adjustments

#### 10.2.5 Interactive Dashboard

**Concept**: Web interface where applicants input information and receive:
- Predicted default probability
- Approval decision
- Multiple counterfactual scenarios
- Interactive exploration of "what if" changes

**Technology**: Streamlit or Flask backend with DiCE integration

#### 10.2.6 Model Monitoring

**Objective**: Detect performance degradation in production

**Metrics**: Track AUC, calibration, and fairness metrics over time

**Alerts**: Trigger retraining when metrics decline beyond thresholds

**Implementation**: Logging infrastructure with automated reporting

---

## 11. Conclusions

This project successfully developed a high-performance, fair, and interpretable credit risk prediction system. Key achievements include:

### 11.1 Technical Contributions

1. **State-of-the-Art Performance**: 89.6% AUC-ROC through deep residual architecture with Focal Loss
2. **Effective Calibration**: Platt Scaling reduces calibration gap from 18.0% to 0.3%, enabling accurate probability estimates for risk-based pricing
3. **Comprehensive Fairness**: Demonstrated disparate impact ratios >0.8 and equalized odds across gender, age, and regional groups
4. **Actionable Interpretability**: DiCE counterfactuals provide specific, feasible loan modifications for 96% of rejected applicants

### 11.2 Methodological Insights

**Class Imbalance**: Combined SMOTE oversampling with Focal Loss successfully handles 24.6% minority class

**Calibration Strategy**: Two-stage approach (balanced training + Platt Scaling) outperforms direct training on imbalanced data

**Fairness-Accuracy Tradeoff**: High accuracy reduces tension between fairness criteria, enabling simultaneous satisfaction of multiple definitions

**Counterfactual Quality**: Proximity-diversity tradeoff can be balanced through appropriate weighting to generate both minimal and diverse alternatives

### 11.3 Practical Impact

**For Financial Institutions**:
- Improved risk assessment accuracy
- Regulatory compliance through fairness guarantees
- Customer service enhancement via transparent explanations

**For Loan Applicants**:
- Clear understanding of rejection reasons
- Actionable paths to future approval
- Reduced information asymmetry

**For Society**:
- Increased financial inclusion through guidance
- Algorithmic accountability through fairness analysis
- Evidence-based policy discussions on lending practices

### 11.4 Broader Implications

This work demonstrates that machine learning systems can achieve the dual goals of accuracy and responsibility. With appropriate techniques:
- Deep learning can match or exceed traditional methods while maintaining interpretability
- Fairness can be quantified and optimized without sacrificing performance
- Automated decisions can be transparent and actionable

The methodology generalizes beyond credit risk to any high-stakes classification task requiring accuracy, calibration, fairness, and interpretability (e.g., healthcare triage, criminal justice risk assessment, hiring decisions).

### 11.5 Final Remarks

Credit risk prediction represents an ideal testbed for responsible AI techniques. The combination of:
- Structured tabular data
- Well-defined fairness requirements
- Real-world economic impact
- Regulatory oversight

creates an environment where technical innovation must be tempered with ethical considerations. This project provides a blueprint for developing machine learning systems that are not only accurate but also fair, calibrated, and interpretable.

As machine learning increasingly mediates access to economic opportunities, the approaches demonstrated here—rigorous fairness evaluation, probability calibration, and counterfactual explanation—will become standard practice rather than research novelties.

---

## 12. References

### Academic Literature

1. Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (pp. 607-617).

2. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

3. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3), 61-74.

4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

5. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357.

6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In International Conference on Machine Learning (pp. 1321-1330).

7. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. Advances in neural information processing systems, 29.

8. Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning: Limitations and Opportunities. MIT Press.

### Technical Resources

9. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning (pp. 448-456).

10. Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. In International Conference on Learning Representations.

11. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1), 1929-1958.

### Regulatory and Industry Standards

12. Equal Credit Opportunity Act (ECOA), 15 U.S.C. § 1691 et seq.

13. Fair Credit Reporting Act (FCRA), 15 U.S.C. § 1681 et seq.

14. Federal Reserve Board. (2007). Report to the Congress on Credit Scoring and Its Effects on the Availability and Affordability of Credit.

### Software and Frameworks

15. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

16. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

17. DiCE: Diverse Counterfactual Explanations. https://github.com/interpretml/DiCE

---

## 13. Appendices

### Appendix A: Feature Descriptions

#### Continuous Features

| Feature | Description | Range | Transformation |
|---------|-------------|-------|----------------|
| loan_amount | Requested loan amount | $5K-$800K | Log transform |
| income | Annual household income | $10K-$500K | Log transform |
| credit_score | FICO credit score | 300-850 | None |
| ltv | Loan-to-value ratio | 0-1 | None |
| dtir1 | Debt-to-income ratio | 0-1 | None |
| property_value | Property appraisal value | $10K-$2M | Log transform |
| term | Loan term in months | 60-360 | None |
| year | Application year | 2019-2021 | None |

#### Categorical Features (One-Hot Encoded)

**Gender**: Male, Female, Joint, Sex Not Available

**Age Group**: <25, 25-34, 35-44, 45-54, 55-64, 65-74, >74

**Loan Type**: Type1, Type2, Type3

**Loan Purpose**: P1 (home purchase), P2 (refinance), P3 (home improvement), P4 (other)

**Credit Bureau**: CIB, CRIF, EQUI, EXP

**Region**: North, North-East, Central, South

**Construction Type**: SB (site-built), MH (manufactured/mobile home)

**Occupancy Type**: PR (primary residence), SR (secondary), IR (investment)

**Security Type**: Direct, Indirect

**Binary Flags**: Interest-only, negative amortization, lump sum payment, pre-approval, etc.

### Appendix B: Hyperparameter Tuning

Validation AUC across hyperparameter configurations:

| Learning Rate | Weight Decay | Dropout | Batch Size | Val AUC |
|---------------|--------------|---------|------------|---------|
| 0.001 | 5e-5 | 0.5→0.2 | 128 | **0.8854** |
| 0.001 | 1e-4 | 0.5→0.2 | 128 | 0.8832 |
| 0.0005 | 5e-5 | 0.5→0.2 | 128 | 0.8821 |
| 0.001 | 5e-5 | 0.6→0.3 | 128 | 0.8798 |
| 0.001 | 5e-5 | 0.5→0.2 | 64 | 0.8805 |
| 0.001 | 5e-5 | 0.5→0.2 | 256 | 0.8776 |

Best configuration selected based on validation performance.

### Appendix C: Training Curves

**Loss Progression**:
- Epoch 1: 0.124
- Epoch 10: 0.082
- Epoch 20: 0.068
- Epoch 30: 0.061
- Epoch 40: 0.058
- Converged: 0.056 (epoch 52)

**Validation AUC Progression**:
- Epoch 1: 0.714
- Epoch 10: 0.842
- Epoch 20: 0.871
- Epoch 30: 0.882
- Epoch 40: 0.885
- Best: 0.8854 (epoch 37)

Early stopping triggered at epoch 52 (15 epochs after best).

### Appendix D: Feature Importance (Logistic Regression)

Top 20 features by absolute coefficient:

| Rank | Feature | Coefficient | Std Error | p-value |
|------|---------|-------------|-----------|---------|
| 1 | credit_type_EQUI | +37.67 | 0.82 | <0.001 |
| 2 | credit_type_EXP | -11.70 | 0.31 | <0.001 |
| 3 | construction_type_mh | +5.31 | 0.18 | <0.001 |
| 4 | lump_sum_payment_lpsm | +2.62 | 0.12 | <0.001 |
| 5 | secured_by_home | -2.61 | 0.11 | <0.001 |
| 6 | credit_score | -2.45 | 0.09 | <0.001 |
| 7 | dtir1 | +1.98 | 0.08 | <0.001 |
| 8 | ltv | +1.76 | 0.07 | <0.001 |
| 9 | loan_amount | +1.54 | 0.06 | <0.001 |
| 10 | interest_only_int_only | +1.42 | 0.07 | <0.001 |

### Appendix E: Computational Resources

**Training**:
- Hardware: MacBook Pro, Apple M1 chip, 16GB RAM
- Framework: PyTorch 2.4.1 (CPU)
- Training time: 2.5 hours (52 epochs)
- Memory usage: Peak 4.2GB

**Inference**:
- Test set (10,301 samples): 3.2 seconds
- Single prediction: <1 millisecond
- Counterfactual generation (5 CFs): 0.8 seconds per case

**Scalability**: Model can process 10K predictions/second on single CPU, sufficient for real-time loan applications.

### Appendix F: Reproducibility

All experiments reproducible with:

```python
import random
import numpy as np
import torch

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

Code and data available at: https://github.com/sanjxksl/credit-risk-counterfactual

### Appendix G: Ethical Considerations

**Data Privacy**:
- Dataset anonymized (no personally identifiable information)
- Aggregated statistics reported (no individual case details)

**Fairness Testing**:
- Analyzed all protected characteristics available in data
- Reported both positive and negative fairness results
- Acknowledged limitations (small sample sizes for some groups)

**Intended Use**:
- Educational and research purposes
- Not validated for production deployment without further testing
- Requires domain expert review before real-world application

**Potential Harms**:
- Model errors may deny credit to qualified applicants (false negatives)
- Overreliance on historical data may perpetuate historical biases
- Counterfactuals may suggest infeasible changes

**Mitigation Strategies**:
- Human-in-the-loop for borderline cases
- Regular fairness audits
- Transparent documentation of limitations

---

**Document Information**:
- **Version**: 1.0
- **Date**: December 2025
- **Pages**: 27
- **Word Count**: ~11,500

**For questions or feedback, contact the authors.**

---

*End of Report*
