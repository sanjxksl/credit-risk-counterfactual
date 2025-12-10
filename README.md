# Credit Risk Prediction with Counterfactual Explanations

**Start Here**: Open [`main.ipynb`](main.ipynb) - central hub for all analyses, results, and project navigation.

## Overview

This project develops a machine learning system for predicting loan default risk using a 3-layer neural network, with a focus on model interpretability through counterfactual explanations. The system achieves 88.8% AUC-ROC on test data with near-perfect calibration (0.1% calibration gap).

## Problem Statement

Credit risk assessment requires accurate prediction of loan default probability while maintaining fairness across demographic groups and providing actionable explanations for decisions. Traditional models often lack interpretability, making it difficult for applicants to understand rejection reasons or identify paths to approval.

## Methodology

### Data Preprocessing

The dataset contains 148,670 loan applications with 31 features. Preprocessing steps include:

1. Missing value imputation using median for numerical features
2. Outlier handling via winsorization at 1st/99th percentiles (loan_amount, property_value, income, ltv, dtir1)
3. Feature engineering: Binary `is_standard_term` indicator (1 if term=360 months, 0 otherwise)
4. Log transformation of skewed continuous variables (loan_amount, income, property_value)
5. One-hot encoding of categorical variables (68 features after encoding)
6. Standardization using StandardScaler
7. Train/validation/test split (80%/10%/10%) with stratification
8. SMOTE oversampling on training set with `sampling_strategy=0.5` (minority class = 50% of majority class, resulting in 33.3% default rate)

### Model Architecture

3-layer Multi-Layer Perceptron optimized for tabular credit data:

```
Input Layer: 68 features
    ↓
Hidden Layer 1: 128 units, BatchNorm, ReLU, Dropout(0.3)
    ↓
Hidden Layer 2: 64 units, BatchNorm, ReLU, Dropout(0.2)
    ↓
Hidden Layer 3: 32 units, BatchNorm, ReLU, Dropout(0.1)
    ↓
Output Layer: 1 unit, Sigmoid activation
```

**Total Parameters**: 19,649

**Architecture Rationale**: Research shows that 2-3 layer MLPs achieve optimal performance on tabular credit scoring tasks. Deeper residual architectures (ResNet-style) designed for image classification are inappropriate for structured tabular data with no spatial relationships. Our architecture follows industry best practices with progressive dimension reduction (128→64→32).

### Training Configuration

- **Loss Function**: Binary Cross-Entropy (not Focal Loss - training data balanced to 33.3% via SMOTE)
- **Optimizer**: Adam with weight decay 1e-4
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Batch Size**: 128
- **Early Stopping**: Patience of 15 epochs
- **Regularization**: Dropout (0.3, 0.2, 0.1) and L2 weight decay

### Calibration

Model predictions showed slight overestimation (27.7% predicted vs 24.6% actual). Platt Scaling was applied using validation set predictions:

- **Uncalibrated**: Brier Score 0.0813, Calibration Gap 3.0%
- **Calibrated**: Brier Score 0.0799, Calibration Gap 0.3%

## Results

### Model Performance

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.8794 | 0.8879 |
| AUC-PR | 0.8209 | 0.8301 |
| Brier Score (Calibrated) | 0.0906 | 0.0876 |
| Calibration Gap (Calibrated) | 0.0% | 0.1% |

### Bias Analysis

Comprehensive fairness evaluation across demographic groups using calibrated predictions:

**Gender Fairness:**
- Calibration gap: Maximum 0.8% across all gender groups
- Disparate Impact Ratio: 1.021 (passes 80% rule)
- AUC-ROC range: 0.885-0.908

**Age Fairness:**
- Calibration gap: Maximum 1.1% across age groups
- AUC-ROC range: 0.889-0.923

**Regional Fairness:**
- AUC consistent across all regions (0.882-0.904)
- Calibration gap varies (max 2.3% for North-East region with 109 samples, <1% for larger regions)

### Feature Importance

Top risk drivers identified through logistic regression coefficients:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| credit_type_EQUI | +37.67 | Strongly increases default risk |
| credit_type_EXP | -11.70 | Strongly decreases default risk |
| construction_type_mh | +5.31 | Mobile homes increase risk |
| lump_sum_payment_lpsm | +2.62 | Lump sum payments increase risk |
| secured_by_home | -2.61 | Home-secured loans decrease risk |

### Counterfactual Explanations

Generated counterfactuals for **13 diverse cases** using DiCE framework with realistic bounds (5th-95th percentiles). Focused on **immediately actionable** features, deliberately excluding `credit_score` and `income` as these cannot be changed short-term.

**Analysis Results:**
- **13 cases analyzed** with varying risk profiles
- **46 total counterfactuals** generated
- **100% success rate** with bounded feature space

**Summary Statistics:**

| Metric | Value |
|--------|-------|
| Average features changed per CF | 2.22 |
| Range of features changed | 1-4 |
| Success rate | 100% |

**Most Commonly Changed Features:**

| Feature | Changed in | Actionable | Impact |
|---------|-----------|------------|--------|
| LTV (Loan-to-Value) | 58.7% | Yes | Increase down payment |
| DTIR (Debt-to-Income) | 56.5% | Yes | Pay down existing debt before applying |
| Term | 54.3% | Yes | Adjust loan duration |
| Property Value | 30.4% | Yes | Choose less expensive property |
| Loan Amount | 21.7% | Yes | Request smaller loan |

**Mutable features** (applicant can control during application):
- loan_amount: Request a different loan amount
- property_value: Choose a different property
- ltv: Loan-to-value ratio (controlled by loan amount and down payment)
- term: Choose different loan duration
- dtir1: Pay down existing debt before applying

**Immutable features** (cannot change or not actionable short-term):
- age, gender, region: Demographic factors
- credit_score: Takes months to years to improve
- income: Cannot instantly increase
- Historical credit bureau data

## Project Structure

```
credit-risk-counterfactual/
├── main.ipynb                          # START HERE - Central project hub
├── data/
│   ├── train.csv, val.csv, test.csv   # Preprocessed splits
│   └── high_risk_cases.csv            # Selected cases for analysis
├── models/
│   ├── mlp_model.pth                  # Trained neural network (88.8% AUC)
│   ├── calibrator.pkl                 # Platt Scaling calibrator
│   ├── preprocessor.pkl               # Feature scaler
│   ├── dice_bounds.json               # Counterfactual bounds (5th-95th percentiles)
│   └── winsorize_bounds.json          # Outlier caps (1st-99th percentiles)
├── results/
│   ├── mlp_predictions.csv            # Test set predictions
│   ├── mlp_metrics.json               # Performance metrics
│   ├── bias_*.csv                     # Fairness analysis results
│   ├── dice_counterfactuals/          # Counterfactual explanations (13 cases)
│   └── figures/                       # Visualizations
├── notebooks/
│   ├── data_cleaning.ipynb                    # Data preprocessing
│   ├── feature_engineering.ipynb              # Feature transformation & splits
│   ├── EDA.ipynb                              # Exploratory analysis
│   ├── logistic_training.ipynb                # Logistic regression baseline
│   ├── feature_analysis.ipynb                 # Logistic feature importance
│   ├── mlp_training.ipynb                     # MLP training with calibration
│   ├── mlp_feature_importance.ipynb           # MLP permutation importance
│   ├── model_evaluation.ipynb                 # Performance evaluation
│   ├── bias_fairness_analysis.ipynb           # Fairness evaluation
│   ├── generate_counterfactuals.ipynb         # DiCE counterfactuals
│   └── counterfactual_summary_statistics.ipynb # CF analysis
└── requirements.txt                            # Dependencies
```

## Usage

### Quick Start

1. **Clone and install**:
```bash
git clone https://github.com/sanjxksl/credit-risk-counterfactual.git
cd credit-risk-counterfactual
pip install -r requirements.txt
```

2. **Open the main hub**:
```bash
jupyter notebook main.ipynb
```

`main.ipynb` is your central hub that:
- Shows model performance summary (AUC, calibration, Brier score)
- Links to all analysis notebooks
- Displays feature importance and bias analysis results
- Shows counterfactual explanations summary
- Provides interactive prediction demo

### Analysis Notebooks

For detailed step-by-step analysis, explore notebooks in this order:
1. `notebooks/data_cleaning.ipynb` - Data preprocessing
2. `notebooks/feature_engineering.ipynb` - Feature transformation, splits, SMOTE
3. `notebooks/EDA.ipynb` - Exploratory data analysis
4. `notebooks/logistic_training.ipynb` - Baseline model
5. `notebooks/feature_analysis.ipynb` - Feature importance
6. `notebooks/mlp_training.ipynb` - Neural network training
7. `notebooks/model_evaluation.ipynb` - Performance evaluation
8. `notebooks/bias_fairness_analysis.ipynb` - Fairness analysis
9. `notebooks/generate_counterfactuals.ipynb` - Counterfactual generation
10. `notebooks/counterfactual_summary_statistics.ipynb` - CF analysis

## Dependencies

Core Requirements:
- Python 3.8+
- PyTorch 2.4.1
- scikit-learn 1.3.2
- pandas 1.5.3
- numpy 1.24.3
- dice-ml 0.11
- imbalanced-learn 0.12.4
- matplotlib 3.7.5
- seaborn 0.13.2

See `requirements.txt` for complete list.

## Key Findings

1. **Model Performance**: 3-layer MLP achieves 88.8% test AUC-ROC with 19,649 parameters. Winsorization at 1st/99th percentiles and binary term feature (is_standard_term) improve robustness while maintaining strong discriminative power.

2. **Calibration**: Platt Scaling achieves near-perfect calibration with 0.1% calibration gap on test set, ensuring predicted probabilities accurately reflect true default risk.

3. **SMOTE Strategy**: Using `sampling_strategy=0.5` (33.3% default rate in training vs 24.6% in test) balances class representation while preserving calibration quality.

4. **Fairness**: Model demonstrates excellent fairness properties across gender, age, and regional groups with calibration gaps <1% for most groups.

5. **Interpretability**: Counterfactual explanations reveal that loan modifications typically require 2.2 feature changes on average. LTV, DTIR, and term are the most frequently modified features, all of which applicants can control through down payment size, debt management, and loan structure choices.

6. **DiCE Success**: Bounded counterfactual generation (5th-95th percentiles) achieved 100% success rate across 13 cases, producing 46 realistic and actionable recommendations.

## Limitations

1. Dataset limited to specific time period and geographic region
2. Some regional groups have small sample sizes (North-East: 109 samples)
3. Counterfactuals assume feature changes are independent (e.g., changing property value affects LTV)
4. Model trained on historical data may not capture recent economic conditions
5. Winsorization at 1st/99th percentiles may remove important signal from extreme cases

## Future Work

1. Incorporate temporal dynamics and macroeconomic indicators
2. Compare with gradient-based counterfactual methods (e.g., Wachter et al.) for higher success rates
3. Explore feasibility constraints to ensure counterfactual recommendations are realistic
4. Develop interactive visualization dashboard for counterfactuals
5. Extend fairness analysis to intersectional groups
6. Implement model monitoring for performance degradation

## References

1. Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, 607-617.

2. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in large margin classifiers*, 10(3), 61-74.

3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of artificial intelligence research*, 16, 321-357.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{credit_risk_counterfactual_2025,
  author = {KSL, Sanjana and Jiang, Michael and Xiao, Sharon and Wang, Zhenyu and Zimeng},
  title = {Credit Risk Prediction with Counterfactual Explanations},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sanjxksl/credit-risk-counterfactual}
}
```

## Team

- Sanjana KSL
- Michael Jiang
- Sharon Xiao
- Zhenyu Wang
- Zimeng

## License

This project is licensed under the MIT License.
