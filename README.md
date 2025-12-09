# Credit Risk Prediction with Counterfactual Explanations

**Start Here**: Open [`main.ipynb`](main.ipynb) - your central hub for all analyses, results, and project navigation.

## Overview

This project develops a machine learning system for predicting loan default risk using deep neural networks, with a focus on model interpretability through counterfactual explanations. The system achieves 89.3% AUC-ROC on test data and demonstrates strong calibration properties after Platt Scaling.

## Problem Statement

Credit risk assessment requires accurate prediction of loan default probability while maintaining fairness across demographic groups and providing actionable explanations for decisions. Traditional models often lack interpretability, making it difficult for applicants to understand rejection reasons or identify paths to approval.

## Methodology

### Data Preprocessing

The dataset contains 148,670 loan applications with 31 features. Preprocessing steps include:

1. Missing value imputation using median for numerical features
2. Log transformation of skewed continuous variables
3. One-hot encoding of categorical variables (67 features after encoding)
4. Standardization using StandardScaler
5. Train/validation/test split (86%/7%/7%)
6. SMOTE oversampling on training set to address 24.6% class imbalance

### Model Architecture

Deep Residual Multi-Layer Perceptron with the following specifications:

```
Input Layer: 67 features
    ↓
Hidden Layer: 512 units, BatchNorm, ReLU, Dropout(0.5)
    ↓
Residual Block 1: 512 units (2 layers with skip connection)
Residual Block 2: 512 units (2 layers with skip connection)
    ↓
Hidden Layer: 256 units, BatchNorm, ReLU, Dropout(0.4)
    ↓
Residual Block 3: 256 units (2 layers with skip connection)
    ↓
Hidden Layer: 128 units, BatchNorm, ReLU, Dropout(0.3)
    ↓
Residual Block 4: 128 units (2 layers with skip connection)
    ↓
Hidden Layer: 64 units, BatchNorm, ReLU, Dropout(0.2)
    ↓
Output Layer: 1 unit, Sigmoid activation
```

Total Parameters: 1,773,569

### Training Configuration

- Loss Function: Focal Loss (alpha=0.25, gamma=2.0) to handle class imbalance
- Optimizer: AdamW with weight decay 5e-5
- Learning Rate: 0.001 with Cosine Annealing schedule
- Batch Size: 128
- Early Stopping: Patience of 15 epochs
- Gradient Clipping: Maximum norm of 1.0

### Calibration

Uncalibrated model predictions showed systematic overestimation (48.0% vs actual 24.6%). Platt Scaling was applied using validation set predictions to correct probability estimates:

- Uncalibrated: Brier Score 0.158, Calibration Gap 23.3%
- Calibrated: Brier Score 0.083, Calibration Gap 0.2%

## Results

### Model Performance

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.8810 | 0.8934 |
| AUC-PR | 0.8257 | 0.8418 |
| Brier Score (Uncalibrated) | 0.1594 | 0.1577 |
| Brier Score (Calibrated) | 0.0873 | 0.0825 |

### Bias Analysis

Comprehensive fairness evaluation across demographic groups:

**Gender Fairness:**
- Calibration gap: Maximum 0.5% across all gender groups
- Disparate Impact Ratio: 1.015 (passes 80% rule)
- Equalized Odds: FPR difference 0.004, FNR difference 0.007 (both < 0.1)

**Age Fairness:**
- Calibration gap: Maximum 1.9% across age groups
- Disparate Impact Ratio: 1.040 (passes 80% rule)
- Equalized Odds: FPR difference 0.007, FNR difference 0.010 (both < 0.1)

**Regional Fairness:**
- AUC consistent across all regions (0.886-0.899)
- Calibration gap varies (max 6.7% for North-East region with only 109 samples)

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

Generated counterfactuals for **10 diverse cases** using DiCE framework:
- **6 high-risk cases** (predicted default): 100% success rate (6/6 cases with actionable counterfactuals)
- **4 low-risk cases** (predicted no default): 0% flip rate (already approved, counterfactuals not applicable)
- **40 total counterfactuals** (4-5 per case) showing different paths to change predictions
- **Overall flip rate**: 60% (30/50 counterfactuals across all cases)

**Summary Statistics:**

| Metric | Value |
|--------|-------|
| Average features changed per CF | 2.33 |
| Range of features changed | 1-3 |
| Average magnitude of changes | 9.42 (standardized scale) |
| Success rate | 100% |

**Most Commonly Changed Features:**

| Feature | Changed in | Actionable? | Impact |
|---------|-----------|-------------|--------|
| LTV (Loan-to-Value) | 82.5% | ✅ YES | Primary driver - reduce via down payment |
| Term | 40.0% | ✅ YES | Choose different loan duration |
| Income | 27.5% | ❌ NO | Cannot instantly increase (not actionable) |
| Property Value | 22.5% | ✅ YES | Choose less expensive property |
| Loan Amount | 22.5% | ✅ YES | Request smaller loan |
| DTIR (Debt-to-Income) | 22.5% | ✅ YES | Pay down existing debt before applying |
| Credit Score | 15.0% | ❌ NO | Takes months/years to improve (not actionable) |

**Note**: Features marked ❌ appear in existing counterfactuals but are not realistically actionable for immediate loan applications. Future counterfactual generation will constrain these as immutable.

Example for high-risk case (Case 6183, P(default)=83.2%):
- **Original**: Default probability 83.2% → **REJECTED**
- **Counterfactuals generated**: 5 scenarios, all successfully flip to approval
- **Changes required**: Average 2.5 features (typically LTV, property value, or loan amount)
- **Actionable path**: Reduce LTV ratio by increasing down payment, choosing less expensive property, or requesting smaller loan

**Mutable features** (Applicant can control during application):
- `loan_amount` - Request a different loan amount
- `property_value` - Choose a different property
- `ltv` - Loan-to-value ratio (controlled by loan amount & down payment)
- `term` - Choose different loan duration
- `dtir1` - Pay down existing debt before applying

**Immutable features** (Cannot change or not actionable short-term):
- `age`, `gender`, `region` - Demographic factors
- `credit_score` - Takes months/years to improve
- `income` - Cannot instantly increase
- Historical credit bureau data

## Project Structure

```
credit-risk-counterfactual/
├── main.ipynb                          # 🎯 START HERE - Central project hub
├── data/
│   ├── train.csv, val.csv, test.csv   # Preprocessed splits
│   └── high_risk_cases.csv            # Selected cases for analysis
├── models/
│   ├── mlp_model.pth                  # Trained neural network (89.6% AUC)
│   ├── calibrator.pkl                 # Platt Scaling calibrator
│   └── preprocessor.pkl               # Feature scaler
├── results/
│   ├── mlp_predictions.csv            # Test set predictions
│   ├── mlp_metrics.json               # Performance metrics
│   ├── bias_*.csv                     # Fairness analysis results
│   ├── dice_counterfactuals/          # Counterfactual explanations (10 cases)
│   └── figures/                       # Visualizations
├── notebooks/
│   ├── data_cleaning.ipynb                    # Data preprocessing
│   ├── feature_engineering.ipynb              # Feature transformation & train/test split
│   ├── EDA.ipynb                              # Exploratory analysis
│   ├── logistic_training.ipynb                # Logistic regression baseline
│   ├── feature_analysis.ipynb                 # Feature importance
│   ├── mlp_training.ipynb                     # Model training
│   ├── model_evaluation.ipynb                 # Performance evaluation
│   ├── bias_fairness_analysis.ipynb           # Fairness evaluation
│   ├── generate_counterfactuals.ipynb         # Counterfactual explanations
│   └── counterfactual_summary_statistics.ipynb # CF analysis & visualizations
├── dice_setup.py                               # Counterfactual utilities (used by notebooks)
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
1. `notebooks/data_cleaning.ipynb` - Data preprocessing (missing values, log transform)
2. `notebooks/feature_engineering.ipynb` - Feature transformation, train/val/test split, SMOTE
3. `notebooks/EDA.ipynb` - Exploratory data analysis
4. `notebooks/logistic_training.ipynb` - Logistic regression baseline model
5. `notebooks/feature_analysis.ipynb` - Feature importance from logistic regression
6. `notebooks/mlp_training.ipynb` - Deep neural network training
7. `notebooks/model_evaluation.ipynb` - Performance evaluation and comparison
8. `notebooks/bias_fairness_analysis.ipynb` - Fairness evaluation across demographics
9. `notebooks/generate_counterfactuals.ipynb` - Counterfactual explanations generation
10. `notebooks/counterfactual_summary_statistics.ipynb` - Comprehensive CF analysis and visualizations

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

1. **Model Performance**: Deep residual architecture with Focal Loss achieves 89.3% AUC-ROC, outperforming logistic regression baseline.

2. **Calibration**: Platt Scaling effectively corrects probability estimates, reducing calibration gap from 23.3% to 0.2% while preserving discrimination.

3. **Fairness**: Model demonstrates excellent fairness properties across gender, age, and regional groups. Passes disparate impact 80% rule and satisfies equalized odds criterion.

4. **Interpretability**: Counterfactual explanations provide actionable recommendations for loan modifications. On average, only 2.33 feature changes are required to flip high-risk predictions to approval. LTV ratio is the most critical factor (changed in 82.5% of counterfactuals).

5. **Risk Drivers**: Credit bureau type, construction type, and loan structure (lump sum payments) are strongest predictors of default risk.

## Limitations

1. Dataset limited to specific time period and geographic region
2. Some regional groups have small sample sizes (North-East: 109 samples)
3. Counterfactuals assume feature changes are independent
4. Model trained on historical data may not capture recent economic conditions

## Future Work

1. Incorporate temporal dynamics and macroeconomic indicators
2. Explore ensemble methods combining multiple architectures
3. Develop interactive visualization dashboard for counterfactuals
4. Extend fairness analysis to intersectional groups
5. Implement model monitoring for performance degradation

## References

1. Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (pp. 607-617).

2. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

3. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in large margin classifiers, 10(3), 61-74.

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
