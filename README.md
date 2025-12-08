# Credit Risk Prediction with Counterfactual Explanations

## Overview

This project develops a machine learning system for predicting loan default risk using deep neural networks, with a focus on model interpretability through counterfactual explanations. The system achieves 89.6% AUC-ROC on test data and demonstrates strong calibration properties after Platt Scaling.

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

Uncalibrated model predictions showed systematic overestimation (42.7% vs actual 24.6%). Platt Scaling was applied using validation set predictions to correct probability estimates:

- Uncalibrated: Brier Score 0.126, Calibration Gap 18.0%
- Calibrated: Brier Score 0.081, Calibration Gap 0.3%

## Results

### Model Performance

| Metric | Validation | Test |
|--------|-----------|------|
| AUC-ROC | 0.8854 | 0.8962 |
| AUC-PR | 0.8322 | 0.8438 |
| Brier Score (Uncalibrated) | 0.1282 | 0.1262 |
| Brier Score (Calibrated) | N/A | 0.0812 |

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

DiCE framework generates actionable recommendations for loan modification. Example for high-risk case:

- Original: Default probability 99.99%
- Counterfactual 1: Reduce loan amount by $15,000 → 17.6% (approved)
- Counterfactual 2: Increase down payment by 10% → 16.6% (approved)

Mutable features: loan_amount, income, credit_score, ltv, dtir1, property_value, term

Immutable features: age, gender, region, historical credit data

## Project Structure

```
credit-risk-counterfactual/
├── data/
│   ├── Loan_Default.csv              # Original dataset
│   ├── cleaned_loan_data.csv          # Preprocessed data
│   ├── train.csv                      # Training set (86%)
│   ├── val.csv                        # Validation set (7%)
│   └── test.csv                       # Test set (7%)
├── models/
│   ├── mlp_model.pth                  # Trained neural network
│   ├── calibrator.pkl                 # Platt Scaling calibrator
│   ├── preprocessor.pkl               # Feature scaler
│   └── feature_names.txt              # Feature mapping
├── results/
│   ├── mlp_predictions.csv            # Model predictions
│   ├── mlp_metrics.json               # Performance metrics
│   ├── bias_analysis.json             # Fairness evaluation
│   ├── bias_gender.csv                # Gender bias metrics
│   ├── bias_age.csv                   # Age bias metrics
│   ├── bias_region.csv                # Regional bias metrics
│   └── figures/                       # Visualizations
├── notebooks/
│   ├── data_cleaning.ipynb            # Data preprocessing
│   ├── EDA.ipynb                      # Exploratory analysis
│   ├── feature_analysis.ipynb         # Feature importance
│   ├── mlp_training.ipynb             # Model training
│   └── model_evaluation.ipynb         # Performance evaluation
├── docs/
│   ├── dice_counterfactual_guide.md   # Counterfactual documentation
│   └── feature_importance_notes.md    # Feature analysis notes
├── bias_analysis.py                    # Fairness evaluation script
├── dice_setup.py                       # Counterfactual generation
├── evaluation_summary.py               # Results aggregation
├── feature_importance.py               # Feature analysis
└── requirements.txt                    # Dependencies
```

## Usage

### Installation

```bash
git clone https://github.com/sanjxksl/credit-risk-counterfactual.git
cd credit-risk-counterfactual
pip install -r requirements.txt
```

### Quick Start - Interactive Demo

For an interactive demonstration with counterfactual explanations:

```bash
jupyter notebook SHOWCASE.ipynb
```

This notebook allows you to:
- Input custom loan application cases
- Get instant predictions with calibrated probabilities
- Generate counterfactual recommendations for rejected applications
- View model performance and fairness metrics

### Analysis Notebooks

Execute notebooks in sequence for complete analysis:

```bash
jupyter notebook notebooks/data_cleaning.ipynb       # Data preprocessing
jupyter notebook notebooks/EDA.ipynb                 # Exploratory analysis
jupyter notebook notebooks/feature_analysis.ipynb    # Feature importance
jupyter notebook notebooks/mlp_training.ipynb        # Model training
jupyter notebook notebooks/model_evaluation.ipynb    # Performance evaluation
```

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

1. **Model Performance**: Deep residual architecture with Focal Loss achieves 89.6% AUC-ROC, outperforming logistic regression baseline by 4.8 percentage points.

2. **Calibration**: Platt Scaling effectively corrects probability estimates, reducing calibration gap from 18.0% to 0.3% while preserving discrimination.

3. **Fairness**: Model demonstrates excellent fairness properties across gender, age, and regional groups. Passes disparate impact 80% rule and satisfies equalized odds criterion.

4. **Interpretability**: Counterfactual explanations provide actionable recommendations for loan modifications, with average 3-5 feature changes required to flip predictions.

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
