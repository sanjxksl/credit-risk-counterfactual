# Execution Guide

Complete workflow for training and evaluating credit risk prediction models.

## Running Order

Execute notebooks sequentially in this order:

```
1. data_cleaning.ipynb       
2. EDA.ipynb                    
3. feature_engineering.ipynb  
4. logistic_regression.ipynb    
5. mlp_training.ipynb           
6. model_evaluation.ipynb  

```

## Notebook Descriptions

### data_cleaning.ipynb
Cleans raw Kaggle dataset: handles missing values, standardizes column names, applies log transformations.

**Input:** `data/Loan_Default.csv`
**Output:** `data/cleaned_loan_data.csv`

### EDA.ipynb
Exploratory data analysis: dataset overview, class distribution, correlation analysis.

**Input:** `data/cleaned_loan_data.csv`
**Output:** Visualizations and insights

### feature_engineering.ipynb
Creates model-ready features: one-hot encoding, standardization, train/val/test split, SMOTE.

**Input:** `data/cleaned_loan_data.csv`
**Output:** `data/train.csv`, `data/val.csv`, `data/test.csv`, `models/preprocessor.pkl`

### logistic_regression.ipynb
Trains logistic regression with 5-fold cross-validation.

**Input:** Train/val/test datasets
**Output:** `models/logistic_model.pkl`, `results/logistic_predictions.csv`, `results/logistic_metrics.json`
**Performance:** Validation AUC-ROC = 0.8418

### mlp_training.ipynb
Trains 4-layer neural network with batch normalization, gradient clipping, and learning rate scheduling.

**Input:** Train/val/test datasets
**Output:** `models/mlp_model.pth`, `results/mlp_predictions.csv`, `results/mlp_metrics.json`
**Performance:** Validation AUC-ROC = 0.8849, Test AUC-ROC = 0.8943

### model_evaluation.ipynb
Compares models with visualizations and performance metrics.

**Input:** Model predictions and metrics
**Output:** `results/model_comparison.csv`, comparison plots
**Result:** MLP outperforms logistic regression by 5%

## Data Flow

```
Loan_Default.csv
    → [data_cleaning]
cleaned_loan_data.csv
    → [EDA] → Insights
    → [feature_engineering]
train.csv + val.csv + test.csv + preprocessor.pkl
    → [logistic] → logistic_model.pkl
    → [mlp] → mlp_model.pth
    → [evaluation] → Comparison table + plots
```


## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages: pandas, numpy, scikit-learn, imbalanced-learn, torch, matplotlib, seaborn

## File Structure

```
data/
├── Loan_Default.csv (original)
├── cleaned_loan_data.csv
├── train.csv
├── val.csv
└── test.csv

models/
├── preprocessor.pkl
├── feature_names.txt
├── logistic_model.pkl
└── mlp_model.pth

results/
├── logistic_predictions.csv
├── logistic_metrics.json
├── mlp_predictions.csv
├── mlp_metrics.json
├── model_comparison.csv
└── figures/
    ├── roc_curve.png
    ├── pr_curve.png
    ├── calibration_plot.png
    └── probability_distribution.png
```
- Preprocessor is fitted on training data only to prevent data leakage
- SMOTE is applied to training set only

## Troubleshooting

**File not found errors:** Ensure previous notebooks have been run successfully
**Module not found:** Run `pip install -r requirements.txt`
**Kernel crashes:** Reduce batch size in notebook 04 (512 → 256)
