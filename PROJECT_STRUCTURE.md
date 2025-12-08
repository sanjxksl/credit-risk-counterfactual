# Project Structure

## Main Files

### Interactive Demo
- **SHOWCASE.ipynb** - Main interactive notebook for demonstrations
  - Load pre-trained model and calibrator
  - Input custom loan applications
  - Get predictions with counterfactual explanations
  - View performance metrics and fairness analysis

### Documentation
- **README.md** - Project overview and usage guide
- **PROJECT_REPORT.md** - Comprehensive technical report (5,300 words)
- **LICENSE** - MIT License

### Requirements
- **requirements.txt** - Python dependencies

## Notebooks (notebooks/)

Analysis notebooks in recommended execution order:

1. **data_cleaning.ipynb** - Data preprocessing and feature engineering
2. **EDA.ipynb** - Exploratory data analysis with visualizations
3. **feature_analysis.ipynb** - Feature importance analysis
4. **mlp_training.ipynb** - Deep learning model training
5. **model_evaluation.ipynb** - Model comparison and evaluation

## Data (data/)

- **Loan_Default.csv** - Original dataset (148,670 samples)
- **cleaned_loan_data.csv** - Preprocessed data
- **train.csv** - Training set (86%, with SMOTE balancing)
- **val.csv** - Validation set (7%, natural distribution)
- **test.csv** - Test set (7%, natural distribution)

## Models (models/)

- **mlp_model.pth** - Trained PyTorch model (1.77M parameters)
- **calibrator.pkl** - Platt Scaling calibrator
- **preprocessor.pkl** - StandardScaler for feature normalization
- **logistic_model.pkl** - Logistic regression baseline
- **feature_names.txt** - Feature name mapping (67 features)

## Results (results/)

### Predictions
- **mlp_predictions.csv** - Test set predictions (calibrated and uncalibrated)
- **logistic_predictions.csv** - Baseline model predictions

### Metrics
- **mlp_metrics.json** - Model performance metrics
- **bias_analysis.json** - Comprehensive fairness evaluation
- **bias_gender.csv** - Gender-specific metrics
- **bias_age.csv** - Age-specific metrics
- **bias_region.csv** - Regional metrics
- **top_features.csv** - Top 10 important features
- **model_comparison.csv** - Model comparison table

### Visualizations (results/figures/)
- ROC curves
- Precision-recall curves
- Calibration plots
- Feature importance charts

### Counterfactuals (results/dice_counterfactuals/)
- **high_risk_cases.csv** - Selected high-risk applications
- **verification_summary.csv** - Counterfactual flip rates
- **counterfactuals_case_*.csv** - Individual counterfactual scenarios

## Documentation (docs/)

- **dice_counterfactual_guide.md** - DiCE framework usage guide
- **feature_importance_notes.md** - Feature analysis documentation

## Workflow

### For Quick Demo:
1. Open **SHOWCASE.ipynb**
2. Run all cells
3. Modify case inputs in Section 3
4. Re-run Sections 3-5 for new predictions

### For Complete Analysis:
1. Run notebooks 1-5 in sequence
2. Review results in `results/` directory
3. Check **PROJECT_REPORT.md** for detailed findings

### For Custom Applications:
1. Use **SHOWCASE.ipynb** to input new cases
2. Get predictions and counterfactual recommendations
3. Analyze fairness across demographics

## Key Results

- **Performance**: 89.6% AUC-ROC, 0.081 Brier Score (calibrated)
- **Fairness**: Passes disparate impact (>0.8) and equalized odds (<0.1 difference)
- **Calibration**: 0.3% gap between predicted and actual default rates
- **Interpretability**: 96% of rejected cases have actionable counterfactuals
