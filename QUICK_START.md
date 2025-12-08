# Quick Start Guide

## Main Entry Point

**Open this file first**: `main.ipynb`

This is your project hub that:
- Links to all other notebooks
- Runs all analyses (feature importance, bias analysis, counterfactuals)
- Shows model performance summary
- Provides quick prediction demo

## File Structure

```
main.ipynb                    ← START HERE (central hub)
├── Links to all notebooks
├── Runs all Python scripts
└── Shows results summary

Python Scripts (run from main.ipynb):
├── feature_importance.py     → Generates top features analysis
├── bias_analysis.py          → Fairness evaluation
└── dice_setup.py            → Counterfactual generation

Analysis Notebooks (linked from main.ipynb):
├── notebooks/data_cleaning.ipynb
├── notebooks/EDA.ipynb
├── notebooks/feature_analysis.ipynb
├── notebooks/mlp_training.ipynb
└── notebooks/model_evaluation.ipynb
```

## How to Use

### Option 1: Run Everything (Recommended First Time)
```bash
jupyter notebook main.ipynb
# Then run all cells (Cell → Run All)
```

This will:
1. Show model performance
2. Generate feature importance analysis
3. Perform bias/fairness analysis
4. Generate counterfactual explanations
5. Display results summary

### Option 2: Interactive Predictions
```bash
# In main.ipynb, scroll to "Interactive Prediction Demo" section
# Or run individual scripts:
python feature_importance.py
python bias_analysis.py
python dice_setup.py
```

### Option 3: Step-by-Step Analysis
Open and run notebooks in order:
1. data_cleaning.ipynb
2. EDA.ipynb
3. feature_analysis.ipynb
4. mlp_training.ipynb
5. model_evaluation.ipynb

## What Each Script Does

**feature_importance.py**
- Analyzes top risk factors
- Generates bar chart visualization
- Saves to: results/top_features.csv, results/figures/feature_importance.png

**bias_analysis.py**
- Evaluates fairness across demographics (gender, age, region)
- Checks disparate impact and equalized odds
- Saves to: results/bias_*.csv, results/bias_analysis.json

**dice_setup.py**
- Generates counterfactual explanations for high-risk cases
- Shows what changes would lead to approval
- Saves to: results/dice_counterfactuals/

## Key Features

From `main.ipynb` you can:
- ✓ View all results in one place
- ✓ Run all analyses with one click
- ✓ Navigate to detailed notebooks
- ✓ Get quick predictions
- ✓ Access full documentation

## Results Location

All results are saved in `results/` directory:
- `mlp_predictions.csv` - Test set predictions
- `mlp_metrics.json` - Model performance
- `bias_*.csv` - Fairness metrics
- `top_features.csv` - Feature importance
- `dice_counterfactuals/` - Counterfactual explanations
- `figures/` - All visualizations

## Documentation

- `README.md` - Project overview
- `PROJECT_REPORT.md` - Technical report (5,300 words) - Convert to Word with pandoc
- `PROJECT_STRUCTURE.md` - Detailed file organization

## Quick Commands

```bash
# Start main hub
jupyter notebook main.ipynb

# Run individual analyses
python feature_importance.py
python bias_analysis.py
python dice_setup.py

# View results
ls results/
ls results/figures/
ls results/dice_counterfactuals/
```
