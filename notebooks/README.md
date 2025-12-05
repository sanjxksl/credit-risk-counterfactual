# Notebooks Guide

This folder contains Jupyter notebooks for the credit risk prediction project.

## 📚 Notebooks Overview

### 1. EDA.ipynb (Sharon - Dec 2) ✅
**Exploratory Data Analysis**
- Dataset overview and statistics
- Missing value analysis
- Class distribution (75% non-default, 25% default)
- Univariate and bivariate analysis
- Correlation analysis

**Key Finding:** Weak linear correlations, but data preprocessing is good.

---

### 2. 02_logistic_regression.ipynb (San - Dec 4) ✅
**Logistic Regression Training & Evaluation**

**What it does:**
- Loads preprocessed data (train/val/test)
- Trains logistic regression with 5-fold cross-validation
- Tunes hyperparameter C (regularization strength)
- Evaluates on validation and test sets
- Saves model and predictions

**Key Results:**
- Best C = 100 (weak regularization)
- Validation AUC-ROC: **0.8418**
- Test AUC-ROC: **0.8512**
- Well-calibrated probabilities

**Outputs:**
- `models/logistic_model.pkl`
- `results/logistic_predictions.csv`
- `results/logistic_metrics.json`

**Run time:** ~3-5 minutes

---

### 3. 03_mlp_training.ipynb (San - Dec 4) ✅
**Neural Network Training & Evaluation**

**What it does:**
- Defines 3-layer MLP (67 → 128 → 64 → 32 → 1)
- Trains with PyTorch using early stopping
- Evaluates on validation and test sets
- Visualizes training progress
- Saves model and predictions

**Architecture:**
```
Input (67 features)
    ↓
[Linear 128] + ReLU + Dropout(0.3)
    ↓
[Linear 64] + ReLU + Dropout(0.3)
    ↓
[Linear 32] + ReLU + Dropout(0.3)
    ↓
[Linear 1] + Sigmoid
    ↓
Output: Probability (0-1)
```

**Key Results:**
- Total parameters: 19,073
- Early stopped at epoch 18
- Validation AUC-ROC: **0.8815**
- Test AUC-ROC: **0.8917**
- Better calibrated than logistic

**Outputs:**
- `models/mlp_model.pth`
- `results/mlp_predictions.csv`
- `results/mlp_metrics.json`

**Run time:** ~5-8 minutes

---

### 4. 04_model_evaluation.ipynb (San - Dec 4) ✅
**Model Comparison & Visualization**

**What it does:**
- Loads predictions from both models
- Generates comparison visualizations:
  - ROC curves
  - Precision-Recall curves
  - Calibration plots
  - Probability distributions
- Creates performance comparison table
- Provides detailed analysis

**Key Insights:**
- MLP outperforms logistic by **~4%** (0.8815 vs 0.8418 AUC-ROC)
- Both models generalize well (test ≥ validation)
- MLP better calibrated (Brier: 0.1101 vs 0.1338)
- Logistic more interpretable → use for feature importance and DiCE

**Outputs:**
- `results/model_comparison.csv`
- `results/evaluation_summary.json`
- All visualizations displayed inline

**Run time:** ~1-2 minutes

---

## 🚀 How to Run

### Prerequisites
```bash
# Install dependencies
pip install -r ../requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- torch
- matplotlib
- seaborn
- jupyter

### Option 1: Run in Jupyter Notebook
```bash
# Start Jupyter from the project root
cd /Users/SanjanaKSL/Desktop/credit-risk-counterfactual
jupyter notebook

# Navigate to notebooks/ folder
# Open and run notebooks in order: EDA → 02 → 03 → 04
```

### Option 2: Run in VS Code
1. Open the project folder in VS Code
2. Install the Jupyter extension
3. Open any `.ipynb` file
4. Select Python kernel (Python 3.8+)
5. Run cells sequentially (Shift + Enter)

### Option 3: Run in Google Colab
1. Upload notebooks to Google Drive
2. Upload `data/` folder to Google Drive
3. Update file paths in notebooks to point to Drive
4. Run cells in order

---

## 📊 Expected Workflow

**For reproducing results:**

1. **Start with EDA.ipynb** (already done by Sharon)
   - Understand the dataset
   - Check data quality

2. **Run 02_logistic_regression.ipynb**
   - Trains logistic regression model
   - ~3-5 minutes
   - Outputs saved to `models/` and `results/`

3. **Run 03_mlp_training.ipynb**
   - Trains MLP model
   - ~5-8 minutes
   - Outputs saved to `models/` and `results/`

4. **Run 04_model_evaluation.ipynb**
   - Loads both models' predictions
   - Generates all comparison plots
   - ~1-2 minutes
   - Creates final comparison table

**Total time:** ~10-15 minutes for all notebooks

---

## 📁 File Dependencies

```
notebooks/
├── EDA.ipynb                      (uses: ../data/Loan_Default.csv)
├── 02_logistic_regression.ipynb   (uses: ../data/train.csv, val.csv, test.csv)
├── 03_mlp_training.ipynb          (uses: ../data/train.csv, val.csv, test.csv)
└── 04_model_evaluation.ipynb      (uses: ../results/*_predictions.csv, *_metrics.json)
```

**Important:** Run notebooks 02 and 03 **before** notebook 04.

---

## 🎯 For Report Writing

### Olivia's Section (Metrics Evaluation - Dec 6)
Use outputs from:
- `04_model_evaluation.ipynb` - All plots and comparison table
- `results/model_comparison.csv` - Performance table
- `results/evaluation_summary.json` - Summary statistics

### VC's Section (Feature Importance - Dec 5)
Start from:
- `models/logistic_model.pkl` - Trained logistic model
- `models/feature_names.txt` - List of feature names
- Extract coefficients to create feature importance ranking

### San's Section (DiCE Counterfactuals - Dec 7)
Use:
- `models/logistic_model.pkl` - Model for predictions
- `models/preprocessor.pkl` - For transforming features
- `results/logistic_predictions.csv` - High-risk cases

---

## ⚠️ Troubleshooting

### Issue: "Module not found" errors
**Solution:**
```bash
pip install -r ../requirements.txt
```

### Issue: "File not found" errors
**Solution:**
- Check that you're running from the `notebooks/` directory
- Verify file paths use `../data/` and `../models/` (relative paths)
- Ensure you've run `feature_engineering.py` first to create train/val/test splits

### Issue: PyTorch not working
**Solution:**
```bash
# For CPU-only (faster download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For Mac M1/M2
pip install torch
```

### Issue: Notebook kernel crashes
**Solution:**
- Reduce batch size in MLP notebook (512 → 256)
- Close other applications to free memory
- Restart kernel and run cells one by one

---

## 📝 Notes

### Code Cells vs Markdown Cells
- **Markdown cells:** Explanations, instructions (like this README)
- **Code cells:** Python code that you can run
- Run code cells in order (top to bottom)

### Saving Outputs
- Notebooks save outputs inline (you can see results without re-running)
- To regenerate outputs: "Kernel" → "Restart & Run All"
- To clear outputs: "Kernel" → "Restart & Clear Output"

### Best Practices
- Run cells sequentially (don't skip cells)
- Read markdown explanations before running code
- Check outputs after each cell to catch errors early
- Save notebook frequently (Ctrl/Cmd + S)

---

## 🎓 For Presentation

**Best notebooks for slides:**

1. **EDA.ipynb** - Show data understanding
   - Dataset shape: 148,670 samples
   - Class imbalance visualization
   - Key feature distributions

2. **04_model_evaluation.ipynb** - Show results
   - ROC curve comparison (MLP wins)
   - Performance table (0.8418 vs 0.8815)
   - Calibration plots (both well-calibrated)

**Suggested slides:**
- Slide 4-7: Screenshots from EDA.ipynb (data overview)
- Slide 8-12: Screenshots from 04_model_evaluation.ipynb (results)
- Slide 13-15: Case studies from DiCE (San - Dec 7)

---

## 🔗 Additional Resources

**Understanding Metrics:**
- **AUC-ROC:** 0.5 = random, 0.7-0.8 = good, 0.8-0.9 = very good, 0.9-1.0 = excellent
- **Brier Score:** 0.0 = perfect calibration, lower is better
- **Precision:** Of predicted defaults, what % are correct?
- **Recall:** Of actual defaults, what % did we catch?

**PyTorch Basics:**
- `model.train()` - Enable dropout during training
- `model.eval()` - Disable dropout during evaluation
- `torch.no_grad()` - Don't compute gradients (faster inference)

**Jupyter Shortcuts:**
- `Shift + Enter` - Run cell and move to next
- `Ctrl/Cmd + Enter` - Run cell and stay
- `A` - Insert cell above
- `B` - Insert cell below
- `DD` - Delete cell

---

## ✅ Checklist for Team

**Sharon (Dec 2)** ✅
- [x] Created EDA.ipynb
- [x] Analyzed dataset thoroughly

**Michael (Dec 3)** ✅
- [x] Created train/val/test splits
- [x] Applied SMOTE to training set

**San (Dec 4)** ✅
- [x] Created 02_logistic_regression.ipynb
- [x] Created 03_mlp_training.ipynb
- [x] Created 04_model_evaluation.ipynb
- [x] All models trained and evaluated

**VC (Dec 5)** 🎯 Next
- [ ] Run feature importance analysis
- [ ] Use `models/logistic_model.pkl`

**Olivia (Dec 6)** 🎯 Next
- [ ] Extract plots from 04_model_evaluation.ipynb
- [ ] Create report tables
- [ ] Organize figures for presentation

**San (Dec 7)** 🎯 Next
- [ ] DiCE setup and counterfactuals
- [ ] Use logistic model + preprocessor

---

**Questions?** Check PROGRESS.md in the project root for detailed documentation!
