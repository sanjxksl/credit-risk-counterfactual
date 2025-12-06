# DiCE Counterfactual Explanations - Quick Start

## Installation Complete ✓

DiCE library (dice-ml) is installed and ready to use.

## Run DiCE

```bash
python dice_setup.py
```

## What It Does

1. Loads the best MLP model (89.6% AUC-ROC)
2. Finds high-risk test cases (P(default) > 0.7)
3. Generates 3 counterfactual scenarios per case
4. Verifies counterfactuals flip predictions
5. Saves results to `results/dice_counterfactuals/`

## Results Location

```
results/dice_counterfactuals/
├── high_risk_cases.csv           # Selected high-risk cases
├── verification_summary.csv       # Flip rate metrics
└── counterfactuals_case_*.csv     # Counterfactual features
```

## Test Run Results

- **Cases analyzed**: 3 high-risk cases
- **Counterfactuals generated**: 1 successful case (14123)
- **Flip rate**: 100% (all 3 counterfactuals flipped prediction)
- **Original**: P(default) = 99.99% → Prediction = 1 (default)
- **Counterfactuals**: P(default) = 16-35% → Prediction = 0 (no default)

## Key Configuration

### Mutable Features (user can change)
- loan_amount
- income
- dtir1 (debt-to-income ratio)
- credit_score
- ltv (loan-to-value ratio)
- property_value
- term

### Immutable Features (fixed)
- Age, gender, region
- Credit history type
- Year

## Common Issues

### "No counterfactuals found"
- Try lowering risk threshold from 0.7 to 0.5
- Increase total_CFs from 3 to 5
- Very high-risk cases are harder to flip

### Check predictions
Run verification summary to see flip rates:
```bash
cat results/dice_counterfactuals/verification_summary.csv
```

## Full Documentation

See `docs/dice_counterfactual_guide.md` for:
- Detailed configuration options
- Advanced usage examples
- Troubleshooting guide
- Integration with project workflow
- Feature customization

## Next Steps

1. Run on more cases: Lower threshold or increase n_cases
2. Analyze feature changes: Compare original vs counterfactual
3. Build dashboard: Create interactive UI for exploration
4. Production integration: Batch process rejected applications
