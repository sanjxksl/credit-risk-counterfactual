"""
Generate DiCE counterfactual explanations for the 10 pre-selected high-risk cases
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path
import pandas as pd
import dice_ml
import joblib
import numpy as np
import torch
import torch.nn as nn

# Use the model classes and functions from dice_setup.py
from dice_setup import (
    SimpleMLP, PyTorchModelWrapper,
    define_mutable_features,
    setup_dice_explainer,
    verify_counterfactuals,
    PROJECT_ROOT, MODELS_DIR, DATA_DIR, RESULTS_DIR, DEVICE, TARGET_COL
)

def load_selected_cases():
    """Load the 10 pre-selected high-risk cases"""
    high_risk_file = DATA_DIR / "high_risk_cases.csv"
    df = pd.read_csv(high_risk_file)

    # Extract case IDs
    case_ids = df['case_id'].tolist()
    print(f"Loaded {len(case_ids)} pre-selected cases: {case_ids}")
    return case_ids

def generate_counterfactuals_for_selected_cases(
    explainer, test_df, case_ids, mutable_features,
    desired_class=0, total_CFs=5
):
    """
    Generate counterfactual explanations for pre-selected cases
    """
    print("\n" + "="*70)
    print(f"Generating {total_CFs} counterfactuals for {len(case_ids)} cases")
    print("="*70)

    X_test = test_df.drop(columns=[TARGET_COL])
    all_results = []

    for case_id in case_ids:
        print(f"\n--- Case {case_id} ---")

        # Get the case from test set
        if case_id >= len(X_test):
            print(f"  ERROR: Case ID {case_id} out of range (test set has {len(X_test)} samples)")
            all_results.append({
                'case_index': case_id,
                'counterfactuals': None,
                'success': False,
                'error': 'Case ID out of range'
            })
            continue

        query_instance = X_test.iloc[[case_id]]

        try:
            # Generate counterfactuals
            dice_exp = explainer.generate_counterfactuals(
                query_instance,
                total_CFs=total_CFs,
                desired_class=desired_class,
                features_to_vary=mutable_features
            )

            # Get counterfactual dataframe
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df

            if cf_df is None or len(cf_df) == 0:
                print(f"  No counterfactuals found for case {case_id}")
                all_results.append({
                    'case_index': case_id,
                    'counterfactuals': None,
                    'success': False,
                    'original': query_instance
                })
            else:
                print(f"  Generated {len(cf_df)} counterfactuals")
                all_results.append({
                    'case_index': case_id,
                    'counterfactuals': cf_df,
                    'original': query_instance,
                    'success': True
                })

        except Exception as e:
            print(f"  Error generating counterfactuals for case {case_id}: {e}")
            all_results.append({
                'case_index': case_id,
                'counterfactuals': None,
                'success': False,
                'error': str(e),
                'original': query_instance if 'query_instance' in locals() else None
            })

    return all_results

def save_results(cf_results, verification_summary):
    """Save counterfactual results to files"""
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)

    dice_results_dir = RESULTS_DIR / "dice_counterfactuals"
    dice_results_dir.mkdir(parents=True, exist_ok=True)

    # Save verification summary
    verification_file = dice_results_dir / "verification_summary.csv"
    verification_summary.to_csv(verification_file, index=False)
    print(f"Saved verification summary to {verification_file}")

    # Save individual counterfactuals
    for result in cf_results:
        if result['success'] and result['counterfactuals'] is not None:
            case_idx = result['case_index']
            cf_file = dice_results_dir / f"counterfactuals_case_{case_idx}.csv"
            result['counterfactuals'].to_csv(cf_file, index=False)
            print(f"Saved counterfactuals for case {case_idx} to {cf_file}")

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("COUNTERFACTUALS FOR 10 SELECTED HIGH-RISK CASES")
    print("="*70)

    # Load model
    print(f"\nLoading model from {MODELS_DIR / 'mlp_model.pth'}")
    checkpoint = torch.load(MODELS_DIR / 'mlp_model.pth', map_location=DEVICE)
    input_dim = checkpoint['input_dim']
    model = SimpleMLP(input_dim).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_wrapper = PyTorchModelWrapper(model, DEVICE)
    print(f"Model loaded successfully (input_dim={input_dim})")

    # Load data
    print(f"\nLoading data from {DATA_DIR}")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")

    # Load selected case IDs
    case_ids = load_selected_cases()

    # Define mutable features
    mutable_features = define_mutable_features()

    # Setup DiCE explainer
    explainer, dice_data = setup_dice_explainer(
        train_df, model_wrapper, mutable_features
    )

    # Generate counterfactuals for selected cases
    cf_results = generate_counterfactuals_for_selected_cases(
        explainer,
        test_df,
        case_ids,
        mutable_features,
        desired_class=0,
        total_CFs=5
    )

    # Verify counterfactuals flip predictions
    verification_summary = verify_counterfactuals(model_wrapper, cf_results)

    # Save results
    save_results(cf_results, verification_summary)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Cases processed: {len(case_ids)}")
    print(f"Counterfactuals generated: {sum(1 for r in cf_results if r['success'])}")
    print(f"Failed: {sum(1 for r in cf_results if not r['success'])}")

    if len(verification_summary) > 0:
        avg_flip_rate = verification_summary['flip_rate'].mean()
        print(f"Average flip rate: {avg_flip_rate:.2%}")
        print(f"\nVerification Summary:")
        print(verification_summary.to_string(index=False))

    print(f"\nResults saved to: {RESULTS_DIR / 'dice_counterfactuals'}")
    print("="*70)

if __name__ == "__main__":
    main()
