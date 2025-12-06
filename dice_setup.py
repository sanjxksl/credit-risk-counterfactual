"""
DiCE Counterfactual Explanations Setup for Credit Risk Model

This script:
1. Loads the trained MLP model
2. Loads test data
3. Configures DiCE explainer
4. Generates counterfactuals for high-risk cases
5. Verifies if counterfactuals flip predictions
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import dice_ml
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===========================
# Configuration
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FILE = DATA_DIR / "test.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
MODEL_FILE = MODELS_DIR / "mlp_model.pth"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.txt"

DICE_RESULTS_DIR = RESULTS_DIR / "dice_counterfactuals"
DICE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "status"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===========================
# MLP Model Definition
# ===========================
class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class SimpleMLP(nn.Module):
    """ResNet-style MLP for credit risk prediction
    Architecture: Input → 512 (2x ResBlocks) → 256 (ResBlock) → 128 (ResBlock) → 64 → 1
    """
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()

        # Input layer: input_dim → 512
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Two residual blocks at 512 dimensions
        self.res_block1 = ResidualBlock(512, dropout=0.4)
        self.res_block2 = ResidualBlock(512, dropout=0.4)

        # Downsample: 512 → 256
        self.down1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # Residual block at 256 dimensions
        self.res_block3 = ResidualBlock(256, dropout=0.3)

        # Downsample: 256 → 128
        self.down2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Residual block at 128 dimensions
        self.res_block4 = ResidualBlock(128, dropout=0.2)

        # Final layer: 128 → 64 → 1
        self.down3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.down1(x)
        x = self.res_block3(x)
        x = self.down2(x)
        x = self.res_block4(x)
        x = self.down3(x)
        x = self.output_layer(x)
        return x


# ===========================
# PyTorch Model Wrapper for DiCE
# ===========================
class PyTorchModelWrapper:
    """Wrapper to make PyTorch model compatible with DiCE"""

    def __init__(self, pytorch_model, device='cpu'):
        self.model = pytorch_model
        self.device = device
        self.model.eval()

    def predict_proba(self, X):
        """
        Predict probability for DiCE compatibility
        X: numpy array or pandas DataFrame
        Returns: numpy array of shape (n_samples, 2) with probabilities [P(0), P(1)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            proba_1 = self.model(X_tensor).cpu().numpy().flatten()

        # DiCE expects probabilities for both classes
        proba_0 = 1 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X):
        """Binary predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


# ===========================
# Feature Configuration
# ===========================
def define_mutable_features():
    """
    Define which features can be changed (mutable) vs fixed (immutable)

    MUTABLE: Features that applicant can potentially change:
    - loan_amount: Can request different loan amount
    - income: Can increase income
    - dtir1: Debt-to-income ratio (can be improved)
    - credit_score: Can improve credit score
    - ltv: Loan-to-value ratio (can change down payment)
    - property_value: Can choose different property

    IMMUTABLE: Features that cannot be changed:
    - age_* categories: Cannot change age
    - gender_*: Cannot change gender
    - region_*: Typically fixed
    - year: Time-related, cannot change
    - credit_type_*: Historical credit bureau data
    """

    mutable = [
        'loan_amount',
        'income',
        'dtir1',
        'credit_score',
        'ltv',
        'property_value',
        'term',
    ]

    return mutable


# ===========================
# Load Components
# ===========================
def load_model():
    """Load trained MLP model"""
    print(f"Loading model from {MODEL_FILE}")
    checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)

    input_dim = checkpoint['input_dim']
    model = SimpleMLP(input_dim).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully (input_dim={input_dim})")
    return PyTorchModelWrapper(model, DEVICE)


def load_data():
    """Load test and training data"""
    print(f"Loading data from {DATA_DIR}")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")

    return train_df, test_df


def load_feature_names():
    """Load feature names"""
    with open(FEATURE_NAMES_FILE) as f:
        features = [line.strip() for line in f if line.strip()]
    return features


# ===========================
# DiCE Setup
# ===========================
def setup_dice_explainer(train_df, model_wrapper, mutable_features):
    """
    Configure DiCE explainer

    Args:
        train_df: Training data for reference distribution
        model_wrapper: Wrapped PyTorch model
        mutable_features: List of mutable feature names

    Returns:
        DiCE explainer object
    """
    print("\n" + "="*60)
    print("Setting up DiCE Explainer")
    print("="*60)

    # Prepare data for DiCE
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    # Create continuous features list (all features that aren't one-hot encoded)
    continuous_features = [
        'term', 'credit_score', 'ltv', 'dtir1', 'loan_amount',
        'income', 'property_value', 'year', 'loan_limit_cf', 'loan_limit_ncf'
    ]

    # Create DiCE data object
    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name=TARGET_COL
    )

    # Create DiCE model
    dice_model = dice_ml.Model(
        model=model_wrapper,
        backend='sklearn',
        model_type='classifier'
    )

    # Create DiCE explainer
    explainer = dice_ml.Dice(
        dice_data,
        dice_model,
        method='random'
    )

    print(f"DiCE explainer created successfully")
    print(f"Continuous features: {len(continuous_features)}")
    print(f"Mutable features: {len(mutable_features)}")
    print(f"Features permitted to vary: {mutable_features}")

    return explainer, dice_data


# ===========================
# Generate Counterfactuals
# ===========================
def find_high_risk_cases(test_df, model_wrapper, n_cases=3, threshold=0.7):
    """
    Find high-risk test cases (predicted default probability > threshold)

    Args:
        test_df: Test dataframe
        model_wrapper: Model wrapper
        n_cases: Number of cases to return
        threshold: Minimum probability threshold for high risk

    Returns:
        DataFrame with high-risk cases and their indices
    """
    print("\n" + "="*60)
    print(f"Finding high-risk cases (probability > {threshold})")
    print("="*60)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Get predictions
    proba = model_wrapper.predict_proba(X_test.values)[:, 1]

    # Find high-risk cases
    high_risk_mask = proba > threshold
    high_risk_indices = np.where(high_risk_mask)[0]

    print(f"Found {len(high_risk_indices)} high-risk cases")

    if len(high_risk_indices) < n_cases:
        print(f"Warning: Only {len(high_risk_indices)} cases found, adjusting n_cases")
        n_cases = len(high_risk_indices)

    # Select top n cases
    selected_indices = high_risk_indices[np.argsort(-proba[high_risk_indices])[:n_cases]]

    results = []
    for idx in selected_indices:
        results.append({
            'test_index': int(idx),
            'predicted_proba': float(proba[idx]),
            'true_label': int(y_test.iloc[idx])
        })
        print(f"Case {idx}: P(default)={proba[idx]:.4f}, True label={y_test.iloc[idx]}")

    return pd.DataFrame(results), selected_indices


def generate_counterfactuals(explainer, test_df, case_indices, mutable_features, desired_class=0, total_CFs=3):
    """
    Generate counterfactual explanations for selected cases

    Args:
        explainer: DiCE explainer
        test_df: Test dataframe
        case_indices: Indices of cases to explain
        mutable_features: List of mutable features
        desired_class: Target class (0 = no default)
        total_CFs: Number of counterfactuals per case

    Returns:
        List of counterfactual results
    """
    print("\n" + "="*60)
    print(f"Generating {total_CFs} counterfactuals per case")
    print("="*60)

    X_test = test_df.drop(columns=[TARGET_COL])

    all_results = []

    for idx in case_indices:
        print(f"\n--- Case {idx} ---")
        query_instance = X_test.iloc[[idx]]

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
                print(f"  No counterfactuals found for case {idx}")
                all_results.append({
                    'case_index': idx,
                    'counterfactuals': None,
                    'success': False
                })
            else:
                print(f"  Generated {len(cf_df)} counterfactuals")
                all_results.append({
                    'case_index': idx,
                    'counterfactuals': cf_df,
                    'original': query_instance,
                    'success': True
                })

        except Exception as e:
            print(f"  Error generating counterfactuals for case {idx}: {e}")
            all_results.append({
                'case_index': idx,
                'counterfactuals': None,
                'success': False,
                'error': str(e)
            })

    return all_results


def verify_counterfactuals(model_wrapper, cf_results):
    """
    Verify that counterfactuals actually flip the prediction

    Args:
        model_wrapper: Model wrapper
        cf_results: List of counterfactual results

    Returns:
        Verification summary
    """
    print("\n" + "="*60)
    print("Verifying Counterfactual Predictions")
    print("="*60)

    verification_summary = []

    for result in cf_results:
        if not result['success'] or result['counterfactuals'] is None:
            continue

        case_idx = result['case_index']
        original = result['original']
        cf_df = result['counterfactuals']

        # Get original prediction
        orig_proba = model_wrapper.predict_proba(original.values)[0, 1]
        orig_pred = int(orig_proba >= 0.5)

        print(f"\n--- Case {case_idx} ---")
        print(f"  Original: P(default)={orig_proba:.4f}, Prediction={orig_pred}")

        # Drop target column if present in counterfactuals
        cf_features = cf_df.drop(columns=[TARGET_COL], errors='ignore')

        # Get counterfactual predictions
        cf_probas = model_wrapper.predict_proba(cf_features.values)[:, 1]
        cf_preds = (cf_probas >= 0.5).astype(int)

        flipped = []
        for i, (cf_proba, cf_pred) in enumerate(zip(cf_probas, cf_preds)):
            is_flipped = cf_pred != orig_pred
            flipped.append(is_flipped)
            status = "✓ FLIPPED" if is_flipped else "✗ NOT FLIPPED"
            print(f"  CF {i+1}: P(default)={cf_proba:.4f}, Prediction={cf_pred} {status}")

        verification_summary.append({
            'case_index': case_idx,
            'original_proba': orig_proba,
            'original_pred': orig_pred,
            'num_counterfactuals': len(cf_probas),
            'num_flipped': sum(flipped),
            'flip_rate': sum(flipped) / len(flipped) if flipped else 0
        })

    return pd.DataFrame(verification_summary)


def save_results(high_risk_cases, cf_results, verification_summary):
    """Save counterfactual results to files"""
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    # Save high-risk cases
    high_risk_file = DICE_RESULTS_DIR / "high_risk_cases.csv"
    high_risk_cases.to_csv(high_risk_file, index=False)
    print(f"Saved high-risk cases to {high_risk_file}")

    # Save verification summary
    verification_file = DICE_RESULTS_DIR / "verification_summary.csv"
    verification_summary.to_csv(verification_file, index=False)
    print(f"Saved verification summary to {verification_file}")

    # Save individual counterfactuals
    for i, result in enumerate(cf_results):
        if result['success'] and result['counterfactuals'] is not None:
            case_idx = result['case_index']
            cf_file = DICE_RESULTS_DIR / f"counterfactuals_case_{case_idx}.csv"
            result['counterfactuals'].to_csv(cf_file, index=False)
            print(f"Saved counterfactuals for case {case_idx} to {cf_file}")


# ===========================
# Main Execution
# ===========================
def main():
    """Main execution pipeline"""
    print("\n" + "="*60)
    print("DiCE COUNTERFACTUAL EXPLANATIONS - CREDIT RISK MODEL")
    print("="*60)

    # Load model
    model_wrapper = load_model()

    # Load data
    train_df, test_df = load_data()

    # Define mutable features
    mutable_features = define_mutable_features()

    # Setup DiCE explainer
    explainer, dice_data = setup_dice_explainer(train_df, model_wrapper, mutable_features)

    # Find high-risk cases
    high_risk_cases, case_indices = find_high_risk_cases(
        test_df,
        model_wrapper,
        n_cases=3,
        threshold=0.7
    )

    # Generate counterfactuals
    cf_results = generate_counterfactuals(
        explainer,
        test_df,
        case_indices,
        mutable_features,
        desired_class=0,
        total_CFs=3
    )

    # Verify counterfactuals flip predictions
    verification_summary = verify_counterfactuals(model_wrapper, cf_results)

    # Save results
    save_results(high_risk_cases, cf_results, verification_summary)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"High-risk cases analyzed: {len(high_risk_cases)}")
    print(f"Counterfactuals generated: {sum(1 for r in cf_results if r['success'])}")
    if len(verification_summary) > 0:
        avg_flip_rate = verification_summary['flip_rate'].mean()
        print(f"Average flip rate: {avg_flip_rate:.2%}")
        print(f"\nVerification Summary:")
        print(verification_summary.to_string(index=False))

    print(f"\nResults saved to: {DICE_RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
