"""
DiCE counterfactual generation for the serving layer.

DiCE finds the smallest set of feature changes that would flip the model's
prediction from default (1) to no-default (0). Only the five mutable loan
structure features can be changed — demographic and financial profile features
are held fixed because an applicant can't instantly change their credit score
or income.

Latency note: DiCE with method='random' typically takes 1-3 seconds per case.
The /predict/explain endpoint should be called separately from /predict when
the latency is acceptable (e.g. a loan officer reviewing a declined application,
not a real-time pre-screen).
"""

import json
import os
from pathlib import Path
from typing import Optional

import dice_ml
import numpy as np
import pandas as pd
import torch

from .pipeline import CreditMLP, InferencePipeline

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))

MUTABLE_FEATURES = ["loan_amount", "property_value", "ltv", "term", "dtir1"]

CONTINUOUS_FEATURES = [
    "term", "credit_score", "ltv", "dtir1", "loan_amount",
    "income", "property_value", "year", "loan_limit_cf", "loan_limit_ncf",
]


class _MLPWrapper:
    """Thin wrapper so DiCE can call predict_proba on the preprocessed feature space."""

    def __init__(self, pipeline: InferencePipeline):
        self.pipeline = pipeline

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.pipeline.model(X_tensor)
            proba_1 = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.column_stack([1 - proba_1, proba_1])


class CounterfactualEngine:
    """Wraps DiCE and formats its output into human-readable recommendations."""

    def __init__(self, pipeline: InferencePipeline, train_csv_path: Path):
        self._pipeline = pipeline
        self._wrapper = _MLPWrapper(pipeline)

        train_df = pd.read_csv(train_csv_path)

        dice_data = dice_ml.Data(
            dataframe=train_df,
            continuous_features=CONTINUOUS_FEATURES,
            outcome_name="status",
        )
        dice_model = dice_ml.Model(
            model=self._wrapper,
            backend="sklearn",
            model_type="classifier",
        )
        self._explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    def generate(
        self, preprocessed_row: pd.DataFrame, n: int = 3
    ) -> Optional[list[dict]]:
        """
        preprocessed_row: single-row DataFrame already through pipeline.preprocess()
        Returns a list of counterfactual dicts, or None if DiCE finds nothing.
        """
        try:
            result = self._explainer.generate_counterfactuals(
                preprocessed_row,
                total_CFs=n,
                desired_class=0,
                features_to_vary=MUTABLE_FEATURES,
            )
            cf_df = result.cf_examples_list[0].final_cfs_df
        except Exception:
            return None

        if cf_df is None or len(cf_df) == 0:
            return None

        original = preprocessed_row.iloc[0]
        scenarios = []
        for _, cf in cf_df.iterrows():
            changes = {}
            descriptions = []
            for feat in MUTABLE_FEATURES:
                orig_val = original[feat]
                cf_val = cf[feat]
                if abs(orig_val - cf_val) > 1e-4:
                    changes[feat] = round(float(cf_val), 2)
                    descriptions.append(_describe_change(feat, orig_val, cf_val))

            cf_features = cf.drop("status", errors="ignore").values.reshape(1, -1)
            cf_proba = float(self._wrapper.predict_proba(cf_features)[0, 1])
            cal_proba = float(
                self._pipeline.calibrator.predict_proba([[cf_proba]])[0, 1]
            )

            scenarios.append({
                "changes": changes,
                "predicted_probability_after": round(cal_proba, 4),
                "description": "; ".join(descriptions) if descriptions else "No changes needed",
            })

        return scenarios


def _describe_change(feature: str, original: float, counterfactual: float) -> str:
    delta = counterfactual - original
    direction = "increase" if delta > 0 else "reduce"

    labels = {
        "loan_amount": ("Loan amount", f"${abs(delta):,.0f}"),
        "property_value": ("Property value", f"${abs(delta):,.0f}"),
        "ltv": ("LTV", f"{abs(delta):.1f} percentage points"),
        "term": ("Term", f"{abs(delta):.0f} months"),
        "dtir1": ("Debt-to-income ratio", f"{abs(delta):.1f} percentage points"),
    }
    label, magnitude = labels.get(feature, (feature, str(abs(round(delta, 2)))))
    return f"{direction.capitalize()} {label} by {magnitude}"
