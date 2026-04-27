from __future__ import annotations

"""
Loads all model artifacts once at startup, then handles per-request inference.

Why load once at startup? Loading a model from disk takes ~200ms. If you
reloaded it on every request, a busy endpoint would be unbearably slow.
FastAPI's lifespan context (used in main.py) calls load() once when the
server starts and keeps the objects in memory for every request.

Preprocessing chain (must exactly match feature_engineering.ipynb):
  1. Winsorize loan_amount / income / property_value  (raw_winsorize_bounds.json)
  2. Log-transform those three features              (np.log1p)
  3. Winsorize ltv / dtir1                           (winsorize_bounds.json)
  4. Derive is_standard_term                         (term == 360)
  5. ColumnTransformer: StandardScaler + OHE         (preprocessor.pkl)
  6. MLP forward pass + sigmoid                      (mlp_model.pth)
  7. Platt scaling calibration                       (calibrator.pkl)
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))


class CreditMLP(nn.Module):
    """Exact architecture used in mlp_training.ipynb. Must match saved weights."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class InferencePipeline:
    def __init__(self):
        self.preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        self.calibrator = joblib.load(MODEL_DIR / "calibrator.pkl")

        checkpoint = torch.load(
            MODEL_DIR / "mlp_model.pth", map_location="cpu", weights_only=False
        )
        self.model = CreditMLP(checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        with open(MODEL_DIR / "winsorize_bounds.json") as f:
            self.winsorize_bounds = json.load(f)          # ltv, dtir1

        with open(MODEL_DIR / "raw_winsorize_bounds.json") as f:
            self.raw_winsorize_bounds = json.load(f)      # loan_amount, income, property_value

    def preprocess(self, raw: dict) -> np.ndarray:
        df = pd.DataFrame([raw])

        # Step 1 — winsorize raw monetary/income features before log transform
        for col, (lo, hi) in self.raw_winsorize_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        # Step 2 — log-transform (same as data_cleaning.ipynb used np.log1p)
        for col in ["loan_amount", "income", "property_value"]:
            df[col] = np.log1p(df[col])

        # Step 3 — winsorize ltv and dtir1
        for col, (lo, hi) in self.winsorize_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        # Step 4 — derive binary term indicator
        df["is_standard_term"] = (df["term"] == 360).astype(int)

        # Step 5 — StandardScaler + OneHotEncoder (fitted on training data)
        return self.preprocessor.transform(df)

    def predict(self, raw: dict) -> tuple[float, int]:
        """Returns (calibrated_probability, predicted_label)."""
        X = self.preprocess(raw)
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            logit = self.model(X_tensor)
            raw_proba = torch.sigmoid(logit).item()

        cal_proba = float(self.calibrator.predict_proba([[raw_proba]])[0, 1])
        label = int(cal_proba >= 0.5)
        return cal_proba, label

    def risk_level(self, probability: float) -> str:
        if probability < 0.30:
            return "Low"
        elif probability < 0.70:
            return "Medium"
        return "High"
