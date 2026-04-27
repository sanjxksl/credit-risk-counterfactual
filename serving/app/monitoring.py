from __future__ import annotations

"""
Monitoring layer: data drift detection and retraining triggers.

What is data drift?
-------------------
Your model was trained on 2019 mortgage applications. If in production you
start seeing applications with very different loan amounts, income levels, or
DTI ratios, the model's learned patterns may no longer apply — its predictions
become unreliable even if its code hasn't changed. Drift detection catches this
before you notice degraded performance in production.

How it works here:
  1. Every prediction request logs the input features to an in-memory buffer
     (in production this would go to S3 or a database).
  2. When you hit GET /monitoring/drift, Evidently compares the buffered
     requests against the training data distribution.
  3. The retraining trigger evaluates the drift report against defined
     thresholds and returns a recommendation.

In a real AWS setup, you would:
  - Log to S3 via boto3
  - Run the drift check on a schedule (CloudWatch Events → Lambda)
  - Trigger a SageMaker Training Job automatically if retrain_recommended=True
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/opt/ml/model"))

# How many requests to buffer before drift analysis is meaningful
MIN_REQUESTS_FOR_DRIFT = 100

# --- Retraining thresholds (tune these based on business tolerance) ---
# If more than 30% of features are drifting, the data distribution has shifted
MAX_DRIFT_SHARE = 0.30
# If the live predicted default rate is more than 5pp away from training baseline
MAX_RATE_SHIFT = 0.05
# Training baseline default rate
REFERENCE_DEFAULT_RATE = 0.246


@dataclass
class RequestLog:
    """Accumulates incoming prediction requests for batch drift analysis."""

    _buffer: list[dict] = field(default_factory=list)
    _predictions: list[float] = field(default_factory=list)

    def log(self, features: dict, predicted_probability: float) -> None:
        self._buffer.append(features)
        self._predictions.append(predicted_probability)

    def clear(self) -> None:
        self._buffer.clear()
        self._predictions.clear()

    @property
    def count(self) -> int:
        return len(self._buffer)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._buffer)
        df["predicted_probability"] = self._predictions
        return df


class DriftMonitor:
    """
    Runs Evidently drift reports comparing live traffic to the training reference.

    Evidently computes a statistical test (Jensen-Shannon divergence for
    numerical features, chi-squared for categorical) for each feature and
    marks it as drifted if p-value < 0.05 or the distance exceeds a threshold.
    """

    def __init__(self, reference_path: Path):
        # Load a sample of training data as the reference distribution.
        # Evidently needs raw (pre-StandardScaler) feature values — use the
        # processed train split since that's what the model sees.
        self._reference = pd.read_csv(reference_path).sample(
            n=min(5000, len(pd.read_csv(reference_path))), random_state=42
        )
        self._reference_default_rate = float(self._reference["status"].mean())

    def run(self, current: pd.DataFrame) -> dict[str, Any]:
        """Compare current batch against training reference, return summary dict."""

        # Evidently's DataDriftPreset checks every feature column automatically
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self._reference.drop(columns=["status"], errors="ignore"),
            current_data=current.drop(
                columns=["status", "predicted_probability"], errors="ignore"
            ),
        )

        result = report.as_dict()
        drift_summary = result["metrics"][0]["result"]

        drifted_features = [
            col
            for col, stats in drift_summary.get("drift_by_columns", {}).items()
            if stats.get("drift_detected", False)
        ]
        total_features = drift_summary.get("number_of_columns", 1)
        share_drifted = len(drifted_features) / total_features

        predicted_rate = float(current["predicted_probability"].mean())

        return {
            "num_requests_analysed": len(current),
            "share_drifted_features": round(share_drifted, 3),
            "drifted_features": drifted_features,
            "predicted_default_rate": round(predicted_rate, 4),
            "reference_default_rate": round(self._reference_default_rate, 4),
        }


class RetrainingTrigger:
    """
    Evaluates a drift report and decides whether to recommend retraining.

    Conditions (any one is sufficient):
      1. More than 30% of features are statistically drifted.
      2. The live predicted default rate deviates by more than 5pp from the
         training baseline — suggests the model is scoring a fundamentally
         different population.

    What happens when retrain_recommended=True in production:
      - A Lambda function calls boto3.client('sagemaker').create_training_job()
      - SageMaker pulls the latest data from S3, re-runs mlp_training and
        xgboost_training, and pushes new model artifacts to /opt/ml/model/
      - A new endpoint version is deployed via blue/green swap
    """

    def evaluate(self, drift_result: dict) -> dict[str, Any]:
        reasons = []

        if drift_result["share_drifted_features"] > MAX_DRIFT_SHARE:
            reasons.append(
                f"{drift_result['share_drifted_features']:.0%} of features drifted "
                f"(threshold: {MAX_DRIFT_SHARE:.0%})"
            )

        rate_shift = abs(
            drift_result["predicted_default_rate"] - REFERENCE_DEFAULT_RATE
        )
        if rate_shift > MAX_RATE_SHIFT:
            reasons.append(
                f"Predicted default rate shifted by {rate_shift:.1%} "
                f"(threshold: {MAX_RATE_SHIFT:.0%})"
            )

        return {
            **drift_result,
            "retrain_recommended": len(reasons) > 0,
            "trigger_reasons": reasons,
        }
