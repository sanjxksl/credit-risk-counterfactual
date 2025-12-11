import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier


def evaluate_split(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Compute core metrics and print a quick classification report."""
    preds = (proba >= 0.5).astype(int)
    auc_roc = roc_auc_score(y_true, proba)
    auc_pr = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    positive_rate = y_true.mean()

    print(f"\n{name} PERFORMANCE (Uncalibrated)")
    print("=" * 80)
    print(f"AUC-ROC:     {auc_roc:.4f}")
    print(f"AUC-PR:      {auc_pr:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Positives:   {positive_rate:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, preds))

    return {
        "dataset": name,
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "brier_score": float(brier),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    results_dir = project_root / "results"

    train_file = data_dir / "train.csv"
    val_file = data_dir / "val.csv"
    test_file = data_dir / "test.csv"

    model_file = models_dir / "xgboost_model.json"
    calibrator_file = models_dir / "xgboost_calibrator.pkl"
    predictions_file = results_dir / "xgboost_predictions.csv"
    metrics_file = results_dir / "xgboost_metrics.json"

    target_col = "status"
    random_state = 42

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=[target_col]).values
    y_train = train_df[target_col].values

    X_val = val_df.drop(columns=[target_col]).values
    y_val = val_df[target_col].values

    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Class balance (train): {y_train.mean():.1%} default rate")

    # Define model
    xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": os.cpu_count(),
    }

    model = XGBClassifier(**xgb_params)

    print("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
        early_stopping_rounds=30,
    )

    best_iteration = getattr(model, "best_iteration", xgb_params["n_estimators"])
    print(f"Best iteration: {best_iteration}")

    # Evaluate
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_split("Validation", y_val, val_proba)
    test_metrics = evaluate_split("Test", y_test, test_proba)

    # Calibrate with Platt scaling on validation predictions
    calibrator = LogisticRegression()
    calibrator.fit(val_proba.reshape(-1, 1), y_val)

    val_proba_cal = calibrator.predict_proba(val_proba.reshape(-1, 1))[:, 1]
    test_proba_cal = calibrator.predict_proba(test_proba.reshape(-1, 1))[:, 1]

    val_calibration = {
        "brier_score_calibrated": brier_score_loss(y_val, val_proba_cal),
        "calibration_gap_uncalibrated": abs(y_val.mean() - val_proba.mean()),
        "calibration_gap_calibrated": abs(y_val.mean() - val_proba_cal.mean()),
    }

    test_calibration = {
        "brier_score_calibrated": brier_score_loss(y_test, test_proba_cal),
        "calibration_gap_uncalibrated": abs(y_test.mean() - test_proba.mean()),
        "calibration_gap_calibrated": abs(y_test.mean() - test_proba_cal.mean()),
    }

    print("\nCalibration analysis complete.")

    # Persist artifacts
    model.save_model(model_file)
    joblib.dump(calibrator, calibrator_file)
    print(f"Model saved to {model_file}")
    print(f"Calibrator saved to {calibrator_file}")

    predictions_df = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted_probability": test_proba,
            "predicted_probability_calibrated": test_proba_cal,
            "predicted_label": (test_proba >= 0.5).astype(int),
            "predicted_label_calibrated": (test_proba_cal >= 0.5).astype(int),
        }
    )
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

    metrics = {
        "model": "XGBoost",
        "hyperparameters": {**xgb_params, "best_iteration": int(best_iteration)},
        "calibration": "Platt Scaling on validation set",
        "validation_metrics": {**val_metrics, **val_calibration},
        "test_metrics": {**test_metrics, **test_calibration},
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    print("\n" + "=" * 80)
    print("XGBoost training complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
