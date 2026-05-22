# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Credit Risk — MLflow Training (MLP + XGBoost)
# MAGIC
# MAGIC Trains both classifiers and logs every run to Databricks MLflow:
# MAGIC - Hyperparameters, metrics at each epoch (MLP), final test metrics
# MAGIC - Trained model artifacts (pytorch + sklearn calibrator, xgboost + calibrator)
# MAGIC - Tags for easy filtering in the Experiments UI
# MAGIC
# MAGIC **Before running:**
# MAGIC 1. Create a Unity Catalog Volume and upload files:
# MAGIC    - Catalog → `workspace` → Create Schema → `credit_risk`
# MAGIC    - Inside `credit_risk` → Create Volume → `data`
# MAGIC    - Upload: `train.csv`, `val.csv`, `test.csv`, `training_meta.json`
# MAGIC 2. Install cluster libraries (Compute → your cluster → Libraries → Install New):
# MAGIC    - PyPI: `torch` (CPU wheel), `xgboost`, `scikit-learn==1.3.2`

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

import json
import os
import tempfile

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# DBFS paths — data uploaded in prerequisite step
TRAIN_PATH = "/Volumes/workspace/credit_risk/data/train.csv"
VAL_PATH   = "/Volumes/workspace/credit_risk/data/val.csv"
TEST_PATH  = "/Volumes/workspace/credit_risk/data/test.csv"
META_PATH  = "/Volumes/workspace/credit_risk/data/training_meta.json"

TARGET_COL   = "status"
RANDOM_STATE = 42
DEVICE = torch.device("cpu")  # Community Edition clusters are CPU-only

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

mlflow.set_experiment("/credit-risk-experiments")
print("MLflow tracking URI:", mlflow.get_tracking_uri())

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[TARGET_COL]).values
y_train = train_df[TARGET_COL].values
X_val   = val_df.drop(columns=[TARGET_COL]).values
y_val   = val_df[TARGET_COL].values
X_test  = test_df.drop(columns=[TARGET_COL]).values
y_test  = test_df[TARGET_COL].values

with open(META_PATH) as f:
    meta = json.load(f)
pos_weight = torch.tensor([meta["pos_weight"]])

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"pos_weight: {pos_weight.item():.4f}")

# COMMAND ----------

# MAGIC %md ## Model Definition

# COMMAND ----------

class CreditMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def platt_calibrate(raw_proba, y_true):
    cal = LogisticRegression()
    cal.fit(raw_proba.reshape(-1, 1), y_true)
    return cal, cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


def compute_metrics(y_true, proba):
    return {
        "auc_roc": roc_auc_score(y_true, proba),
        "auc_pr":  average_precision_score(y_true, proba),
        "brier":   brier_score_loss(y_true, proba),
        "pred_rate": float(proba.mean()),
    }

# COMMAND ----------

# MAGIC %md ## Train MLP with MLflow

# COMMAND ----------

BATCH_SIZE    = 128
MAX_EPOCHS    = 100
LEARNING_RATE = 0.001
PATIENCE      = 15
WEIGHT_DECAY  = 1e-4

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_tensor   = torch.FloatTensor(X_val)
test_tensor  = torch.FloatTensor(X_test)

with mlflow.start_run(run_name="mlp-credit-risk") as run:
    mlflow.set_tags({
        "model_type": "MLP",
        "calibration": "platt_scaling",
        "imbalance_strategy": "pos_weight",
        "features": "56",
        "dataset": "148670_loans",
    })
    mlflow.log_params({
        "architecture": "56->128->64->32->1",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "pos_weight": pos_weight.item(),
        "dropout": "0.3/0.2/0.1",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
    })

    model     = CreditMLP(X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_auc     = 0.0
    patience_counter = 0
    best_state       = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # train
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b.to(DEVICE)), y_b.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)
        epoch_loss /= len(X_train)

        # validate
        model.eval()
        with torch.no_grad():
            val_proba = torch.sigmoid(model(val_tensor.to(DEVICE))).cpu().numpy().flatten()
        val_auc = roc_auc_score(y_val, val_proba)
        scheduler.step(val_auc)

        # log per-epoch metrics to MLflow
        mlflow.log_metrics({
            "train_loss": epoch_loss,
            "val_auc_roc": val_auc,
        }, step=epoch)

        # early stopping
        if val_auc > best_val_auc:
            best_val_auc     = val_auc
            patience_counter = 0
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}  best val AUC: {best_val_auc:.4f}")
                break

    mlflow.log_param("epochs_trained", epoch)

    # restore best weights and get final probabilities
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_proba_raw  = torch.sigmoid(model(val_tensor.to(DEVICE))).cpu().numpy().flatten()
        test_proba_raw = torch.sigmoid(model(test_tensor.to(DEVICE))).cpu().numpy().flatten()

    # Platt calibration
    calibrator, val_proba_cal  = platt_calibrate(val_proba_raw, y_val)
    test_proba_cal             = calibrator.predict_proba(test_proba_raw.reshape(-1, 1))[:, 1]

    # log final metrics (uncalibrated + calibrated)
    val_raw_m  = compute_metrics(y_val,  val_proba_raw)
    test_raw_m = compute_metrics(y_test, test_proba_raw)
    val_cal_m  = compute_metrics(y_val,  val_proba_cal)
    test_cal_m = compute_metrics(y_test, test_proba_cal)

    mlflow.log_metrics({
        "val_auc_roc_raw":    val_raw_m["auc_roc"],
        "val_auc_pr_raw":     val_raw_m["auc_pr"],
        "val_brier_raw":      val_raw_m["brier"],
        "test_auc_roc_raw":   test_raw_m["auc_roc"],
        "test_auc_pr_raw":    test_raw_m["auc_pr"],
        "test_brier_raw":     test_raw_m["brier"],
        "val_auc_roc_cal":    val_cal_m["auc_roc"],
        "val_auc_pr_cal":     val_cal_m["auc_pr"],
        "val_brier_cal":      val_cal_m["brier"],
        "test_auc_roc_cal":   test_cal_m["auc_roc"],
        "test_auc_pr_cal":    test_cal_m["auc_pr"],
        "test_brier_cal":     test_cal_m["brier"],
        "calibration_gap_pct": abs(test_cal_m["pred_rate"] - y_test.mean()) * 100,
    })

    # log model artifacts
    mlflow.pytorch.log_model(model, "mlp_model")
    mlflow.sklearn.log_model(calibrator, "mlp_calibrator")

    print(f"\nMLP run ID: {run.info.run_id}")
    print(f"Test AUC-ROC (calibrated): {test_cal_m['auc_roc']:.4f}")
    print(f"Test AUC-PR  (calibrated): {test_cal_m['auc_pr']:.4f}")
    print(f"Test Brier   (calibrated): {test_cal_m['brier']:.4f}")

# COMMAND ----------

# MAGIC %md ## Train XGBoost with MLflow

# COMMAND ----------

scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

xgb_params = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         5,
    "min_child_weight":  1,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "gamma":             0.0,
    "reg_lambda":        1.0,
    "reg_alpha":         0.0,
    "scale_pos_weight":  scale_pos_weight,
    "early_stopping_rounds": 30,
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "tree_method":       "hist",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}

with mlflow.start_run(run_name="xgboost-credit-risk") as run:
    mlflow.set_tags({
        "model_type": "XGBoost",
        "calibration": "platt_scaling",
        "imbalance_strategy": "scale_pos_weight",
        "features": "56",
        "dataset": "148670_loans",
    })
    mlflow.log_params(xgb_params)

    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_proba_raw  = xgb_model.predict_proba(X_val)[:, 1]
    test_proba_raw = xgb_model.predict_proba(X_test)[:, 1]

    calibrator_xgb, val_proba_cal = platt_calibrate(val_proba_raw, y_val)
    test_proba_cal = calibrator_xgb.predict_proba(test_proba_raw.reshape(-1, 1))[:, 1]

    val_raw_m  = compute_metrics(y_val,  val_proba_raw)
    test_raw_m = compute_metrics(y_test, test_proba_raw)
    val_cal_m  = compute_metrics(y_val,  val_proba_cal)
    test_cal_m = compute_metrics(y_test, test_proba_cal)

    mlflow.log_metrics({
        "n_estimators_used":  xgb_model.best_iteration + 1,
        "val_auc_roc_raw":    val_raw_m["auc_roc"],
        "val_auc_pr_raw":     val_raw_m["auc_pr"],
        "val_brier_raw":      val_raw_m["brier"],
        "test_auc_roc_raw":   test_raw_m["auc_roc"],
        "test_auc_pr_raw":    test_raw_m["auc_pr"],
        "test_brier_raw":     test_raw_m["brier"],
        "val_auc_roc_cal":    val_cal_m["auc_roc"],
        "val_auc_pr_cal":     val_cal_m["auc_pr"],
        "val_brier_cal":      val_cal_m["brier"],
        "test_auc_roc_cal":   test_cal_m["auc_roc"],
        "test_auc_pr_cal":    test_cal_m["auc_pr"],
        "test_brier_cal":     test_cal_m["brier"],
        "calibration_gap_pct": abs(test_cal_m["pred_rate"] - y_test.mean()) * 100,
    })

    mlflow.xgboost.log_model(xgb_model, "xgboost_model")
    mlflow.sklearn.log_model(calibrator_xgb, "xgboost_calibrator")

    print(f"\nXGBoost run ID: {run.info.run_id}")
    print(f"Estimators used: {xgb_model.best_iteration + 1} / {xgb_params['n_estimators']}")
    print(f"Test AUC-ROC (calibrated): {test_cal_m['auc_roc']:.4f}")
    print(f"Test AUC-PR  (calibrated): {test_cal_m['auc_pr']:.4f}")
    print(f"Test Brier   (calibrated): {test_cal_m['brier']:.4f}")

# COMMAND ----------

# MAGIC %md ## Compare Runs in the MLflow UI
# MAGIC
# MAGIC Click **Experiments** in the left sidebar → select `/credit-risk-experiments`.
# MAGIC
# MAGIC The table shows both runs side-by-side. Click **Chart** to plot metrics.
# MAGIC Useful comparisons:
# MAGIC - `test_auc_roc_cal` and `test_auc_pr_cal` — primary model quality
# MAGIC - `test_brier_cal` — calibration quality (lower is better)
# MAGIC - `calibration_gap_pct` — how far predicted default rate drifts from true 24.6%
# MAGIC
# MAGIC The MLP run also has per-epoch `val_auc_roc` logged as a time series —
# MAGIC click the run → **Metrics → val_auc_roc** to see the learning curve.

# COMMAND ----------

# MAGIC %md ## Register the Best Model
# MAGIC
# MAGIC Once you've compared both runs, register the winner in the Model Registry.
# MAGIC Replace `<RUN_ID>` with the run ID printed above.

# COMMAND ----------

# Replace with the run ID of whichever model performed best
BEST_RUN_ID    = "<RUN_ID>"
BEST_MODEL_KEY = "mlp_model"   # or "xgboost_model"

model_uri = f"runs:/{BEST_RUN_ID}/{BEST_MODEL_KEY}"

registered = mlflow.register_model(
    model_uri=model_uri,
    name="credit-risk-default-predictor",
)
print(f"Registered: {registered.name}  version {registered.version}")
