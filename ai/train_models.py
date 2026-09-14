"""
ArogyaFlow AI
B4 - ML Model Training Pipeline

Models:
1. Random Forest Regressor  -> Next-day demand forecasting
2. Random Forest Classifier -> 7-day stockout prediction
3. Isolation Forest         -> Demand anomaly detection

Development dataset:
500 PHCs x 180 days x 8 medicines

Important:
- Chronological train/test split
- No target leakage
- Categorical encoding learned from training data
- Class balancing for stockout prediction
- Models and metrics saved for later Flask integration
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    IsolationForest,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data_pipeline"
    / "temporal"
)

MODEL_DIR = (
    BASE_DIR
    / "ai"
    / "models"
)

METRICS_DIR = (
    BASE_DIR
    / "ai"
    / "metrics"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DEMAND_FILE = (
    DATA_DIR
    / "ml_demand_forecasting_dev.csv"
)

STOCKOUT_FILE = (
    DATA_DIR
    / "ml_stockout_prediction_dev.csv"
)

ANOMALY_FILE = (
    DATA_DIR
    / "ml_anomaly_detection_dev.csv"
)


# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Category",
]


# ============================================================
# B4.1 DEMAND FORECASTING
# ============================================================

print("=" * 70)
print("B4 - ML MODEL TRAINING")
print("=" * 70)

print("\n" + "=" * 70)
print("B4.1 - DEMAND FORECASTING")
print("=" * 70)

print("\nLoading demand dataset...")

demand_df = pd.read_csv(
    DEMAND_FILE
)

demand_df["Date"] = pd.to_datetime(
    demand_df["Date"]
)

demand_df = demand_df.sort_values(
    "Date"
).reset_index(drop=True)

print(
    f"Dataset shape: {demand_df.shape}"
)


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

demand_features = [
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Category",
    "Day_of_Week",
    "Day_of_Month",
    "Month",
    "Week_of_Year",
    "Patient_Footfall",
    "Population_Factor",
    "Capacity_Factor",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Outbreak_Flag",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
]

demand_target = "Target_Demand_Next_Day"


# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ------------------------------------------------------------

unique_dates = sorted(
    demand_df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

train_dates = unique_dates[
    :split_index
]

test_dates = unique_dates[
    split_index:
]

demand_train = demand_df[
    demand_df["Date"].isin(train_dates)
].copy()

demand_test = demand_df[
    demand_df["Date"].isin(test_dates)
].copy()

print(
    f"Training dates: {train_dates[0]} → {train_dates[-1]}"
)

print(
    f"Testing dates : {test_dates[0]} → {test_dates[-1]}"
)

print(
    f"Training rows: {len(demand_train):,}"
)

print(
    f"Testing rows : {len(demand_test):,}"
)


# ------------------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

print("\nEncoding categorical features...")

demand_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

demand_encoder.fit(
    demand_train[CATEGORICAL_FEATURES]
)


def encode_dataset(
    df,
    encoder,
    features
):
    result = df[features].copy()

    result[CATEGORICAL_FEATURES] = (
        encoder.transform(
            result[CATEGORICAL_FEATURES]
        )
    )

    return result


X_train_demand = encode_dataset(
    demand_train,
    demand_encoder,
    demand_features
)

X_test_demand = encode_dataset(
    demand_test,
    demand_encoder,
    demand_features
)

y_train_demand = demand_train[
    demand_target
]

y_test_demand = demand_test[
    demand_target
]


# ------------------------------------------------------------
# TRAIN RANDOM FOREST REGRESSOR
# ------------------------------------------------------------

print("\nTraining Random Forest Regressor...")

demand_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

demand_model.fit(
    X_train_demand,
    y_train_demand
)


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print("Generating demand predictions...")

demand_predictions = demand_model.predict(
    X_test_demand
)

demand_predictions = np.maximum(
    demand_predictions,
    0
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

demand_mae = mean_absolute_error(
    y_test_demand,
    demand_predictions
)

demand_rmse = np.sqrt(
    mean_squared_error(
        y_test_demand,
        demand_predictions
    )
)

demand_r2 = r2_score(
    y_test_demand,
    demand_predictions
)

print("\nDemand Forecasting Results")
print("-" * 40)

print(
    f"MAE  : {demand_mae:.4f}"
)

print(
    f"RMSE : {demand_rmse:.4f}"
)

print(
    f"R²   : {demand_r2:.4f}"
)


# ------------------------------------------------------------
# SAVE DEMAND MODEL
# ------------------------------------------------------------

joblib.dump(
    demand_model,
    MODEL_DIR / "demand_forecaster.joblib"
)

joblib.dump(
    demand_encoder,
    MODEL_DIR / "demand_encoder.joblib"
)

print(
    "\n✓ Demand model saved"
)


# ============================================================
# B4.2 STOCKOUT PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("B4.2 - STOCKOUT PREDICTION")
print("=" * 70)

print("\nLoading stockout dataset...")

stockout_df = pd.read_csv(
    STOCKOUT_FILE
)

stockout_df["Date"] = pd.to_datetime(
    stockout_df["Date"]
)

stockout_df = stockout_df.sort_values(
    "Date"
).reset_index(drop=True)

print(
    f"Dataset shape: {stockout_df.shape}"
)


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

stockout_features = [
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Category",
    "Day_of_Week",
    "Day_of_Month",
    "Month",
    "Week_of_Year",
    "Patient_Footfall",
    "Population_Factor",
    "Capacity_Factor",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Opening_Stock",
    "Received_Stock",
    "Days_of_Stock",
    "Outbreak_Flag",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
]

stockout_target = "Stockout_Next_7_Days"


# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ------------------------------------------------------------

unique_dates = sorted(
    stockout_df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

train_dates = unique_dates[
    :split_index
]

test_dates = unique_dates[
    split_index:
]

stockout_train = stockout_df[
    stockout_df["Date"].isin(train_dates)
].copy()

stockout_test = stockout_df[
    stockout_df["Date"].isin(test_dates)
].copy()

print(
    f"Training rows: {len(stockout_train):,}"
)

print(
    f"Testing rows : {len(stockout_test):,}"
)


# ------------------------------------------------------------
# ENCODING
# ------------------------------------------------------------

print("\nEncoding categorical features...")

stockout_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

stockout_encoder.fit(
    stockout_train[CATEGORICAL_FEATURES]
)


X_train_stockout = encode_dataset(
    stockout_train,
    stockout_encoder,
    stockout_features
)

X_test_stockout = encode_dataset(
    stockout_test,
    stockout_encoder,
    stockout_features
)

y_train_stockout = stockout_train[
    stockout_target
].astype(int)

y_test_stockout = stockout_test[
    stockout_target
].astype(int)


print("\nTraining class distribution:")

print(
    y_train_stockout.value_counts()
)


# ------------------------------------------------------------
# TRAIN RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

print("\nTraining Random Forest Classifier...")

stockout_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

stockout_model.fit(
    X_train_stockout,
    y_train_stockout
)


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print("Generating stockout predictions...")

stockout_predictions = (
    stockout_model.predict(
        X_test_stockout
    )
)

stockout_probabilities = (
    stockout_model.predict_proba(
        X_test_stockout
    )[:, 1]
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

stockout_precision = precision_score(
    y_test_stockout,
    stockout_predictions,
    zero_division=0
)

stockout_recall = recall_score(
    y_test_stockout,
    stockout_predictions,
    zero_division=0
)

stockout_f1 = f1_score(
    y_test_stockout,
    stockout_predictions,
    zero_division=0
)

stockout_roc_auc = roc_auc_score(
    y_test_stockout,
    stockout_probabilities
)

stockout_pr_auc = average_precision_score(
    y_test_stockout,
    stockout_probabilities
)

stockout_cm = confusion_matrix(
    y_test_stockout,
    stockout_predictions
)


print("\nStockout Prediction Results")
print("-" * 40)

print(
    f"Precision : {stockout_precision:.4f}"
)

print(
    f"Recall    : {stockout_recall:.4f}"
)

print(
    f"F1-score  : {stockout_f1:.4f}"
)

print(
    f"ROC-AUC   : {stockout_roc_auc:.4f}"
)

print(
    f"PR-AUC    : {stockout_pr_auc:.4f}"
)

print("\nConfusion Matrix:")

print(stockout_cm)


# ------------------------------------------------------------
# SAVE STOCKOUT MODEL
# ------------------------------------------------------------

joblib.dump(
    stockout_model,
    MODEL_DIR / "stockout_classifier.joblib"
)

joblib.dump(
    stockout_encoder,
    MODEL_DIR / "stockout_encoder.joblib"
)

print(
    "\n✓ Stockout model saved"
)


# ============================================================
# B4.3 ANOMALY DETECTION
# ============================================================

print("\n" + "=" * 70)
print("B4.3 - ANOMALY DETECTION")
print("=" * 70)

print("\nLoading anomaly dataset...")

anomaly_df = pd.read_csv(
    ANOMALY_FILE
)

anomaly_df["Date"] = pd.to_datetime(
    anomaly_df["Date"]
)

anomaly_df = anomaly_df.sort_values(
    "Date"
).reset_index(drop=True)

print(
    f"Dataset shape: {anomaly_df.shape}"
)


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

anomaly_features = [
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Category",
    "Day_of_Week",
    "Day_of_Month",
    "Month",
    "Week_of_Year",
    "Patient_Footfall",
    "Population_Factor",
    "Capacity_Factor",
    "Demand",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Opening_Stock",
    "Days_of_Stock",
    "Outbreak_Flag",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
]

anomaly_target = "Anomaly_Flag"


# ------------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ------------------------------------------------------------

unique_dates = sorted(
    anomaly_df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

train_dates = unique_dates[
    :split_index
]

test_dates = unique_dates[
    split_index:
]

anomaly_train = anomaly_df[
    anomaly_df["Date"].isin(train_dates)
].copy()

anomaly_test = anomaly_df[
    anomaly_df["Date"].isin(test_dates)
].copy()


print(
    f"Training rows: {len(anomaly_train):,}"
)

print(
    f"Testing rows : {len(anomaly_test):,}"
)


# ------------------------------------------------------------
# ENCODING
# ------------------------------------------------------------

print("\nEncoding categorical features...")

anomaly_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

anomaly_encoder.fit(
    anomaly_train[CATEGORICAL_FEATURES]
)


X_train_anomaly = encode_dataset(
    anomaly_train,
    anomaly_encoder,
    anomaly_features
)

X_test_anomaly = encode_dataset(
    anomaly_test,
    anomaly_encoder,
    anomaly_features
)

y_test_anomaly = anomaly_test[
    anomaly_target
].astype(int)


# ------------------------------------------------------------
# CONTAMINATION
# ------------------------------------------------------------

contamination = (
    anomaly_train[anomaly_target]
    .mean()
)

# Safety bounds for Isolation Forest
contamination = min(
    max(contamination, 0.001),
    0.20
)

print(
    f"\nEstimated anomaly contamination: "
    f"{contamination:.4f}"
)


# ------------------------------------------------------------
# TRAIN ISOLATION FOREST
# ------------------------------------------------------------

print("\nTraining Isolation Forest...")

anomaly_model = IsolationForest(
    n_estimators=300,
    contamination=contamination,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

anomaly_model.fit(
    X_train_anomaly
)


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

print(
    "Detecting anomalies..."
)

raw_anomaly_predictions = (
    anomaly_model.predict(
        X_test_anomaly
    )
)

# Isolation Forest:
# +1 = normal
# -1 = anomaly

anomaly_predictions = (
    raw_anomaly_predictions == -1
).astype(int)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

anomaly_precision = precision_score(
    y_test_anomaly,
    anomaly_predictions,
    zero_division=0
)

anomaly_recall = recall_score(
    y_test_anomaly,
    anomaly_predictions,
    zero_division=0
)

anomaly_f1 = f1_score(
    y_test_anomaly,
    anomaly_predictions,
    zero_division=0
)

anomaly_cm = confusion_matrix(
    y_test_anomaly,
    anomaly_predictions
)


print("\nAnomaly Detection Results")
print("-" * 40)

print(
    f"Precision : {anomaly_precision:.4f}"
)

print(
    f"Recall    : {anomaly_recall:.4f}"
)

print(
    f"F1-score  : {anomaly_f1:.4f}"
)

print("\nConfusion Matrix:")

print(anomaly_cm)


# ------------------------------------------------------------
# SAVE ANOMALY MODEL
# ------------------------------------------------------------

joblib.dump(
    anomaly_model,
    MODEL_DIR / "anomaly_detector.joblib"
)

joblib.dump(
    anomaly_encoder,
    MODEL_DIR / "anomaly_encoder.joblib"
)

print(
    "\n✓ Anomaly model saved"
)


# ============================================================
# B4.4 SAVE METRICS
# ============================================================

print("\n" + "=" * 70)
print("B4.4 - SAVING MODEL METRICS")
print("=" * 70)


metrics = {

    "demand_forecasting": {
        "model": "RandomForestRegressor",
        "n_estimators": 300,
        "max_depth": 12,
        "mae": float(demand_mae),
        "rmse": float(demand_rmse),
        "r2": float(demand_r2),
        "train_rows": int(len(demand_train)),
        "test_rows": int(len(demand_test)),
    },

    "stockout_prediction": {
        "model": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 12,
        "class_weight": "balanced",
        "precision": float(stockout_precision),
        "recall": float(stockout_recall),
        "f1": float(stockout_f1),
        "roc_auc": float(stockout_roc_auc),
        "pr_auc": float(stockout_pr_auc),
        "confusion_matrix": stockout_cm.tolist(),
        "train_rows": int(len(stockout_train)),
        "test_rows": int(len(stockout_test)),
    },

    "anomaly_detection": {
        "model": "IsolationForest",
        "n_estimators": 300,
        "contamination": float(contamination),
        "precision": float(anomaly_precision),
        "recall": float(anomaly_recall),
        "f1": float(anomaly_f1),
        "confusion_matrix": anomaly_cm.tolist(),
        "train_rows": int(len(anomaly_train)),
        "test_rows": int(len(anomaly_test)),
    }
}


metrics_file = (
    METRICS_DIR
    / "model_metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print(
    f"✓ Metrics saved to:\n{metrics_file}"
)


# ============================================================
# B4.5 FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("B4 MODEL TRAINING COMPLETE")
print("=" * 70)

print("\nModels created:")

print(
    "✓ demand_forecaster.joblib"
)

print(
    "✓ stockout_classifier.joblib"
)

print(
    "✓ anomaly_detector.joblib"
)

print("\nEncoders created:")

print(
    "✓ demand_encoder.joblib"
)

print(
    "✓ stockout_encoder.joblib"
)

print(
    "✓ anomaly_encoder.joblib"
)

print("\nMetrics:")

print(
    f"Demand R²       : {demand_r2:.4f}"
)

print(
    f"Demand MAE      : {demand_mae:.4f}"
)

print(
    f"Stockout Recall : {stockout_recall:.4f}"
)

print(
    f"Stockout F1     : {stockout_f1:.4f}"
)

print(
    f"Stockout PR-AUC : {stockout_pr_auc:.4f}"
)

print(
    f"Anomaly Recall  : {anomaly_recall:.4f}"
)

print(
    f"Anomaly F1      : {anomaly_f1:.4f}"
)

print("\n" + "=" * 70)
print("✓ READY FOR B5 - AI INFERENCE PIPELINE")
print("=" * 70)