"""
ArogyaFlow AI - B5 Model Service

Loads the already-trained B4/B4.3 models once and provides
small inference helpers for:
1. Next-day demand forecasting
2. 7-day stockout classification
3. Behavioral anomaly detection

IMPORTANT:
- Demand/stockout models use the exact categorical encoders from B4.
- The improved anomaly model from train_anomaly_model.py is numeric-only.
- Do NOT use anomaly_encoder.joblib with the improved anomaly model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ai" / "models"


CATEGORICAL_FEATURES = [
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Category",
]

DEMAND_FEATURES = [
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

STOCKOUT_FEATURES = [
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


def _required_file(name: str) -> Path:
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Required ML artifact not found: {path}\n"
            "Run the B4 model training scripts first."
        )
    return path


@lru_cache(maxsize=1)
def load_models() -> Dict[str, Any]:
    """Load all trained models/encoders once per Python process."""
    anomaly_features_path = _required_file("anomaly_features.json")

    with open(anomaly_features_path, "r", encoding="utf-8") as f:
        anomaly_features = json.load(f)

    models = {
        "demand_model": joblib.load(
            _required_file("demand_forecaster.joblib")
        ),
        "demand_encoder": joblib.load(
            _required_file("demand_encoder.joblib")
        ),
        "stockout_model": joblib.load(
            _required_file("stockout_classifier.joblib")
        ),
        "stockout_encoder": joblib.load(
            _required_file("stockout_encoder.joblib")
        ),
        "anomaly_model": joblib.load(
            _required_file("anomaly_detector.joblib")
        ),
        "anomaly_features": anomaly_features,
    }

    return models


def _clean_frame(
    feature_row: Dict[str, Any],
    features: list[str],
) -> pd.DataFrame:
    """Create a one-row dataframe with exact feature order."""
    row = {
        feature: feature_row.get(feature, 0)
        for feature in features
    }

    frame = pd.DataFrame([row], columns=features)

    # Convert numeric fields safely. Categorical columns remain strings.
    for column in features:
        if column not in CATEGORICAL_FEATURES:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).fillna(0.0)

    return frame


def _encode(
    frame: pd.DataFrame,
    encoder,
) -> pd.DataFrame:
    """Apply the same B4 OrdinalEncoder to categorical columns."""
    result = frame.copy()

    result[CATEGORICAL_FEATURES] = encoder.transform(
        result[CATEGORICAL_FEATURES]
    )

    return result


def predict_demand(feature_row: Dict[str, Any]) -> float:
    """
    Predict next-day medicine demand.

    The B4 target was Target_Demand_Next_Day, so this predicts
    demand for the day immediately following the feature row.
    """
    models = load_models()

    frame = _clean_frame(
        feature_row,
        DEMAND_FEATURES,
    )

    encoded = _encode(
        frame,
        models["demand_encoder"],
    )

    prediction = models["demand_model"].predict(encoded)[0]

    return round(float(max(0.0, prediction)), 2)


def predict_stockout(
    feature_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Predict probability of stockout within the next 7 days.

    Returns both the ML class and probability for the positive
    stockout class.
    """
    models = load_models()

    frame = _clean_frame(
        feature_row,
        STOCKOUT_FEATURES,
    )

    encoded = _encode(
        frame,
        models["stockout_encoder"],
    )

    model = models["stockout_model"]

    predicted_class = int(model.predict(encoded)[0])

    probabilities = model.predict_proba(encoded)[0]
    classes = list(model.classes_)

    if 1 in classes:
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])
    else:
        probability = 0.0

    return {
        "stockout_prediction": predicted_class,
        "stockout_probability": round(probability, 4),
        "stockout_probability_percent": round(
            probability * 100,
            2,
        ),
    }


def build_anomaly_features(feature_row: Dict[str, Any]) -> Dict[str, float]:
    """
    Build the behavioral features used by B4.3 Isolation Forest.

    This mirrors train_anomaly_model.py.
    """
    demand = max(float(feature_row.get("Demand", 0)), 0.0)
    lag_1 = max(float(feature_row.get("Lag_1_Demand", 0)), 0.0)
    lag_7 = max(float(feature_row.get("Lag_7_Demand", 0)), 0.0)
    rolling = max(
        float(feature_row.get("Rolling_7_Demand", 0)),
        0.0,
    )
    demand_change = float(
        feature_row.get("Demand_Change", 0)
    )
    footfall = max(
        float(feature_row.get("Patient_Footfall", 0)),
        0.0,
    )
    opening_stock = max(
        float(feature_row.get("Opening_Stock", 0)),
        0.0,
    )
    days_stock = max(
        float(feature_row.get("Days_of_Stock", 0)),
        0.0,
    )
    seasonal = float(
        feature_row.get("Seasonal_Factor", 1.0)
    )

    return {
        "Demand": demand,
        "Lag_1_Demand": lag_1,
        "Lag_7_Demand": lag_7,
        "Rolling_7_Demand": rolling,
        "Demand_Change": demand_change,
        "Patient_Footfall": footfall,
        "Opening_Stock": opening_stock,
        "Days_of_Stock": days_stock,
        "Seasonal_Factor": seasonal,
        "Demand_vs_Rolling": (
            demand / max(rolling, 1.0)
        ),
        "Demand_Change_Ratio": (
            abs(demand_change) / max(lag_1, 1.0)
        ),
        "Demand_per_Patient": (
            demand / max(footfall, 1.0)
        ),
        "Stock_to_Demand_Ratio": (
            opening_stock / max(demand, 1.0)
        ),
        "Demand_Acceleration": (
            demand - lag_1
        ),
        "Demand_Acceleration_7": (
            demand - lag_7
        ),
        "Footfall_Demand_Ratio": (
            footfall / max(demand, 1.0)
        ),
        "Stock_Pressure": (
            1.0 / max(days_stock, 0.1)
        ),
    }


def detect_anomaly(
    feature_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect unusual demand/stock behavior.

    Isolation Forest returns:
      +1 = normal
      -1 = anomaly
    """
    models = load_models()

    behavioral = build_anomaly_features(
        feature_row
    )

    features = models["anomaly_features"]

    frame = pd.DataFrame(
        [{
            feature: behavioral.get(feature, 0.0)
            for feature in features
        }],
        columns=features,
    )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    model = models["anomaly_model"]

    raw_prediction = int(model.predict(frame)[0])
    is_anomaly = raw_prediction == -1

    # Isolation Forest's decision_function:
    # lower values indicate observations more isolated
    # from the normal population.
    decision_score = float(
        model.decision_function(frame)[0]
    )

    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_prediction": int(is_anomaly),
        "isolation_score": round(
            decision_score,
            6,
        ),
    }


def model_status() -> Dict[str, Any]:
    """Useful for a health-check/API endpoint."""
    models = load_models()

    return {
        "status": "ready",
        "demand_model": type(
            models["demand_model"]
        ).__name__,
        "stockout_model": type(
            models["stockout_model"]
        ).__name__,
        "anomaly_model": type(
            models["anomaly_model"]
        ).__name__,
        "demand_feature_count": len(
            DEMAND_FEATURES
        ),
        "stockout_feature_count": len(
            STOCKOUT_FEATURES
        ),
        "anomaly_feature_count": len(
            models["anomaly_features"]
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ArogyaFlow AI - B5 Model Service")
    print("=" * 60)

    status = model_status()

    for key, value in status.items():
        print(f"{key}: {value}")

    print("\n✓ All saved ML artifacts loaded successfully.")
