"""
ArogyaFlow AI - Anomaly Detection

B5 integration of the trained Isolation Forest anomaly model.

Purpose:
    Detect unusual medicine demand / inventory behavior
    for a PHC and medicine using recent Firestore data.

The trained model is loaded through model_service.py.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from google.cloud.firestore_v1.base_query import FieldFilter

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# FIREBASE
# ============================================================

from firebase_config import db


# ============================================================
# MODEL SERVICE
# ============================================================

from ai.model_service import (
    detect_anomaly,
    load_models,
)


# ============================================================
# CONFIGURATION
# ============================================================

HISTORY_LIMIT = 14


# ============================================================
# SAFE CONVERSION HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """

    try:
        if value is None:
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def parse_date(value: Any) -> pd.Timestamp:
    """
    Convert Firestore date/timestamp/string to pandas Timestamp.
    """

    try:
        if hasattr(value, "to_datetime"):
            value = value.to_datetime()

        timestamp = pd.to_datetime(
            value,
            errors="coerce"
        )

        if pd.isna(timestamp):
            return pd.Timestamp.now()

        return timestamp

    except Exception:
        return pd.Timestamp.now()


# ============================================================
# FIRESTORE - DEMAND HISTORY
# ============================================================

def get_recent_demand_history(
    phc_id: str,
    medicine_name: str,
    limit: int = HISTORY_LIMIT
) -> List[Dict[str, Any]]:
    """
    Get recent medicine demand history from Firestore.

    Collection:
        demand_history
    """

    query = (
        db.collection("demand_history")
        .where(
            filter=FieldFilter(
                "phc_id",
                "==",
                phc_id
            )
        )
        .where(
            filter=FieldFilter(
                "medicine_name",
                "==",
                medicine_name
            )
        )
        .order_by(
            "date",
            direction="DESCENDING"
        )
        .limit(limit)
    )

    records = []

    for document in query.stream():

        data = document.to_dict() or {}

        records.append(data)

    return records


# ============================================================
# FIRESTORE - INVENTORY
# ============================================================

def get_current_inventory(
    phc_id: str,
    medicine_name: str
) -> Dict[str, Any]:
    """
    Get current medicine inventory.

    Expected document pattern:
        medicines/{PHC_ID}_{medicine_name}
    """

    document_id = (
        f"{phc_id}_"
        f"{medicine_name.lower().replace(' ', '_')}"
    )

    document = (
        db.collection("medicines")
        .document(document_id)
        .get()
    )

    if document.exists:
        return document.to_dict() or {}

    # Fallback: search by PHC + medicine name
    query = (
        db.collection("medicines")
        .where(
            filter=FieldFilter(
                "phc_id",
                "==",
                phc_id
            )
        )
        .where(
            filter=FieldFilter(
                "medicine_name",
                "==",
                medicine_name
            )
        )
        .limit(1)
    )

    for document in query.stream():
        return document.to_dict() or {}

    return {}


# ============================================================
# FIRESTORE - PHC CONTEXT
# ============================================================

def get_phc_context(
    phc_id: str
) -> Dict[str, Any]:
    """
    Get PHC information from Firestore.
    """

    document = (
        db.collection("phcs")
        .document(phc_id)
        .get()
    )

    if document.exists:
        return document.to_dict() or {}

    return {}


# ============================================================
# MEDICINE INFORMATION
# ============================================================

MEDICINE_INFO = {
    "Paracetamol": {
        "medicine_id": "MED-001",
        "category": "Analgesic/Antipyretic",
    },
    "Amoxicillin": {
        "medicine_id": "MED-002",
        "category": "Antibiotic",
    },
    "ORS": {
        "medicine_id": "MED-003",
        "category": "Rehydration",
    },
    "Metformin": {
        "medicine_id": "MED-004",
        "category": "Antidiabetic",
    },
    "Amlodipine": {
        "medicine_id": "MED-005",
        "category": "Antihypertensive",
    },
    "Cetirizine": {
        "medicine_id": "MED-006",
        "category": "Antihistamine",
    },
    "Iron Folic Acid Tablet": {
        "medicine_id": "MED-007",
        "category": "Supplement",
    },
    "Omeprazole": {
        "medicine_id": "MED-008",
        "category": "Gastrointestinal",
    },
}


# ============================================================
# BUILD BEHAVIORAL FEATURE ROW
# ============================================================

def build_anomaly_feature_row(
    phc_id: str,
    medicine_name: str,
) -> Dict[str, Any]:
    """
    Build the current behavioral feature row required by
    the trained Isolation Forest model.
    """

    history = get_recent_demand_history(
        phc_id,
        medicine_name
    )

    if not history:
        raise ValueError(
            f"No demand history found for "
            f"{phc_id} / {medicine_name}"
        )

    # --------------------------------------------------------
    # Convert history to dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(history)

    # Make sure required columns exist
    if "demand" not in df.columns:
        raise ValueError(
            "demand_history documents do not contain "
            "'demand' field."
        )

    if "date" in df.columns:
        df["Date"] = df["date"].apply(parse_date)

    else:
        df["Date"] = pd.Timestamp.now()

    df["Demand"] = pd.to_numeric(
        df["demand"],
        errors="coerce"
    ).fillna(0.0)

    # Patient footfall
    if "patient_footfall" in df.columns:
        df["Patient_Footfall"] = pd.to_numeric(
            df["patient_footfall"],
            errors="coerce"
        ).fillna(0.0)

    elif "Patient_Footfall" in df.columns:
        df["Patient_Footfall"] = pd.to_numeric(
            df["Patient_Footfall"],
            errors="coerce"
        ).fillna(0.0)

    else:
        df["Patient_Footfall"] = 0.0

    # Seasonal factor
    if "seasonal_factor" in df.columns:
        df["Seasonal_Factor"] = pd.to_numeric(
            df["seasonal_factor"],
            errors="coerce"
        ).fillna(1.0)

    elif "Seasonal_Factor" in df.columns:
        df["Seasonal_Factor"] = pd.to_numeric(
            df["Seasonal_Factor"],
            errors="coerce"
        ).fillna(1.0)

    else:
        df["Seasonal_Factor"] = 1.0

    # --------------------------------------------------------
    # Sort oldest → newest
    # --------------------------------------------------------

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Current demand
    # --------------------------------------------------------

    current = df.iloc[-1]

    current_demand = safe_float(
        current["Demand"]
    )

    # --------------------------------------------------------
    # Lag 1
    # --------------------------------------------------------

    if len(df) >= 2:
        lag_1 = safe_float(
            df.iloc[-2]["Demand"]
        )
    else:
        lag_1 = current_demand

    # --------------------------------------------------------
    # Lag 7
    # --------------------------------------------------------

    if len(df) >= 8:
        lag_7 = safe_float(
            df.iloc[-8]["Demand"]
        )
    elif len(df) >= 2:
        lag_7 = safe_float(
            df.iloc[0]["Demand"]
        )
    else:
        lag_7 = current_demand

    # --------------------------------------------------------
    # Rolling 7-day demand
    # --------------------------------------------------------

    recent_demands = (
        df["Demand"]
        .tail(7)
    )

    rolling_7 = safe_float(
        recent_demands.mean(),
        current_demand
    )

    # --------------------------------------------------------
    # Demand change
    # --------------------------------------------------------

    demand_change = (
        current_demand - lag_1
    )

    # --------------------------------------------------------
    # Current inventory
    # --------------------------------------------------------

    inventory = get_current_inventory(
        phc_id,
        medicine_name
    )

    current_stock = safe_float(
        inventory.get(
            "current_stock",
            inventory.get(
                "stock",
                0
            )
        )
    )

    daily_demand = safe_float(
        inventory.get(
            "daily_demand",
            current_demand
        ),
        current_demand
    )

    # Days of stock
    if daily_demand > 0:
        days_of_stock = (
            current_stock / daily_demand
        )
    else:
        days_of_stock = 999.0

    # --------------------------------------------------------
    # Return raw feature row
    # --------------------------------------------------------

    return {
        "Demand": current_demand,

        "Lag_1_Demand": lag_1,

        "Lag_7_Demand": lag_7,

        "Rolling_7_Demand": rolling_7,

        "Demand_Change": demand_change,

        "Patient_Footfall": safe_float(
            current.get(
                "Patient_Footfall",
                0
            )
        ),

        "Opening_Stock": current_stock,

        "Days_of_Stock": days_of_stock,

        "Seasonal_Factor": safe_float(
            current.get(
                "Seasonal_Factor",
                1.0
            ),
            1.0
        ),

        "PHC_ID": phc_id,

        "Medicine_Name": medicine_name,

        "Date": current.get(
            "Date",
            pd.Timestamp.now()
        ),
    }


# ============================================================
# ANALYZE ANOMALY
# ============================================================

def analyze_anomaly(
    phc_id: str,
    medicine_name: str,
    save_result: bool = True,
) -> Dict[str, Any]:
    """
    Run Isolation Forest anomaly detection.
    """

    print("\n🔍 Starting anomaly analysis...")

    # Load model once
    load_models()

    print(
        "\n🤖 Running Isolation Forest "
        "anomaly detector..."
    )

    feature_row = build_anomaly_feature_row(
        phc_id,
        medicine_name
    )

    result = detect_anomaly(
        feature_row
    )

    is_anomaly = result[
        "is_anomaly"
    ]

    isolation_score = result[
        "isolation_score"
    ]

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 50)

    print(
        f"🏥 PHC: {phc_id}"
    )

    print(
        f"💊 Medicine: {medicine_name}"
    )

    print("=" * 50)

    print("\n📊 BEHAVIORAL ANOMALY ANALYSIS")
    print("-" * 50)

    print(
        f"Current Demand: "
        f"{feature_row['Demand']:.2f}"
    )

    print(
        f"7-Day Average Demand: "
        f"{feature_row['Rolling_7_Demand']:.2f}"
    )

    print(
        f"Demand Change: "
        f"{feature_row['Demand_Change']:.2f}"
    )

    print(
        f"Days of Stock: "
        f"{feature_row['Days_of_Stock']:.2f}"
    )

    print(
        f"Isolation Score: "
        f"{isolation_score:.6f}"
    )

    print("\n🚨 ANOMALY RESULT")
    print("-" * 50)

    if is_anomaly:

        print(
            "⚠️ ANOMALY DETECTED"
        )

        anomaly_status = "ANOMALY"

    else:

        print(
            "✅ NORMAL BEHAVIOR"
        )

        anomaly_status = "NORMAL"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result_data = {
        "phc_id": phc_id,
        "medicine_name": medicine_name,

        "is_anomaly": bool(is_anomaly),

        "anomaly_prediction": int(
            result["anomaly_prediction"]
        ),

        "isolation_score": float(
            isolation_score
        ),

        "anomaly_status": anomaly_status,

        "current_demand": float(
            feature_row["Demand"]
        ),

        "rolling_7_demand": float(
            feature_row[
                "Rolling_7_Demand"
            ]
        ),

        "demand_change": float(
            feature_row[
                "Demand_Change"
            ]
        ),

        "days_of_stock": float(
            feature_row[
                "Days_of_Stock"
            ]
        ),

        "model": "IsolationForest",

        "model_version": "B5.3",

        "created_at": datetime.now(
            timezone.utc
        ),
    }

    # --------------------------------------------------------
    # Save to Firestore
    # --------------------------------------------------------

    if save_result:

        document_id = (
            f"{phc_id}_"
            f"{medicine_name.lower().replace(' ', '_')}"
        )

        db.collection(
            "anomalies"
        ).document(
            document_id
        ).set(
            result_data,
            merge=True
        )

        print(
            "\n🔥 Anomaly analysis "
            "saved to Firestore."
        )

    return result_data


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)

    print(
        "      AROGYAFLOW AI "
        "ANOMALY DETECTION TEST"
    )

    print("=" * 50)

    try:

        result = analyze_anomaly(
            phc_id="PHC001",
            medicine_name="Paracetamol",
            save_result=True,
        )

        print("\n" + "=" * 50)

        print("FINAL RESULT")

        print("=" * 50)

        print(
            f"Anomaly: "
            f"{result['anomaly_status']}"
        )

        print(
            f"Isolation Score: "
            f"{result['isolation_score']}"
        )

        print(
            f"Demand: "
            f"{result['current_demand']}"
        )

        print(
            f"7-Day Average: "
            f"{result['rolling_7_demand']}"
        )

        print(
            f"Days of Stock: "
            f"{result['days_of_stock']:.2f}"
        )

        print(
            "\n✅ Anomaly detection test complete."
        )

    except Exception as error:

        print(
            "\n❌ ANOMALY DETECTION ERROR"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise