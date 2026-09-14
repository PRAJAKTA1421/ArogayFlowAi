"""
ArogyaFlow AI - B5 Feature Builder

Converts Firestore PHC + medicine + demand-history data into
the exact feature schema expected by the saved B4/B4.3 models.

The builder is intentionally tolerant of different Firestore
field names because the existing project may contain older
documents.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from google.cloud.firestore_v1.base_query import FieldFilter


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from firebase_config import db


MEDICINE_MAP = {
    "Paracetamol": {
        "id": "MED-001",
        "category": "Analgesic",
    },
    "Amoxicillin": {
        "id": "MED-002",
        "category": "Antibiotic",
    },
    "ORS": {
        "id": "MED-003",
        "category": "Rehydration",
    },
    "ORS Sachet": {
        "id": "MED-003",
        "category": "Rehydration",
    },
    "Metformin": {
        "id": "MED-004",
        "category": "Antidiabetic",
    },
    "Amlodipine": {
        "id": "MED-005",
        "category": "Antihypertensive",
    },
    "Cetirizine": {
        "id": "MED-006",
        "category": "Antihistamine",
    },
    "Iron Folic Acid Tablet": {
        "id": "MED-007",
        "category": "Supplement",
    },
    "Omeprazole": {
        "id": "MED-008",
        "category": "Gastrointestinal",
    },
}


def _first(
    data: Dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    """Return the first non-empty value among possible field names."""
    for key in keys:
        value = data.get(key)

        if value is not None and value != "":
            return value

    return default


def _number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _bool_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value != 0)

    if isinstance(value, str):
        return int(
            value.strip().lower()
            in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        )

    return 0


def get_phc_context(phc_id: str) -> Dict[str, Any]:
    """Read PHC metadata from Firestore."""
    document = (
        db.collection("phcs")
        .document(phc_id)
        .get()
    )

    if not document.exists:
        raise ValueError(
            f"PHC not found in Firestore: {phc_id}"
        )

    data = document.to_dict() or {}

    return {
        "raw": data,
        "State": _first(
            data,
            [
                "State",
                "state",
                "State_Name",
                "state_name",
            ],
            "Unknown",
        ),
        "PHC_Type": _first(
            data,
            [
                "PHC_Type",
                "phc_type",
                "type",
            ],
            "PHC",
        ),
        "Urban_Rural": _first(
            data,
            [
                "Urban_Rural",
                "urban_rural",
                "area_type",
                "location_type",
            ],
            "Rural",
        ),
        "Population_Factor": _number(
            _first(
                data,
                [
                    "Population_Factor",
                    "population_factor",
                ],
                1.0,
            ),
            1.0,
        ),
        "Capacity_Factor": _number(
            _first(
                data,
                [
                    "Capacity_Factor",
                    "capacity_factor",
                ],
                1.0,
            ),
            1.0,
        ),
    }


def get_medicine_context(
    phc_id: str,
    medicine_name: str,
) -> Dict[str, Any]:
    """Read medicine metadata/current inventory from Firestore."""
    medicine_id = (
        f"{phc_id}_"
        f"{medicine_name.lower().replace(' ', '_')}"
    )

    document = (
        db.collection("medicines")
        .document(medicine_id)
        .get()
    )

    if not document.exists:
        raise ValueError(
            f"Medicine inventory not found: {medicine_id}"
        )

    data = document.to_dict() or {}

    known = MEDICINE_MAP.get(
        medicine_name,
        {},
    )

    medicine_code = _first(
        data,
        [
            "Medicine_ID",
            "medicine_id",
            "medicineId",
        ],
        known.get("id", "UNKNOWN"),
    )

    category = _first(
        data,
        [
            "Category",
            "category",
            "medicine_category",
        ],
        known.get("category", "Unknown"),
    )

    current_stock = _number(
        _first(
            data,
            [
                "Current_Stock",
                "current_stock",
                "stock",
                "quantity",
            ],
            0.0,
        )
    )

    received_stock = _number(
        _first(
            data,
            [
                "Received_Stock",
                "received_stock",
                "last_received",
                "received",
            ],
            0.0,
        )
    )

    return {
        "raw": data,
        "medicine_id": medicine_code,
        "category": category,
        "current_stock": max(
            0.0,
            current_stock,
        ),
        "received_stock": max(
            0.0,
            received_stock,
        ),
    }


def get_demand_history(
    phc_id: str,
    medicine_name: str,
) -> pd.DataFrame:
    """
    Read demand_history while preserving optional B5 fields
    when they exist in Firestore.
    """
    query = (
        db.collection("demand_history")
        .where(
            filter=FieldFilter(
                "phc_id",
                "==",
                phc_id,
            )
        )
        .where(
            filter=FieldFilter(
                "medicine_name",
                "==",
                medicine_name,
            )
        )
    )

    records = []

    for document in query.stream():
        data = document.to_dict() or {}

        records.append(
            {
                "date": data.get("date"),
                "demand": _number(
                    data.get("demand"),
                    0.0,
                ),
                "patient_footfall": _number(
                    data.get(
                        "patient_footfall",
                        0.0,
                    ),
                    0.0,
                ),
                "outbreak_flag": _bool_flag(
                    _first(
                        data,
                        [
                            "outbreak_flag",
                            "Outbreak_Flag",
                        ],
                        0,
                    )
                ),
                "supply_disruption_flag": _bool_flag(
                    _first(
                        data,
                        [
                            "supply_disruption_flag",
                            "Supply_Disruption_Flag",
                        ],
                        0,
                    )
                ),
                "seasonal_factor": _number(
                    _first(
                        data,
                        [
                            "seasonal_factor",
                            "Seasonal_Factor",
                        ],
                        1.0,
                    ),
                    1.0,
                ),
            }
        )

    if not records:
        raise ValueError(
            f"No demand history found for "
            f"{phc_id} / {medicine_name}"
        )

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            f"Demand history contains no valid dates for "
            f"{phc_id} / {medicine_name}"
        )

    return df


def _seasonal_factor(row: pd.Series) -> float:
    value = _number(
        row.get("seasonal_factor"),
        1.0,
    )

    return value if value > 0 else 1.0


def _latest_context(
    history: pd.DataFrame,
) -> pd.Series:
    return history.iloc[-1]


def build_current_features(
    history: pd.DataFrame,
    phc_context: Dict[str, Any],
    medicine_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the feature row representing the latest observed day.

    This row is used to predict the following day.
    """
    if len(history) < 7:
        raise ValueError(
            "At least 7 historical demand records are "
            "required for B5 lag/rolling features."
        )

    df = history.copy().sort_values("date")
    latest = _latest_context(df)

    demand_values = (
        pd.to_numeric(
            df["demand"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )

    latest_demand = float(
        demand_values[-1]
    )

    lag_1 = float(
        demand_values[-2]
    )

    lag_7 = float(
        demand_values[-8]
        if len(demand_values) >= 8
        else demand_values[0]
    )

    rolling_7 = float(
        np.mean(
            demand_values[-7:]
        )
    )

    demand_change = (
        latest_demand - lag_1
    )

    footfall = float(
        pd.to_numeric(
            df["patient_footfall"],
            errors="coerce",
        )
        .fillna(0.0)
        .tail(7)
        .mean()
    )

    latest_date = pd.Timestamp(
        latest["date"]
    )

    # Current inventory is the stock state at inference time.
    current_stock = medicine_context[
        "current_stock"
    ]

    days_of_stock = (
        current_stock
        / max(rolling_7, 1.0)
    )

    return {
        "State": phc_context["State"],
        "PHC_Type": phc_context["PHC_Type"],
        "Urban_Rural": phc_context["Urban_Rural"],
        "Medicine_ID": medicine_context["medicine_id"],
        "Category": medicine_context["category"],

        "Day_of_Week": latest_date.dayofweek,
        "Day_of_Month": latest_date.day,
        "Month": latest_date.month,
        "Week_of_Year": int(
            latest_date.isocalendar().week
        ),

        "Patient_Footfall": footfall,

        "Population_Factor": phc_context[
            "Population_Factor"
        ],
        "Capacity_Factor": phc_context[
            "Capacity_Factor"
        ],

        "Lag_1_Demand": lag_1,
        "Lag_7_Demand": lag_7,
        "Rolling_7_Demand": rolling_7,
        "Demand_Change": demand_change,

        "Outbreak_Flag": int(
            _bool_flag(
                latest.get(
                    "outbreak_flag",
                    0,
                )
            )
        ),
        "Supply_Disruption_Flag": int(
            _bool_flag(
                latest.get(
                    "supply_disruption_flag",
                    0,
                )
            )
        ),
        "Seasonal_Factor": _seasonal_factor(
            latest
        ),

        "Demand": latest_demand,
        "Opening_Stock": current_stock,
        "Received_Stock": medicine_context[
            "received_stock"
        ],
        "Days_of_Stock": days_of_stock,

        "Date": latest_date,
    }


def build_next_day_features(
    history: pd.DataFrame,
    phc_context: Dict[str, Any],
    medicine_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the feature row for the latest known day.

    The saved RF regressor predicts the demand on the next day.
    """
    return build_current_features(
        history,
        phc_context,
        medicine_context,
    )


def append_predicted_day(
    history: pd.DataFrame,
    prediction_date: pd.Timestamp,
    predicted_demand: float,
    patient_footfall: float,
    source_row: Dict[str, Any],
) -> pd.DataFrame:
    """
    Append a predicted day so recursive forecasting can continue.
    """
    new_row = {
        "date": pd.Timestamp(
            prediction_date
        ),
        "demand": max(
            0.0,
            float(predicted_demand),
        ),
        "patient_footfall": max(
            0.0,
            float(patient_footfall),
        ),
        "outbreak_flag": int(
            source_row.get(
                "Outbreak_Flag",
                0,
            )
        ),
        "supply_disruption_flag": int(
            source_row.get(
                "Supply_Disruption_Flag",
                0,
            )
        ),
        "seasonal_factor": _seasonal_factor(
            pd.Series(source_row)
        ),
    }

    return pd.concat(
        [
            history,
            pd.DataFrame([new_row]),
        ],
        ignore_index=True,
    )


def prepare_feature_row_for_date(
    history: pd.DataFrame,
    phc_context: Dict[str, Any],
    medicine_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience alias used by forecasting/stockout code.
    """
    return build_current_features(
        history,
        phc_context,
        medicine_context,
    )
