
import sys
import os

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from google.cloud.firestore_v1.base_query import FieldFilter

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from firebase_config import db

from ai.model_service import load_models


# ============================================================
# MODEL FEATURES
# ============================================================

# IMPORTANT:
# These are the EXACT 19 features used by the trained
# demand Random Forest model.

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


CATEGORICAL_FEATURES = [
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Category",
]


# ============================================================
# MEDICINE INFORMATION
# ============================================================

MEDICINE_INFO = {

    "Paracetamol": {
        "id": "MED-001",
        "category": "Analgesic/Antipyretic",
        "base_demand": 35,
    },

    "Amoxicillin": {
        "id": "MED-002",
        "category": "Antibiotic",
        "base_demand": 18,
    },

    "ORS Sachet": {
        "id": "MED-003",
        "category": "Rehydration",
        "base_demand": 22,
    },

    "Metformin": {
        "id": "MED-004",
        "category": "Antidiabetic",
        "base_demand": 14,
    },

    "Amlodipine": {
        "id": "MED-005",
        "category": "Antihypertensive",
        "base_demand": 12,
    },

    "Cetirizine": {
        "id": "MED-006",
        "category": "Antihistamine",
        "base_demand": 16,
    },

    "Iron Folic Acid Tablet": {
        "id": "MED-007",
        "category": "Supplement",
        "base_demand": 20,
    },

    "Omeprazole": {
        "id": "MED-008",
        "category": "Gastrointestinal",
        "base_demand": 10,
    },
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except (TypeError, ValueError):

        return default


def safe_int(value, default=0):

    try:

        if value is None:
            return default

        return int(float(value))

    except (TypeError, ValueError):

        return default


def normalize_medicine_name(name):

    if not name:
        return ""

    name = str(name).strip()

    aliases = {

        "Paracetamol 500mg":
            "Paracetamol",

        "Amoxicillin 500mg":
            "Amoxicillin",

        "ORS":
            "ORS Sachet",

        "Metformin 500mg":
            "Metformin",

        "Amlodipine 5mg":
            "Amlodipine",

        "Cetirizine 10mg":
            "Cetirizine",

        "Iron Folic Acid":
            "Iron Folic Acid Tablet",

        "Omeprazole 20mg":
            "Omeprazole",
    }

    return aliases.get(
        name,
        name
    )


def get_medicine_info(medicine_name):

    medicine_name = normalize_medicine_name(
        medicine_name
    )

    if medicine_name in MEDICINE_INFO:

        return MEDICINE_INFO[
            medicine_name
        ]

    return {

        "id": medicine_name,

        "category": "Other",

        "base_demand": 10,
    }


# ============================================================
# FIRESTORE - DEMAND HISTORY
# ============================================================

def get_demand_history(
    phc_id,
    medicine_name
):
    """
    Fetch historical demand data from Firestore.

    Collection:
        demand_history

    Required fields:
        phc_id
        medicine_name
        date
        demand
        patient_footfall
    """

    medicine_name = normalize_medicine_name(
        medicine_name
    )

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
    )

    documents = query.stream()

    records = []

    for document in documents:

        data = document.to_dict()

        records.append({

            "date":
                data.get("date"),

            "demand":
                safe_float(
                    data.get("demand")
                ),

            "patient_footfall":
                safe_float(
                    data.get(
                        "patient_footfall"
                    ),
                    1.0
                ),

            "outbreak_flag":
                safe_int(
                    data.get(
                        "outbreak_flag"
                    ),
                    0
                ),

            "supply_disruption_flag":
                safe_int(
                    data.get(
                        "supply_disruption_flag"
                    ),
                    0
                ),

            "seasonal_factor":
                safe_float(
                    data.get(
                        "seasonal_factor"
                    ),
                    1.0
                ),
        })

    if not records:

        raise ValueError(
            f"No demand history found for "
            f"{phc_id} / {medicine_name}"
        )

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# FIRESTORE - PHC INFORMATION
# ============================================================

def get_phc_context(phc_id):

    try:

        document = (
            db.collection("phcs")
            .document(phc_id)
            .get()
        )

        if not document.exists:

            print(
                f"[Forecasting] "
                f"PHC not found: {phc_id}"
            )

            return {}

        return document.to_dict() or {}

    except Exception as e:

        print(
            f"[Forecasting] "
            f"Error loading PHC: {e}"
        )

        return {}


# ============================================================
# PHC CONTEXT HELPERS
# ============================================================

def get_context_value(
    context,
    possible_names,
    default
):

    for name in possible_names:

        if name in context:

            value = context.get(name)

            if value is not None:

                return value

    return default


def get_population_factor(
    phc_context
):

    value = get_context_value(

        phc_context,

        [
            "Population_Factor",
            "population_factor",
        ],

        1.0,
    )

    return safe_float(
        value,
        1.0
    )


def get_capacity_factor(
    phc_context
):

    value = get_context_value(

        phc_context,

        [
            "Capacity_Factor",
            "capacity_factor",
        ],

        1.0,
    )

    return safe_float(
        value,
        1.0
    )


# ============================================================
# FEATURE CALCULATIONS
# ============================================================

def calculate_patient_footfall(
    history
):

    if history.empty:

        return 1.0

    recent = history.tail(7)

    value = recent[
        "patient_footfall"
    ].mean()

    return max(
        safe_float(value, 1.0),
        1.0
    )


def calculate_seasonal_factor(
    target_date
):

    day_of_year = (
        target_date
        .timetuple()
        .tm_yday
    )

    factor = (
        1.0
        +
        0.10
        *
        np.sin(
            2
            *
            np.pi
            *
            day_of_year
            /
            365.25
        )
    )

    return float(factor)


def get_recent_flag(
    history,
    column_name,
    target_date
):

    if history.empty:

        return 0

    if column_name not in history.columns:

        return 0

    recent = history[
        history["date"]
        >=
        (
            target_date
            -
            timedelta(days=14)
        )
    ]

    if recent.empty:

        return 0

    return int(
        recent[column_name]
        .fillna(0)
        .max()
    )


# ============================================================
# BUILD MODEL FEATURES
# ============================================================

def build_feature_row(
    target_date,
    history,
    phc_context,
    medicine_name
):
    """
    Build exactly the 19 features expected
    by the trained Random Forest model.
    """

    medicine = get_medicine_info(
        medicine_name
    )

    # --------------------------------------------------------
    # Demand values
    # --------------------------------------------------------

    if history.empty:

        demand_values = [
            medicine["base_demand"]
        ]

    else:

        demand_values = (
            history["demand"]
            .astype(float)
            .tolist()
        )

    if not demand_values:

        demand_values = [
            medicine["base_demand"]
        ]

    # --------------------------------------------------------
    # Lag 1
    # --------------------------------------------------------

    lag_1_demand = demand_values[-1]

    # --------------------------------------------------------
    # Lag 7
    # --------------------------------------------------------

    if len(demand_values) >= 7:

        lag_7_demand = demand_values[-7]

    else:

        lag_7_demand = np.mean(
            demand_values
        )

    # --------------------------------------------------------
    # Rolling 7
    # --------------------------------------------------------

    rolling_7_demand = np.mean(
        demand_values[-7:]
    )

    # --------------------------------------------------------
    # Demand change
    # --------------------------------------------------------

    if len(demand_values) >= 2:

        demand_change = (
            demand_values[-1]
            -
            demand_values[-2]
        )

    else:

        demand_change = 0.0

    # --------------------------------------------------------
    # Patient footfall
    # --------------------------------------------------------

    patient_footfall = (
        calculate_patient_footfall(
            history
        )
    )

    # --------------------------------------------------------
    # PHC context
    # --------------------------------------------------------

    state = get_context_value(

        phc_context,

        [
            "State",
            "state",
        ],

        "Unknown",
    )

    phc_type = get_context_value(

        phc_context,

        [
            "PHC_Type",
            "phc_type",
        ],

        "PHC",
    )

    urban_rural = get_context_value(

        phc_context,

        [
            "Urban_Rural",
            "urban_rural",
        ],

        "Rural",
    )

    population_factor = (
        get_population_factor(
            phc_context
        )
    )

    capacity_factor = (
        get_capacity_factor(
            phc_context
        )
    )

    # --------------------------------------------------------
    # Outbreak
    # --------------------------------------------------------

    outbreak_flag = get_recent_flag(

        history,

        "outbreak_flag",

        target_date
    )

    # --------------------------------------------------------
    # Supply disruption
    # --------------------------------------------------------

    supply_disruption_flag = (
        get_recent_flag(

            history,

            "supply_disruption_flag",

            target_date
        )
    )

    # --------------------------------------------------------
    # Seasonal factor
    # --------------------------------------------------------

    seasonal_factor = (
        calculate_seasonal_factor(
            target_date
        )
    )

    # If Firestore already has a valid
    # seasonal factor, use the latest value.

    if (
        not history.empty
        and
        "seasonal_factor" in history.columns
    ):

        valid_seasonal = (
            history[
                history[
                    "seasonal_factor"
                ].notna()
            ]
        )

        if not valid_seasonal.empty:

            seasonal_factor = safe_float(

                valid_seasonal[
                    "seasonal_factor"
                ].iloc[-1],

                seasonal_factor
            )

    # --------------------------------------------------------
    # EXACT MODEL ROW
    # --------------------------------------------------------

    row = {

        "State":
            state,

        "PHC_Type":
            phc_type,

        "Urban_Rural":
            urban_rural,

        "Medicine_ID":
            medicine["id"],

        "Category":
            medicine["category"],

        "Day_of_Week":
            target_date.weekday(),

        "Day_of_Month":
            target_date.day,

        "Month":
            target_date.month,

        "Week_of_Year":
            int(
                target_date
                .isocalendar()
                .week
            ),

        "Patient_Footfall":
            patient_footfall,

        "Population_Factor":
            population_factor,

        "Capacity_Factor":
            capacity_factor,

        "Lag_1_Demand":
            lag_1_demand,

        "Lag_7_Demand":
            lag_7_demand,

        "Rolling_7_Demand":
            rolling_7_demand,

        "Demand_Change":
            demand_change,

        "Outbreak_Flag":
            outbreak_flag,

        "Supply_Disruption_Flag":
            supply_disruption_flag,

        "Seasonal_Factor":
            seasonal_factor,
    }

    return row


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_demand(
    model,
    encoder,
    feature_row
):
    """
    Run prediction using the already-trained
    Random Forest model.
    """

    df = pd.DataFrame(
        [feature_row]
    )

    # Ensure exact feature order.
    df = df[
        DEMAND_FEATURES
    ]

    # Encode the five categorical fields
    # using the SAME encoder used during training.

    df[
        CATEGORICAL_FEATURES
    ] = encoder.transform(
        df[
            CATEGORICAL_FEATURES
        ]
    )

    # Final safety check.

    if list(
        df.columns
    ) != DEMAND_FEATURES:

        raise ValueError(
            "Feature order mismatch "
            "before demand prediction."
        )

    prediction = model.predict(
        df
    )[0]

    prediction = safe_float(
        prediction,
        0.0
    )

    # Demand cannot be negative.

    return max(
        0.0,
        round(
            prediction,
            2
        )
    )


# ============================================================
# FORECAST NEXT N DAYS
# ============================================================

def forecast_demand(
    phc_id,
    medicine_name,
    days=7
):
    """
    Generate demand forecast using the
    trained B4 Random Forest model.

    IMPORTANT:
    The model is NOT retrained here.

    Forecasting is recursive:
        Day 1 prediction
        -> used as future demand
        -> Day 2 prediction
        -> etc.
    """

    if days <= 0:

        return []

    medicine_name = normalize_medicine_name(
        medicine_name
    )

    print(
        "\n[Forecasting] "
        f"PHC={phc_id}, "
        f"Medicine={medicine_name}, "
        f"Days={days}"
    )

    # --------------------------------------------------------
    # Load trained models
    # --------------------------------------------------------

    models = load_models()

    model = models.get(
        "demand_model"
    )

    encoder = models.get(
        "demand_encoder"
    )

    if model is None:

        raise RuntimeError(
            "Trained demand model "
            "is not available."
        )

    if encoder is None:

        raise RuntimeError(
            "Demand encoder "
            "is not available."
        )

    # --------------------------------------------------------
    # Load Firestore data
    # --------------------------------------------------------

    history = get_demand_history(

        phc_id,

        medicine_name
    )

    phc_context = get_phc_context(
        phc_id
    )

    # --------------------------------------------------------
    # Last historical date
    # --------------------------------------------------------

    last_date = (
        history["date"]
        .max()
    )

    # --------------------------------------------------------
    # Working history
    #
    # This is used for recursive forecasting.
    # --------------------------------------------------------

    working_history = (
        history.copy()
    )

    predictions = []

    # --------------------------------------------------------
    # Predict one day at a time
    # --------------------------------------------------------

    for i in range(
        1,
        days + 1
    ):

        future_date = (

            last_date

            +

            timedelta(
                days=i
            )
        )

        # ----------------------------------------------------
        # Build exact 19-feature input
        # ----------------------------------------------------

        feature_row = (
            build_feature_row(

                target_date=future_date,

                history=working_history,

                phc_context=phc_context,

                medicine_name=medicine_name,
            )
        )

        # ----------------------------------------------------
        # Predict using trained model
        # ----------------------------------------------------

        prediction = predict_demand(

            model=model,

            encoder=encoder,

            feature_row=feature_row
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        predictions.append({

            "date":
                future_date.strftime(
                    "%Y-%m-%d"
                ),

            "predicted_demand":
                prediction,
        })

        # ----------------------------------------------------
        # Add prediction to working history
        #
        # This allows the next day's prediction to use
        # the previous prediction as Lag_1_Demand.
        # ----------------------------------------------------

        new_record = {

            "date":
                future_date,

            "demand":
                prediction,

            "patient_footfall":
                feature_row[
                    "Patient_Footfall"
                ],

            "outbreak_flag":
                feature_row[
                    "Outbreak_Flag"
                ],

            "supply_disruption_flag":
                feature_row[
                    "Supply_Disruption_Flag"
                ],

            "seasonal_factor":
                feature_row[
                    "Seasonal_Factor"
                ],
        }

        working_history = pd.concat(

            [

                working_history,

                pd.DataFrame(
                    [new_record]
                ),

            ],

            ignore_index=True
        )

    print(
        "[Forecasting] "
        f"Generated {len(predictions)} predictions."
    )

    return predictions


# ============================================================
# SAVE FORECAST TO FIRESTORE
# ============================================================

def save_forecast(
    phc_id,
    medicine_name,
    predictions
):
    """
    Save forecast results to Firestore.
    """

    medicine_name = normalize_medicine_name(
        medicine_name
    )

    forecast_id = (

        f"{phc_id}_"

        f"{medicine_name.lower().replace(' ', '_')}"
    )

    data = {

        "phc_id":
            phc_id,

        "medicine_name":
            medicine_name,

        "predictions":
            predictions,

        "generated_at":
            datetime.utcnow(),

        "forecast_days":
            len(predictions),

        "model":
            "RandomForestRegressor",

        "model_version":
            "B5.2",
    }

    db.collection(
        "forecasts"
    ).document(
        forecast_id
    ).set(
        data
    )

    return forecast_id


# ============================================================
# COMPLETE FORECAST PIPELINE
# ============================================================

def generate_forecast(
    phc_id,
    medicine_name,
    days=7
):
    """
    Generate forecast and save it to Firestore.
    """

    predictions = forecast_demand(

        phc_id,

        medicine_name,

        days
    )

    forecast_id = save_forecast(

        phc_id,

        medicine_name,

        predictions
    )

    return {

        "forecast_id":
            forecast_id,

        "phc_id":
            phc_id,

        "medicine_name":
            normalize_medicine_name(
                medicine_name
            ),

        "predictions":
            predictions,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "   AROGYAFLOW AI - B5.2 FORECAST TEST"
    )

    print(
        "======================================\n"
    )

    try:

        result = generate_forecast(

            phc_id="PHC001",

            medicine_name="Paracetamol",

            days=7,
        )

        print(
            f"🏥 PHC: "
            f"{result['phc_id']}"
        )

        print(
            f"💊 Medicine: "
            f"{result['medicine_name']}"
        )

        print(
            "\n📈 7-Day ML Forecast:\n"
        )

        for prediction in result[
            "predictions"
        ]:

            print(

                f"  "
                f"{prediction['date']} "
                f"→ "
                f"{prediction['predicted_demand']} "
                f"units"
            )

        print(
            "\n✅ Forecast saved to Firestore."
        )

        print(
            "\n🔥 B5.2 TEST PASSED"
        )

    except Exception as e:

        print(
            "\n❌ B5.2 TEST FAILED"
        )

        print(
            f"Error: {e}"
        )

        raise
