import sys
import os
from google.cloud.firestore_v1.base_query import FieldFilter
# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor

from firebase_config import db


# ============================================================
# FIRESTORE DATA LOADING
# ============================================================

def get_demand_history(phc_id, medicine_name):
    """
    Fetch historical demand data for a specific
    PHC + medicine combination.
    """

    query = (
        db.collection("demand_history")
        .where(filter=FieldFilter("phc_id", "==", phc_id))
        .where(filter=FieldFilter("medicine_name", "==", medicine_name))
    )

    documents = query.stream()

    records = []

    for document in documents:

        data = document.to_dict()

        records.append({
            "date": data["date"],
            "demand": data["demand"],
            "patient_footfall": data["patient_footfall"],
        })

    if not records:
        raise ValueError(
            f"No demand history found for "
            f"{phc_id} / {medicine_name}"
        )

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(df):

    df = df.copy()

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek

    df["day_of_month"] = df["date"].dt.day

    # Lag features
    df["lag_1"] = df["demand"].shift(1)

    df["lag_2"] = df["demand"].shift(2)

    df["lag_3"] = df["demand"].shift(3)

    # Rolling averages
    df["rolling_3"] = (
        df["demand"]
        .rolling(window=3)
        .mean()
    )

    df["rolling_7"] = (
        df["demand"]
        .rolling(window=7)
        .mean()
    )

    # Remove rows with missing lag values
    df = df.dropna()

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(df):

    feature_columns = [
        "patient_footfall",
        "day_of_week",
        "day_of_month",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_3",
        "rolling_7",
    ]

    X = df[feature_columns]

    y = df["demand"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model


# ============================================================
# FORECAST NEXT 7 DAYS
# ============================================================

def forecast_demand(
    phc_id,
    medicine_name,
    days=7
):

    # --------------------------------------------------------
    # Load historical data
    # --------------------------------------------------------

    df = get_demand_history(
        phc_id,
        medicine_name
    )

    # --------------------------------------------------------
    # Prepare training data
    # --------------------------------------------------------

    training_df = prepare_features(df)

    if len(training_df) < 10:

        raise ValueError(
            "Not enough historical data "
            "to train forecasting model."
        )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = train_model(training_df)

    # --------------------------------------------------------
    # Prepare data for recursive prediction
    # --------------------------------------------------------

    working_df = df.copy()

    predictions = []

    last_date = working_df["date"].max()

    # Average patient footfall
    avg_footfall = (
        working_df["patient_footfall"]
        .tail(7)
        .mean()
    )

    # --------------------------------------------------------
    # Predict one day at a time
    # --------------------------------------------------------

    for i in range(1, days + 1):

        future_date = (
            last_date
            + timedelta(days=i)
        )

        demand_values = (
            working_df["demand"]
            .tolist()
        )

        lag_1 = demand_values[-1]

        lag_2 = demand_values[-2]

        lag_3 = demand_values[-3]

        rolling_3 = np.mean(
            demand_values[-3:]
        )

        rolling_7 = np.mean(
            demand_values[-7:]
        )

        features = pd.DataFrame([{
            "patient_footfall":
                avg_footfall,

            "day_of_week":
                future_date.dayofweek,

            "day_of_month":
                future_date.day,

            "lag_1":
                lag_1,

            "lag_2":
                lag_2,

            "lag_3":
                lag_3,

            "rolling_3":
                rolling_3,

            "rolling_7":
                rolling_7,
        }])

        prediction = model.predict(
            features
        )[0]

        prediction = max(
            1,
            round(float(prediction), 2)
        )

        predictions.append({
            "date":
                future_date.strftime(
                    "%Y-%m-%d"
                ),

            "predicted_demand":
                prediction,
        })

        # Add prediction to working dataset
        working_df.loc[
            len(working_df)
        ] = {
            "date":
                future_date,

            "demand":
                prediction,

            "patient_footfall":
                avg_footfall,
        }

    return predictions


# ============================================================
# SAVE FORECAST TO FIRESTORE
# ============================================================

def save_forecast(
    phc_id,
    medicine_name,
    predictions
):

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
    }

    db.collection(
        "forecasts"
    ).document(
        forecast_id
    ).set(data)

    return forecast_id


# ============================================================
# COMPLETE FORECAST PIPELINE
# ============================================================

def generate_forecast(
    phc_id,
    medicine_name,
    days=7
):

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
            medicine_name,

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
        "      AROGYAFLOW AI FORECAST TEST"
    )

    print(
        "======================================\n"
    )

    result = generate_forecast(
        phc_id="PHC001",
        medicine_name="Paracetamol",
        days=7,
    )

    print(
        f"🏥 PHC: {result['phc_id']}"
    )

    print(
        f"💊 Medicine: "
        f"{result['medicine_name']}"
    )

    print("\n📈 7-Day Forecast:\n")

    for prediction in result[
        "predictions"
    ]:

        print(
            f"  {prediction['date']} "
            f"→ "
            f"{prediction['predicted_demand']} units"
        )

    print(
        "\n🔥 Forecast saved to Firestore."
    )