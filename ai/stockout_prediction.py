import sys
import os
import math

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from datetime import datetime, timedelta

from firebase_config import db

from ai.forecasting import forecast_demand

from ai.model_service import predict_stockout


# ============================================================
# MEDICINE INFORMATION
# ============================================================

MEDICINE_INFO = {
    "Paracetamol": {
        "medicine_id": "MED-001",
        "category": "Analgesic/Antipyretic"
    },
    "Amoxicillin": {
        "medicine_id": "MED-002",
        "category": "Antibiotic"
    },
    "ORS Sachet": {
        "medicine_id": "MED-003",
        "category": "Rehydration"
    },
    "Metformin": {
        "medicine_id": "MED-004",
        "category": "Antidiabetic"
    },
    "Amlodipine": {
        "medicine_id": "MED-005",
        "category": "Antihypertensive"
    },
    "Cetirizine": {
        "medicine_id": "MED-006",
        "category": "Antihistamine"
    },
    "Iron Folic Acid Tablet": {
        "medicine_id": "MED-007",
        "category": "Supplement"
    },
    "Omeprazole": {
        "medicine_id": "MED-008",
        "category": "Gastrointestinal"
    }
}


# ============================================================
# GET CURRENT INVENTORY
# ============================================================

def get_current_inventory(
    phc_id,
    medicine_name
):
    """
    Fetch current medicine inventory
    from Firestore.
    """

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
            f"Medicine inventory not found: "
            f"{medicine_id}"
        )

    data = document.to_dict()

    return data


# ============================================================
# GET PHC INFORMATION
# ============================================================

def get_phc_information(phc_id):
    """
    Fetch PHC information from Firestore.
    """

    document = (
        db.collection("phcs")
        .document(phc_id)
        .get()
    )

    if not document.exists:

        raise ValueError(
            f"PHC not found in Firestore: {phc_id}"
        )

    return document.to_dict()


# ============================================================
# GET DEMAND HISTORY
# ============================================================

def get_recent_demand_history(
    phc_id,
    medicine_name,
    limit=30
):
    """
    Fetch recent demand history for
    the PHC and medicine.
    """

    query = (
        db.collection("demand_history")
        .where(
            "phc_id",
            "==",
            phc_id
        )
        .where(
            "medicine_name",
            "==",
            medicine_name
        )
        .order_by(
            "date",
            direction="DESCENDING"
        )
        .limit(limit)
    )

    documents = query.stream()

    history = []

    for document in documents:

        data = document.to_dict()

        history.append(data)

    # Oldest → newest
    history.reverse()

    return history


# ============================================================
# SAFE NUMERIC VALUE
# ============================================================

def safe_float(
    value,
    default=0.0
):
    """
    Safely convert value to float.
    """

    try:

        if value is None:
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):

        return default


# ============================================================
# GET FIELD USING MULTIPLE POSSIBLE NAMES
# ============================================================

def get_field(
    data,
    names,
    default=None
):
    """
    Get the first available field from a dictionary.
    """

    for name in names:

        if name in data:

            value = data[name]

            if value is not None:

                return value

    return default


# ============================================================
# CALCULATE DEMAND FEATURES
# ============================================================

def calculate_demand_features(
    history
):
    """
    Calculate lag and rolling demand features.
    """

    demands = []

    for item in history:

        demand = safe_float(
            get_field(
                item,
                ["demand"],
                0
            )
        )

        demands.append(demand)

    if len(demands) == 0:

        return {
            "Lag_1_Demand": 0,
            "Lag_7_Demand": 0,
            "Rolling_7_Demand": 0,
            "Demand_Change": 0
        }

    # Latest demand
    latest_demand = demands[-1]

    # Previous day
    lag_1 = (
        demands[-2]
        if len(demands) >= 2
        else latest_demand
    )

    # Seven days ago
    lag_7 = (
        demands[-8]
        if len(demands) >= 8
        else demands[0]
    )

    # Seven-day rolling average
    recent_7 = demands[-7:]

    rolling_7 = (
        sum(recent_7)
        / len(recent_7)
    )

    demand_change = (
        latest_demand - lag_1
    )

    return {

        "Lag_1_Demand":
            round(lag_1, 4),

        "Lag_7_Demand":
            round(lag_7, 4),

        "Rolling_7_Demand":
            round(rolling_7, 4),

        "Demand_Change":
            round(demand_change, 4)
    }


# ============================================================
# GET LATEST HISTORY RECORD
# ============================================================

def get_latest_history(
    history
):
    """
    Return latest demand-history record.
    """

    if not history:

        return {}

    return history[-1]


# ============================================================
# BUILD STOCKOUT ML FEATURES
# ============================================================

def build_stockout_features(
    phc_id,
    medicine_name,
    inventory,
    phc_data,
    history
):
    """
    Build the exact 22 features expected
    by the trained Random Forest stockout model.
    """

    # --------------------------------------------------------
    # Medicine information
    # --------------------------------------------------------

    medicine_info = MEDICINE_INFO.get(
        medicine_name,
        {
            "medicine_id": "MED-001",
            "category": "Analgesic/Antipyretic"
        }
    )

    medicine_id = medicine_info[
        "medicine_id"
    ]

    category = medicine_info[
        "category"
    ]

    # --------------------------------------------------------
    # Latest history
    # --------------------------------------------------------

    latest = get_latest_history(
        history
    )

    # --------------------------------------------------------
    # Current date
    # --------------------------------------------------------

    now = datetime.now()

    # --------------------------------------------------------
    # PHC information
    # --------------------------------------------------------

    state = get_field(
        phc_data,
        [
            "State",
            "state",
            "state_name"
        ],
        "Maharashtra"
    )

    phc_type = get_field(
        phc_data,
        [
            "PHC_Type",
            "phc_type",
            "type"
        ],
        "PHC"
    )

    urban_rural = get_field(
        phc_data,
        [
            "Urban_Rural",
            "urban_rural",
            "area_type"
        ],
        "Rural"
    )

    population_factor = safe_float(
        get_field(
            phc_data,
            [
                "Population_Factor",
                "population_factor"
            ],
            1.0
        ),
        1.0
    )

    capacity_factor = safe_float(
        get_field(
            phc_data,
            [
                "Capacity_Factor",
                "capacity_factor"
            ],
            1.0
        ),
        1.0
    )

    # --------------------------------------------------------
    # Patient footfall
    # --------------------------------------------------------

    patient_footfall = safe_float(
        get_field(
            latest,
            [
                "patient_footfall",
                "Patient_Footfall",
                "footfall"
            ],
            0
        )
    )

    # --------------------------------------------------------
    # Demand features
    # --------------------------------------------------------

    demand_features = calculate_demand_features(
        history
    )

    lag_1 = demand_features[
        "Lag_1_Demand"
    ]

    lag_7 = demand_features[
        "Lag_7_Demand"
    ]

    rolling_7 = demand_features[
        "Rolling_7_Demand"
    ]

    demand_change = demand_features[
        "Demand_Change"
    ]

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    current_stock = safe_float(
        inventory.get(
            "current_stock",
            0
        )
    )

    daily_demand = safe_float(
        inventory.get(
            "daily_demand",
            rolling_7
        )
    )

    # If daily demand is zero,
    # use rolling demand.
    if daily_demand <= 0:

        daily_demand = max(
            rolling_7,
            1.0
        )

    # --------------------------------------------------------
    # Opening stock
    # --------------------------------------------------------

    opening_stock = safe_float(
        inventory.get(
            "opening_stock",
            current_stock
        )
    )

    # --------------------------------------------------------
    # Received stock
    # --------------------------------------------------------

    received_stock = safe_float(
        inventory.get(
            "received_stock",
            0
        )
    )

    # --------------------------------------------------------
    # Days of stock
    # --------------------------------------------------------

    days_of_stock = (
        current_stock
        / max(daily_demand, 1.0)
    )

    # --------------------------------------------------------
    # Outbreak flag
    # --------------------------------------------------------

    outbreak_flag = safe_float(
        get_field(
            latest,
            [
                "outbreak_flag",
                "Outbreak_Flag"
            ],
            0
        )
    )

    # --------------------------------------------------------
    # Supply disruption flag
    # --------------------------------------------------------

    supply_disruption_flag = safe_float(
        get_field(
            latest,
            [
                "supply_disruption_flag",
                "Supply_Disruption_Flag"
            ],
            0
        )
    )

    # --------------------------------------------------------
    # Seasonal factor
    # --------------------------------------------------------

    seasonal_factor = safe_float(
        get_field(
            latest,
            [
                "seasonal_factor",
                "Seasonal_Factor"
            ],
            1.0
        ),
        1.0
    )

    # --------------------------------------------------------
    # Build exact 22-feature row
    # --------------------------------------------------------

    features = {

        "State":
            state,

        "PHC_Type":
            phc_type,

        "Urban_Rural":
            urban_rural,

        "Medicine_ID":
            medicine_id,

        "Category":
            category,

        "Day_of_Week":
            now.weekday(),

        "Day_of_Month":
            now.day,

        "Month":
            now.month,

        "Week_of_Year":
            now.isocalendar().week,

        "Patient_Footfall":
            patient_footfall,

        "Population_Factor":
            population_factor,

        "Capacity_Factor":
            capacity_factor,

        "Lag_1_Demand":
            lag_1,

        "Lag_7_Demand":
            lag_7,

        "Rolling_7_Demand":
            rolling_7,

        "Demand_Change":
            demand_change,

        "Opening_Stock":
            opening_stock,

        "Received_Stock":
            received_stock,

        "Days_of_Stock":
            days_of_stock,

        "Outbreak_Flag":
            outbreak_flag,

        "Supply_Disruption_Flag":
            supply_disruption_flag,

        "Seasonal_Factor":
            seasonal_factor
    }

    return features


# ============================================================
# CALCULATE STOCK-OUT FROM FORECAST
# ============================================================

def calculate_stockout(
    current_stock,
    predictions
):
    """
    Determine approximately when
    medicine stock will run out.
    """

    remaining_stock = float(
        current_stock
    )

    cumulative_demand = 0

    for index, prediction in enumerate(
        predictions
    ):

        daily_demand = float(
            prediction["predicted_demand"]
        )

        previous_stock = remaining_stock

        remaining_stock -= daily_demand

        cumulative_demand += daily_demand

        # ----------------------------------------------------
        # Stock-out happens during this day
        # ----------------------------------------------------

        if remaining_stock <= 0:

            if daily_demand > 0:

                fraction_of_day = (
                    previous_stock
                    / daily_demand
                )

            else:

                fraction_of_day = 0

            days_until_stockout = (
                index
                + fraction_of_day
            )

            return {

                "will_stockout": True,

                "days_until_stockout":
                    round(
                        days_until_stockout,
                        2
                    ),

                "stockout_date":
                    prediction["date"],

                "remaining_stock":
                    round(
                        max(
                            0,
                            remaining_stock
                        ),
                        2
                    ),

                "cumulative_demand":
                    round(
                        cumulative_demand,
                        2
                    )
            }

    # --------------------------------------------------------
    # No stock-out inside forecast window
    # --------------------------------------------------------

    return {

        "will_stockout": False,

        "days_until_stockout": None,

        "stockout_date": None,

        "remaining_stock":
            round(
                remaining_stock,
                2
            ),

        "cumulative_demand":
            round(
                cumulative_demand,
                2
            )
    }


# ============================================================
# DETERMINE FINAL RISK
# ============================================================

def determine_risk(
    stockout_result,
    ml_result,
    current_stock,
    minimum_stock
):
    """
    Combine:
    1. ML stockout probability
    2. Forecast-based stock depletion
    3. Minimum stock threshold
    """

    ml_probability = safe_float(
        ml_result.get(
            "stockout_probability",
            0
        )
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if current_stock <= 0:

        return "CRITICAL"

    if ml_probability >= 0.80:

        return "CRITICAL"

    if stockout_result[
        "will_stockout"
    ]:

        days = stockout_result[
            "days_until_stockout"
        ]

        if days <= 2:

            return "CRITICAL"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if ml_probability >= 0.60:

        return "HIGH"

    if stockout_result[
        "will_stockout"
    ]:

        days = stockout_result[
            "days_until_stockout"
        ]

        if days <= 4:

            return "HIGH"

    # --------------------------------------------------------
    # Minimum stock
    # --------------------------------------------------------

    if minimum_stock > 0:

        if current_stock <= minimum_stock:

            if ml_probability >= 0.30:

                return "HIGH"

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if ml_probability >= 0.30:

        return "MEDIUM"

    if stockout_result[
        "will_stockout"
    ]:

        days = stockout_result[
            "days_until_stockout"
        ]

        if days <= 7:

            return "MEDIUM"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "LOW"


# ============================================================
# COMPLETE STOCK-OUT ANALYSIS
# ============================================================

def analyze_stockout(
    phc_id,
    medicine_name
):

    print(
        "\n🔍 Starting stockout analysis..."
    )

    # --------------------------------------------------------
    # 1. Get current inventory
    # --------------------------------------------------------

    inventory = get_current_inventory(
        phc_id,
        medicine_name
    )

    current_stock = safe_float(
        inventory.get(
            "current_stock",
            0
        )
    )

    minimum_stock = safe_float(
        inventory.get(
            "minimum_stock",
            0
        )
    )

    # --------------------------------------------------------
    # 2. Get PHC information
    # --------------------------------------------------------

    phc_data = get_phc_information(
        phc_id
    )

    # --------------------------------------------------------
    # 3. Get demand history
    # --------------------------------------------------------

    history = get_recent_demand_history(
        phc_id,
        medicine_name,
        limit=30
    )

    # --------------------------------------------------------
    # 4. Build ML features
    # --------------------------------------------------------

    stockout_features = build_stockout_features(
        phc_id,
        medicine_name,
        inventory,
        phc_data,
        history
    )

    print(
        "\n🤖 Running Random Forest stockout model..."
    )

    # --------------------------------------------------------
    # 5. Random Forest stockout prediction
    # --------------------------------------------------------

    ml_result = predict_stockout(
        stockout_features
    )

    # --------------------------------------------------------
    # 6. Generate AI demand forecast
    # --------------------------------------------------------

    print(
        "\n📈 Generating 7-day demand forecast..."
    )

    predictions = forecast_demand(
        phc_id,
        medicine_name,
        days=7
    )

    # --------------------------------------------------------
    # 7. Calculate physical stock depletion
    # --------------------------------------------------------

    stockout_result = calculate_stockout(
        current_stock,
        predictions
    )

    # --------------------------------------------------------
    # 8. Determine final risk
    # --------------------------------------------------------

    risk = determine_risk(
        stockout_result,
        ml_result,
        current_stock,
        minimum_stock
    )

    # --------------------------------------------------------
    # 9. Create final result
    # --------------------------------------------------------

    result = {

        "phc_id":
            phc_id,

        "medicine_name":
            medicine_name,

        "current_stock":
            current_stock,

        "minimum_stock":
            minimum_stock,

        "ml_prediction":
            ml_result,

        "stockout_features":
            stockout_features,

        "forecast":
            predictions,

        "stockout":
            stockout_result,

        "risk":
            risk,

        "analyzed_at":
            datetime.utcnow()
    }

    # --------------------------------------------------------
    # 10. Save to Firestore
    # --------------------------------------------------------

    analysis_id = (
        f"{phc_id}_"
        f"{medicine_name.lower().replace(' ', '_')}"
    )

    db.collection(
        "stockout_predictions"
    ).document(
        analysis_id
    ).set(
        result
    )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "      AROGYAFLOW AI STOCK-OUT TEST"
    )

    print(
        "==========================================\n"
    )

    try:

        result = analyze_stockout(
            phc_id="PHC001",
            medicine_name="Paracetamol"
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
            f"\n📦 Current Stock: "
            f"{result['current_stock']}"
        )

        print(
            f"📉 Minimum Stock: "
            f"{result['minimum_stock']}"
        )

        # ----------------------------------------------------
        # ML RESULT
        # ----------------------------------------------------

        ml = result[
            "ml_prediction"
        ]

        print(
            "\n🤖 ML STOCKOUT PREDICTION"
        )

        print(
            "------------------------------------------"
        )

        print(
            f"Prediction: "
            f"{ml['stockout_prediction']}"
        )

        print(
            f"Probability: "
            f"{ml['stockout_probability_percent']}%"
        )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        stockout = result[
            "stockout"
        ]

        print(
            "\n📊 FORECAST-BASED ANALYSIS"
        )

        print(
            "------------------------------------------"
        )

        print(
            f"Cumulative Demand: "
            f"{stockout['cumulative_demand']}"
        )

        if stockout[
            "will_stockout"
        ]:

            print(
                "⚠️ Stock-out predicted!"
            )

            print(
                f"⏳ Days remaining: "
                f"{stockout['days_until_stockout']}"
            )

            print(
                f"📅 Stock-out date: "
                f"{stockout['stockout_date']}"
            )

        else:

            print(
                "✅ No physical stock-out "
                "within 7-day forecast."
            )

            print(
                f"📦 Remaining after forecast: "
                f"{stockout['remaining_stock']}"
            )

        # ----------------------------------------------------
        # FINAL RISK
        # ----------------------------------------------------

        print(
            "\n🚨 FINAL RISK:"
        )

        print(
            f"   {result['risk']}"
        )

        print(
            "\n🔥 Analysis saved to Firestore."
        )

    except Exception as error:

        print(
            "\n❌ STOCKOUT ANALYSIS FAILED"
        )

        print(
            f"Error: {error}"
        )

        raise