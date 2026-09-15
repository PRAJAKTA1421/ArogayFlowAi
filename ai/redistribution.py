import sys
import os
import math
from datetime import datetime

from google.cloud.firestore_v1.base_query import FieldFilter


# ============================================================
# PROJECT ROOT
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from firebase_config import db
from ai.forecasting import forecast_demand
from ai.stockout_prediction import get_current_inventory


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_DAYS = 7

# Source PHC safety-stock buffer.
#
# The source PHC must retain:
#
#     7-day predicted demand
#     +
#     safety stock
#
# Safety stock = 25% of predicted demand.
#
SAFETY_STOCK_RATIO = 0.25

# Optimization weights
DISTANCE_WEIGHT = 0.75
SURPLUS_WEIGHT = 0.25

# Distance scale used by the optimization score.
DISTANCE_SCALE_KM = 30.0


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):

        return default


# ============================================================
# GET ALL PHCs
# ============================================================

def get_all_phcs():
    """
    Retrieve all PHCs from Firestore.
    """

    documents = (
        db.collection("phcs")
        .stream()
    )

    phcs = []

    for document in documents:

        data = document.to_dict() or {}

        data["phc_id"] = document.id

        phcs.append(data)

    return phcs


# ============================================================
# GET ALL MEDICINE INVENTORY
# ============================================================

def get_all_medicine_inventory(medicine_name):
    """
    Retrieve medicine inventory for all PHCs.
    """

    documents = (
        db.collection("medicines")
        .where(
            filter=FieldFilter(
                "medicine_name",
                "==",
                medicine_name
            )
        )
        .stream()
    )

    inventory = []

    for document in documents:

        data = document.to_dict() or {}

        data["medicine_id"] = document.id

        inventory.append(data)

    return inventory


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between two PHCs
    using the Haversine formula.

    Returns:
        Distance in kilometres.
    """

    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)

    R = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = max(
        0.0,
        min(1.0, a)
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# FIND PHC BY ID
# ============================================================

def find_phc_by_id(
    phcs,
    phc_id
):
    """
    Find a PHC from a list of PHCs.
    """

    for phc in phcs:

        if phc.get("phc_id") == phc_id:

            return phc

    return None


# ============================================================
# CALCULATE SAFETY STOCK
# ============================================================

def calculate_safety_stock(
    predicted_demand
):
    """
    Safety stock is calculated as 25% of
    predicted 7-day demand.
    """

    predicted_demand = safe_float(
        predicted_demand
    )

    return max(
        0.0,
        predicted_demand * SAFETY_STOCK_RATIO
    )


# ============================================================
# CALCULATE SOURCE RAW SURPLUS
# ============================================================

def calculate_surplus(
    current_stock,
    predicted_demand
):
    """
    Raw surplus after predicted demand.

    Raw Surplus =
        Current Stock - Predicted Demand
    """

    current_stock = safe_float(
        current_stock
    )

    predicted_demand = safe_float(
        predicted_demand
    )

    return max(
        0.0,
        current_stock - predicted_demand
    )


# ============================================================
# CALCULATE SOURCE SAFE SURPLUS
# ============================================================

def calculate_safe_surplus(
    current_stock,
    predicted_demand,
    safety_stock=None
):
    """
    Calculate the amount that can safely be transferred
    from a source PHC.

    Formula:

        Safe Surplus =
            Current Stock
            - Predicted Demand
            - Safety Stock
    """

    current_stock = safe_float(
        current_stock
    )

    predicted_demand = safe_float(
        predicted_demand
    )

    if safety_stock is None:

        safety_stock = calculate_safety_stock(
            predicted_demand
        )

    safety_stock = safe_float(
        safety_stock
    )

    safe_surplus = (
        current_stock
        - predicted_demand
        - safety_stock
    )

    return max(
        0.0,
        safe_surplus
    )


# ============================================================
# CALCULATE DESTINATION SAFETY-STOCK SHORTAGE
# ============================================================

def calculate_destination_shortage(
    current_stock,
    predicted_demand,
    minimum_stock
):
    """
    Calculate destination shortage using the minimum
    stock requirement.

    Formula:

        Forecast Remaining =
            Current Stock - Predicted Demand

        Safety-Stock Shortage =
            Minimum Stock - Forecast Remaining

    If the forecasted remaining stock is above the
    minimum stock, shortage is zero.
    """

    current_stock = safe_float(
        current_stock
    )

    predicted_demand = safe_float(
        predicted_demand
    )

    minimum_stock = max(
        0.0,
        safe_float(minimum_stock)
    )

    forecast_remaining_stock = (
        current_stock
        - predicted_demand
    )

    shortage = (
        minimum_stock
        - forecast_remaining_stock
    )

    return max(
        0.0,
        shortage
    )


# ============================================================
# CALCULATE OPTIMIZATION SCORE
# ============================================================

def calculate_optimization_score(
    distance_km,
    safe_surplus,
    destination_shortage
):
    """
    Calculate redistribution optimization score.

    Distance:
        75%

    Safe surplus:
        25%

    Final score:
        0 - 100
    """

    distance_km = max(
        0.0,
        safe_float(distance_km)
    )

    safe_surplus = max(
        0.0,
        safe_float(safe_surplus)
    )

    destination_shortage = max(
        0.0,
        safe_float(destination_shortage)
    )

    # --------------------------------------------------------
    # Distance score
    # --------------------------------------------------------

    distance_score = 1.0 / (
        1.0
        +
        distance_km / DISTANCE_SCALE_KM
    )

    # --------------------------------------------------------
    # Safe surplus score
    # --------------------------------------------------------

    if destination_shortage > 0:

        surplus_score = min(
            safe_surplus / destination_shortage,
            1.0
        )

    else:

        surplus_score = 0.0

    # --------------------------------------------------------
    # Weighted score
    # --------------------------------------------------------

    score = (
        distance_score * DISTANCE_WEIGHT
        +
        surplus_score * SURPLUS_WEIGHT
    ) * 100.0

    return round(
        score,
        2
    )


# ============================================================
# FIND SOURCE PHCs
# ============================================================

def find_source_phcs(
    destination_phc_id,
    medicine_name
):
    """
    Find suitable source PHCs for a destination PHC.

    Destination shortage is based on the minimum stock
    requirement after the 7-day demand forecast.
    """

    print(
        "\n🔍 Starting redistribution optimization..."
    )

    # ========================================================
    # LOAD PHCs
    # ========================================================

    phcs = get_all_phcs()

    # Create fast PHC lookup
    phc_lookup = {
        phc.get("phc_id"): phc
        for phc in phcs
        if phc.get("phc_id")
    }

    # ========================================================
    # LOAD MEDICINE INVENTORY
    # ========================================================

    inventories = get_all_medicine_inventory(
        medicine_name
    )

    # ========================================================
    # DESTINATION PHC
    # ========================================================

    destination = phc_lookup.get(
        destination_phc_id
    )

    if not destination:

        raise ValueError(
            f"PHC not found: {destination_phc_id}"
        )

    destination_name = destination.get(
        "name",
        destination_phc_id
    )

    # ========================================================
    # DESTINATION FORECAST
    # ========================================================

    print(
        f"📈 Forecasting destination demand "
        f"for {medicine_name}..."
    )

    destination_forecast = forecast_demand(
        destination_phc_id,
        medicine_name,
        days=FORECAST_DAYS
    )

    destination_demand = sum(
        safe_float(
            item.get(
                "predicted_demand",
                0
            )
        )
        for item in destination_forecast
    )

    # ========================================================
    # DESTINATION INVENTORY
    # ========================================================

    destination_inventory = get_current_inventory(
        destination_phc_id,
        medicine_name
    )

    destination_stock = safe_float(
        destination_inventory.get(
            "current_stock",
            0
        )
    )

    # ========================================================
    # MINIMUM STOCK
    # ========================================================

    minimum_stock = safe_float(
        destination_inventory.get(
            "minimum_stock",
            0
        )
    )

    # --------------------------------------------------------
    # Fallback
    #
    # If minimum_stock isn't present in the inventory
    # document, use zero rather than inventing a value.
    # --------------------------------------------------------

    forecast_remaining_stock = (
        destination_stock
        - destination_demand
    )

    # ========================================================
    # DESTINATION SHORTAGE
    # ========================================================

    destination_shortage = (
        calculate_destination_shortage(
            destination_stock,
            destination_demand,
            minimum_stock
        )
    )

    print(
        f"📦 Destination stock: "
        f"{round(destination_stock, 2)}"
    )

    print(
        f"📊 Minimum stock: "
        f"{round(minimum_stock, 2)}"
    )

    print(
        f"📈 7-day predicted demand: "
        f"{round(destination_demand, 2)}"
    )

    print(
        f"📦 Forecast remaining stock: "
        f"{round(forecast_remaining_stock, 2)}"
    )

    print(
        f"🚨 Safety-stock shortage: "
        f"{round(destination_shortage, 2)}"
    )

    # ========================================================
    # NO REDISTRIBUTION REQUIRED
    # ========================================================

    if destination_shortage <= 0:

        print(
            "✅ Destination remains above minimum stock."
        )

        return {

            "destination_phc":
                destination_phc_id,

            "destination_name":
                destination_name,

            "medicine":
                medicine_name,

            "shortage":
                0,

            "forecasted_7_day_demand":
                round(
                    destination_demand,
                    2
                ),

            "current_stock":
                round(
                    destination_stock,
                    2
                ),

            "minimum_stock":
                round(
                    minimum_stock,
                    2
                ),

            "forecast_remaining_stock":
                round(
                    forecast_remaining_stock,
                    2
                ),

            "recommendations":
                [],

            "transfer_plan":
                [],

            "total_transfer":
                0,

            "remaining_shortage":
                0,

            "action":
                "NO_ACTION_REQUIRED"
        }

    # ========================================================
    # FIND SOURCE PHCs
    # ========================================================

    recommendations = []

    print(
        "\n🔎 Searching for suitable source PHCs..."
    )

    for inventory in inventories:

        source_phc_id = inventory.get(
            "phc_id"
        )

        # ----------------------------------------------------
        # Invalid source
        # ----------------------------------------------------

        if not source_phc_id:

            continue

        # ----------------------------------------------------
        # Don't transfer from destination itself
        # ----------------------------------------------------

        if source_phc_id == destination_phc_id:

            continue

        # ----------------------------------------------------
        # Find source PHC
        # ----------------------------------------------------

        source_phc = phc_lookup.get(
            source_phc_id
        )

        if not source_phc:

            continue

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        source_lat = source_phc.get(
            "latitude"
        )

        source_lon = source_phc.get(
            "longitude"
        )

        destination_lat = destination.get(
            "latitude"
        )

        destination_lon = destination.get(
            "longitude"
        )

        if (
            source_lat is None
            or source_lon is None
            or destination_lat is None
            or destination_lon is None
        ):

            continue

        # ----------------------------------------------------
        # Source current stock
        # ----------------------------------------------------

        source_stock = safe_float(
            inventory.get(
                "current_stock",
                0
            )
        )

        if source_stock <= 0:

            continue

        # ====================================================
        # SOURCE FORECAST
        # ====================================================

        try:

            source_forecast = forecast_demand(
                source_phc_id,
                medicine_name,
                days=FORECAST_DAYS
            )

        except Exception as error:

            print(
                f"⚠️ Forecast failed for "
                f"{source_phc_id}: {error}"
            )

            continue

        source_demand = sum(
            safe_float(
                item.get(
                    "predicted_demand",
                    0
                )
            )
            for item in source_forecast
        )

        # ====================================================
        # SOURCE SAFETY STOCK
        # ====================================================

        source_safety_stock = (
            calculate_safety_stock(
                source_demand
            )
        )

        # ====================================================
        # SOURCE RAW SURPLUS
        # ====================================================

        source_raw_surplus = (
            calculate_surplus(
                source_stock,
                source_demand
            )
        )

        # ====================================================
        # SOURCE SAFE SURPLUS
        # ====================================================

        source_safe_surplus = (
            calculate_safe_surplus(
                source_stock,
                source_demand,
                source_safety_stock
            )
        )

        # ----------------------------------------------------
        # Source cannot safely transfer stock
        # ----------------------------------------------------

        if source_safe_surplus <= 0:

            continue

        # ====================================================
        # DISTANCE
        # ====================================================

        distance = calculate_distance(
            destination_lat,
            destination_lon,
            source_lat,
            source_lon
        )

        # ====================================================
        # TRANSFER QUANTITY
        # ====================================================

        transfer_quantity = min(
            source_safe_surplus,
            destination_shortage
        )

        if transfer_quantity <= 0:

            continue

        # ====================================================
        # OPTIMIZATION SCORE
        # ====================================================

        optimization_score = (
            calculate_optimization_score(
                distance,
                source_safe_surplus,
                destination_shortage
            )
        )

        # ====================================================
        # SOURCE RECOMMENDATION
        # ====================================================

        recommendation = {

            "source_phc":
                source_phc_id,

            "source_name":
                source_phc.get(
                    "name",
                    source_phc_id
                ),

            "destination_phc":
                destination_phc_id,

            "destination_name":
                destination_name,

            "medicine":
                medicine_name,

            "source_stock":
                round(
                    source_stock,
                    2
                ),

            "source_predicted_demand":
                round(
                    source_demand,
                    2
                ),

            "source_raw_surplus":
                round(
                    source_raw_surplus,
                    2
                ),

            "source_safety_stock":
                round(
                    source_safety_stock,
                    2
                ),

            "source_safe_surplus":
                round(
                    source_safe_surplus,
                    2
                ),

            "destination_shortage":
                round(
                    destination_shortage,
                    2
                ),

            "recommended_quantity":
                round(
                    transfer_quantity,
                    2
                ),

            "distance_km":
                round(
                    distance,
                    2
                ),

            "optimization_score":
                optimization_score
        }

        recommendations.append(
            recommendation
        )

    # ========================================================
    # SORT SOURCE PHCs
    # ========================================================

    recommendations.sort(
        key=lambda item:
            item.get(
                "optimization_score",
                0
            ),
        reverse=True
    )

    # ========================================================
    # CREATE TRANSFER PLAN
    # ========================================================

    transfer_result = create_transfer_plan(
        destination_shortage,
        recommendations
    )

    # ========================================================
    # DETERMINE ACTION
    # ========================================================

    if transfer_result["total_transfer"] > 0:

        if transfer_result["remaining_shortage"] > 0:

            action = "PARTIAL_REDISTRIBUTION"

        else:

            action = "REDISTRIBUTION_RECOMMENDED"

    else:

        action = "NO_SOURCE_AVAILABLE"

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "destination_phc":
            destination_phc_id,

        "destination_name":
            destination_name,

        "medicine":
            medicine_name,

        "shortage":
            round(
                destination_shortage,
                2
            ),

        "forecasted_7_day_demand":
            round(
                destination_demand,
                2
            ),

        "current_stock":
            round(
                destination_stock,
                2
            ),

        "minimum_stock":
            round(
                minimum_stock,
                2
            ),

        "forecast_remaining_stock":
            round(
                forecast_remaining_stock,
                2
            ),

        "recommendations":
            recommendations,

        "transfer_plan":
            transfer_result["plan"],

        "total_transfer":
            transfer_result[
                "total_transfer"
            ],

        "remaining_shortage":
            transfer_result[
                "remaining_shortage"
            ],

        "optimization_method":
            "Distance + Safe Surplus Weighted Optimization",

        "distance_weight":
            DISTANCE_WEIGHT,

        "surplus_weight":
            SURPLUS_WEIGHT,

        "safety_stock_ratio":
            SAFETY_STOCK_RATIO,

        "action":
            action
    }

    return result


# ============================================================
# CREATE TRANSFER PLAN
# ============================================================

def create_transfer_plan(
    shortage,
    recommendations
):
    """
    Create the actual transfer plan.

    Sources are processed in descending optimization score.

    The system transfers only the amount that the source
    can safely provide.
    """

    shortage = max(
        0.0,
        safe_float(shortage)
    )

    remaining = shortage

    plan = []

    # ========================================================
    # PROCESS SOURCES
    # ========================================================

    for recommendation in recommendations:

        if remaining <= 0:

            break

        available = safe_float(
            recommendation.get(
                "source_safe_surplus",
                0
            )
        )

        if available <= 0:

            continue

        quantity = min(
            available,
            remaining
        )

        if quantity <= 0:

            continue

        transfer = recommendation.copy()

        transfer[
            "recommended_quantity"
        ] = round(
            quantity,
            2
        )

        plan.append(
            transfer
        )

        remaining -= quantity

    # ========================================================
    # FINAL VALUES
    # ========================================================

    remaining_shortage = max(
        0.0,
        remaining
    )

    total_transfer = (
        shortage
        - remaining_shortage
    )

    return {

        "plan":
            plan,

        "remaining_shortage":
            round(
                remaining_shortage,
                2
            ),

        "total_transfer":
            round(
                total_transfer,
                2
            )
    }


# ============================================================
# SAVE RECOMMENDATION
# ============================================================

def save_recommendation(result):
    """
    Save redistribution recommendation to Firestore.
    """

    timestamp = datetime.utcnow()

    medicine = result.get(
        "medicine",
        "medicine"
    )

    safe_medicine_name = (
        medicine
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    document_id = (
        f"{result['destination_phc']}_"
        f"{safe_medicine_name}_"
        f"{timestamp.strftime('%Y%m%d%H%M%S')}"
    )

    data = {

        **result,

        "created_at":
            timestamp,

        "status":
            "pending"
    }

    db.collection(
        "transfer_recommendations"
    ).document(
        document_id
    ).set(data)

    return document_id


# ============================================================
# PRINT TRANSFER PLAN
# ============================================================

def print_transfer_plan(result):
    """
    Print the redistribution result.
    """

    print(
        "\n=========================================="
    )

    print(
        "        REDISTRIBUTION RESULT"
    )

    print(
        "=========================================="
    )

    print(
        f"🏥 Destination: "
        f"{result.get('destination_name', 'N/A')}"
    )

    print(
        f"💊 Medicine: "
        f"{result.get('medicine', 'N/A')}"
    )

    print(
        f"📦 Current Stock: "
        f"{result.get('current_stock', 0)}"
    )

    print(
        f"📦 Minimum Stock: "
        f"{result.get('minimum_stock', 0)}"
    )

    print(
        f"📈 7-Day Demand: "
        f"{result.get('forecasted_7_day_demand', 0)}"
    )

    print(
        f"📦 Forecast Remaining: "
        f"{result.get('forecast_remaining_stock', 0)}"
    )

    print(
        f"🚨 Safety-Stock Shortage: "
        f"{result.get('shortage', 0)}"
    )

    print(
        f"🎯 Action: "
        f"{result.get('action', 'N/A')}"
    )

    print(
        "\n🤖 OPTIMIZED TRANSFER PLAN"
    )

    print(
        "------------------------------------------"
    )

    transfer_plan = result.get(
        "transfer_plan",
        []
    )

    if not transfer_plan:

        if result.get("action") == "NO_SOURCE_AVAILABLE":

            print(
                "❌ No suitable source PHC found."
            )

        elif result.get("action") == "NO_ACTION_REQUIRED":

            print(
                "✅ No redistribution required."
            )

        else:

            print(
                "❌ No transfer plan generated."
            )

    else:

        for index, recommendation in enumerate(
            transfer_plan,
            start=1
        ):

            print(
                f"\n{index}. "
                f"{recommendation['source_name']}"
                f" → "
                f"{recommendation['destination_name']}"
            )

            print(
                f"   💊 Quantity: "
                f"{recommendation['recommended_quantity']} units"
            )

            print(
                f"   📦 Source Stock: "
                f"{recommendation['source_stock']}"
            )

            print(
                f"   📈 Source 7-Day Demand: "
                f"{recommendation['source_predicted_demand']}"
            )

            print(
                f"   🛡️ Source Safety Stock: "
                f"{recommendation['source_safety_stock']}"
            )

            print(
                f"   📦 Safe Surplus: "
                f"{recommendation['source_safe_surplus']}"
            )

            print(
                f"   📍 Distance: "
                f"{recommendation['distance_km']} km"
            )

            print(
                f"   🤖 Optimization Score: "
                f"{recommendation['optimization_score']}"
            )

    print(
        "\n------------------------------------------"
    )

    print(
        f"📦 Total Transfer: "
        f"{result.get('total_transfer', 0)} units"
    )

    print(
        f"⚠️ Remaining Shortage: "
        f"{result.get('remaining_shortage', 0)} units"
    )

    print(
        "=========================================="
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "   AROGYAFLOW AI REDISTRIBUTION TEST"
    )

    print(
        "==========================================\n"
    )

    destination = "PHC001"

    medicine = "Paracetamol"

    print(
        f"🎯 Destination: {destination}"
    )

    print(
        f"💊 Medicine: {medicine}\n"
    )

    try:

        result = find_source_phcs(
            destination,
            medicine
        )

        print_transfer_plan(
            result
        )

        # ====================================================
        # SAVE RESULT
        # ====================================================

        document_id = save_recommendation(
            result
        )

        print(
            "\n🔥 Recommendation saved to Firestore:"
        )

        print(
            f"   {document_id}"
        )

    except Exception as error:

        print(
            "\n❌ Redistribution failed."
        )

        print(
            f"Error: {error}"
        )

        raise