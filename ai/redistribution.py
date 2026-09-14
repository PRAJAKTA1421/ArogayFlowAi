import sys
import os
import math
from datetime import datetime
from google.cloud.firestore_v1.base_query import FieldFilter
# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from firebase_config import db

from ai.forecasting import forecast_demand
from ai.stockout import get_current_inventory


# ============================================================
# GET ALL PHCs
# ============================================================

def get_all_phcs():

    documents = (
        db.collection("phcs")
        .stream()
    )

    phcs = []

    for document in documents:

        data = document.to_dict()

        data["phc_id"] = document.id

        phcs.append(data)

    return phcs


# ============================================================
# GET MEDICINE INVENTORY
# ============================================================

def get_all_medicine_inventory(
    medicine_name
):

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

        data = document.to_dict()

        data["medicine_id"] = document.id

        inventory.append(data)

    return inventory


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

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
    Calculate the great-circle distance between
    two PHCs using the Haversine formula.

    Returns:
        Distance in kilometers.
    """

    R = 6371.0

    # Convert latitude and longitude from
    # degrees to radians exactly once.
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    # Difference between coordinates
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c

# ============================================================
# CALCULATE PHC SURPLUS
# ============================================================

def calculate_surplus(
    current_stock,
    predicted_demand
):

    surplus = (
        current_stock
        - predicted_demand
    )

    return max(
        0,
        surplus
    )


# ============================================================
# FIND BEST SOURCE PHCs
# ============================================================

def find_source_phcs(
    destination_phc_id,
    medicine_name
):

    phcs = get_all_phcs()

    inventories = get_all_medicine_inventory(
        medicine_name
    )

    # --------------------------------------------------------
    # Destination PHC
    # --------------------------------------------------------

    destination = None

    for phc in phcs:

        if phc["phc_id"] == destination_phc_id:

            destination = phc

            break

    if not destination:

        raise ValueError(
            f"PHC not found: "
            f"{destination_phc_id}"
        )

    # --------------------------------------------------------
    # Destination forecast
    # --------------------------------------------------------

    destination_forecast = forecast_demand(
        destination_phc_id,
        medicine_name,
        days=7
    )

    destination_demand = sum(
        item["predicted_demand"]
        for item in destination_forecast
    )

    destination_inventory = (
        get_current_inventory(
            destination_phc_id,
            medicine_name
        )
    )

    destination_stock = float(
        destination_inventory[
            "current_stock"
        ]
    )

    destination_shortage = max(
        0,
        destination_demand
        - destination_stock
    )

    # --------------------------------------------------------
    # If there is no shortage
    # --------------------------------------------------------

    if destination_shortage <= 0:

        return {
            "destination_phc":
                destination_phc_id,

            "shortage":
                0,

            "recommendations":
                [],
        }

    # --------------------------------------------------------
    # Find source PHCs
    # --------------------------------------------------------

    recommendations = []

    for inventory in inventories:

        source_phc_id = inventory[
            "phc_id"
        ]

        # Don't transfer from destination itself
        if source_phc_id == destination_phc_id:

            continue

        source_stock = float(
            inventory[
                "current_stock"
            ]
        )

        # ----------------------------------------------------
        # Source forecast
        # ----------------------------------------------------

        source_forecast = forecast_demand(
            source_phc_id,
            medicine_name,
            days=7
        )

        source_demand = sum(
            item["predicted_demand"]
            for item in source_forecast
        )

        # ----------------------------------------------------
        # Calculate surplus
        # ----------------------------------------------------

        surplus = calculate_surplus(
            source_stock,
            source_demand
        )

        if surplus <= 0:

            continue

        # ----------------------------------------------------
        # Find source PHC coordinates
        # ----------------------------------------------------

        source_phc = None

        for phc in phcs:

            if phc["phc_id"] == source_phc_id:

                source_phc = phc

                break

        if not source_phc:

            continue

        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        distance = calculate_distance(
            destination["latitude"],
            destination["longitude"],
            source_phc["latitude"],
            source_phc["longitude"]
        )

        # ----------------------------------------------------
        # Transfer quantity
        # ----------------------------------------------------

        transfer_quantity = min(
            surplus,
            destination_shortage
        )

        # --------------------------------------------------------
        # AI SCORING
        # --------------------------------------------------------

        # Normalize surplus relative to the destination shortage.
        # A source with enough surplus to cover the shortage
        # receives the maximum surplus score.

        surplus_score = min(
            surplus / max(destination_shortage, 1),
            1.0
        )

        # Distance score.
        # Closer PHCs receive significantly higher scores.
        # 30 km is used as the practical distance scale.
        # This makes local transfers strongly preferable
        # to distant transfers.

        distance_score = 1 / (
            1 + distance / 30
        )

        # --------------------------------------------------------
        # Combined AI score
        # --------------------------------------------------------
        #
        # Distance = 75%
        # Surplus  = 25%
        #
        # Healthcare redistribution should prioritize
        # geographically practical transfers.

        score = (
            (distance_score * 0.75)
            +
            (surplus_score * 0.25)
        ) * 100

        recommendations.append({

            "source_phc":
                source_phc_id,

            "source_name":
                source_phc["name"],

            "destination_phc":
                destination_phc_id,

            "destination_name":
                destination["name"],

            "medicine":
                medicine_name,

            "source_stock":
                source_stock,

            "source_predicted_demand":
                round(
                    source_demand,
                    2
                ),

            "source_surplus":
                round(
                    surplus,
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

            "score":
                round(
                    score,
                    2
                ),
        })

    # --------------------------------------------------------
    # Sort recommendations
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Sort by AI score
    # --------------------------------------------------------

    recommendations.sort(
       key=lambda x: x["score"],
       reverse=True
   )

    # --------------------------------------------------------
    # Create actual transfer plan
    # --------------------------------------------------------

    transfer_result = create_transfer_plan(
        destination_shortage,
        recommendations
    )

    return {

        "destination_phc":
            destination_phc_id,

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
    }

# ============================================================
# SAVE RECOMMENDATION
# ============================================================

def create_transfer_plan(
    shortage,
    recommendations
):
    """
    Build a practical transfer plan that
    satisfies the destination shortage
    using the best available source PHCs.
    """

    remaining = shortage

    plan = []

    for recommendation in recommendations:

        if remaining <= 0:
            break

        available = float(
            recommendation[
                "source_surplus"
            ]
        )

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

    return {
        "plan": plan,

        "remaining_shortage":
            round(
                max(0, remaining),
                2
            ),

        "total_transfer":
            round(
                shortage - max(0, remaining),
                2
            )
    }

def save_recommendation(
    result
):

    document_id = (
        f"{result['destination_phc']}_"
        f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    )

    data = {

        **result,

        "created_at":
            datetime.utcnow(),

        "status":
            "pending",
    }

    db.collection(
        "transfer_recommendations"
    ).document(
        document_id
    ).set(data)

    return document_id


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "   AROGYAFLOW AI REDISTRIBUTION TEST"
    )

    print(
        "======================================\n"
    )

    destination = "PHC001"

    medicine = "Paracetamol"

    print(
        f"🎯 Destination: {destination}"
    )

    print(
        f"💊 Medicine: {medicine}\n"
    )

    result = find_source_phcs(
        destination,
        medicine
    )

    print(
        f"📦 Current Stock: "
        f"{result.get('current_stock', 0)}"
    )

    print(
        f"📈 7-Day Demand: "
        f"{result.get('forecasted_7_day_demand', 0)}"
    )

    print(
        f"🚨 Shortage: "
        f"{result.get('shortage', 0)}"
    )

    print(
        "\n🤖 AI TRANSFER PLAN\n"
    )

    transfer_plan = result.get(
        "transfer_plan",
        []
    )

    if not transfer_plan:

        print(
            "❌ No suitable source PHC found."
        )

    else:

        for index, recommendation in enumerate(
            transfer_plan,
            start=1
        ):

            print(
                f"{index}. "
                f"{recommendation['source_name']} "
                f"→ "
                f"{recommendation['destination_name']}"
            )

            print(
                f"   💊 Quantity: "
                f"{recommendation['recommended_quantity']} "
                f"units"
            )

            print(
                f"   📦 Source surplus: "
                f"{recommendation['source_surplus']}"
            )

            print(
                f"   📍 Distance: "
                f"{recommendation['distance_km']} km"
            )

            print(
                f"   🤖 AI Score: "
                f"{recommendation['score']}"
            )

            print()

    print(
        f"📦 Total Transfer: "
        f"{result.get('total_transfer', 0)} units"
    )

    print(
        f"⚠️ Remaining Shortage: "
        f"{result.get('remaining_shortage', 0)} units"
    )

    document_id = save_recommendation(
        result
    )

    print(
        f"🔥 Recommendation saved: "
        f"{document_id}"
    )