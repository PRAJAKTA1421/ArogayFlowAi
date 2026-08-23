import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from datetime import datetime

from firebase_config import db

from ai.forecasting import forecast_demand


# ============================================================
# GET CURRENT INVENTORY
# ============================================================

def get_current_inventory(
    phc_id,
    medicine_name
):
    """
    Fetch current medicine stock
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
# CALCULATE STOCK-OUT
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
                    ),
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
            ),
    }


# ============================================================
# DETERMINE RISK LEVEL
# ============================================================

def determine_risk(
    stockout_result,
    current_stock,
    forecast_days=7
):

    if stockout_result[
        "will_stockout"
    ]:

        days = stockout_result[
            "days_until_stockout"
        ]

        if days <= 2:

            return "CRITICAL"

        elif days <= 4:

            return "HIGH"

        elif days <= 7:

            return "MEDIUM"

        else:

            return "LOW"

    # --------------------------------------------------------
    # Stock doesn't run out during forecast
    # --------------------------------------------------------

    remaining_stock = (
        stockout_result[
            "remaining_stock"
        ]
    )

    if current_stock <= 0:

        return "CRITICAL"

    stock_ratio = (
        remaining_stock
        / current_stock
    )

    if stock_ratio < 0.15:

        return "HIGH"

    elif stock_ratio < 0.30:

        return "MEDIUM"

    return "LOW"


# ============================================================
# COMPLETE STOCK-OUT ANALYSIS
# ============================================================

def analyze_stockout(
    phc_id,
    medicine_name
):

    # --------------------------------------------------------
    # Get current inventory
    # --------------------------------------------------------

    inventory = get_current_inventory(
        phc_id,
        medicine_name
    )

    current_stock = float(
        inventory["current_stock"]
    )

    # --------------------------------------------------------
    # Generate AI forecast
    # --------------------------------------------------------

    predictions = forecast_demand(
        phc_id,
        medicine_name,
        days=7
    )

    # --------------------------------------------------------
    # Calculate stock-out
    # --------------------------------------------------------

    stockout_result = calculate_stockout(
        current_stock,
        predictions
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    risk = determine_risk(
        stockout_result,
        current_stock
    )

    # --------------------------------------------------------
    # Create final result
    # --------------------------------------------------------

    result = {

        "phc_id":
            phc_id,

        "medicine_name":
            medicine_name,

        "current_stock":
            current_stock,

        "forecast":
            predictions,

        "stockout":
            stockout_result,

        "risk":
            risk,

        "analyzed_at":
            datetime.utcnow(),
    }

    # --------------------------------------------------------
    # Save to Firestore
    # --------------------------------------------------------

    analysis_id = (
        f"{phc_id}_"
        f"{medicine_name.lower().replace(' ', '_')}"
    )

    db.collection(
        "stockout_predictions"
    ).document(
        analysis_id
    ).set(result)

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "     AROGYAFLOW AI STOCK-OUT TEST"
    )

    print(
        "======================================\n"
    )

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
        f"\n🚨 Risk: "
        f"{result['risk']}"
    )

    stockout = result[
        "stockout"
    ]

    if stockout[
        "will_stockout"
    ]:

        print(
            f"\n⚠️ Stock-out predicted!"
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
            "\n✅ No stock-out predicted "
            "within 7 days."
        )

        print(
            f"📦 Remaining after forecast: "
            f"{stockout['remaining_stock']}"
        )

    print(
        "\n🔥 Analysis saved to Firestore."
    )