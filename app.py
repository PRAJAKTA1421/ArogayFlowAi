from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import datetime, date


from ai.forecasting import forecast_demand
from ai.stockout_prediction import analyze_stockout
from ai.anomaly_detection import analyze_anomaly
from ai.redistribution import find_source_phcs, save_recommendation


app = Flask(__name__)
app.config["SECRET_KEY"] = "arogyaflow-demo-key"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def phc_name_from_login(value):
    """Create a readable PHC label for the demo login identifier."""

    identifier = (
        value.split("@", 1)[0]
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

    name = " ".join(
        part.capitalize()
        for part in identifier.split()
    ) or "My PHC"

    return (
        name
        if "phc" in name.lower()
        else f"{name} PHC"
    )


def make_json_safe(value):
    """
    Convert Firestore/Python values into JSON-safe values.

    This is important because some AI modules return
    datetime/Timestamp objects.
    """

    if isinstance(value, (datetime, date)):
        return value.isoformat()


    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    # Handle numpy numeric values without importing numpy here.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ============================================================
# TEMPLATE CONTEXT
# ============================================================

@app.context_processor
def logged_in_phc():
    return {
        "current_phc": session.get(
            "phc_name",
            ""
        ),
        "current_manager": session.get(
            "manager_name",
            "PHC Manager"
        ),
    }


# ============================================================
# BASIC WEBSITE ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        session["phc_name"] = phc_name_from_login(
            request.form.get("email", "")
        )

        session["manager_name"] = "PHC Manager"

        return redirect(
            url_for("dashboard")
        )

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        return redirect(
            url_for("dashboard")
        )

    return render_template("register.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/phc-network")
def phc_network():
    return render_template("phc_network.html")


@app.route("/medicine-inventory")
def medicine_inventory():
    return render_template(
        "medicine_inventory.html"
    )


@app.route("/beds-overview")
def beds_overview():
    return render_template(
        "beds_overview.html"
    )


@app.route("/medical-staff")
def medical_staff():
    return render_template(
        "medical_staff.html"
    )


@app.route("/ai-predictions")
def ai_predictions():
    return render_template(
        "ai_predictions.html"
    )


@app.route("/resource-transfer")
def resource_transfer():
    return render_template(
        "resource_transfer.html"
    )


@app.route("/settings")
def settings():
    return render_template(
        "settings.html"
    )


# ============================================================
# AI ANALYSIS API
# ============================================================

@app.route(
    "/api/ai/analyze",
    methods=["POST"]
)
def ai_analyze():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        phc_id = str(
            data.get("phc_id", "")
        ).strip()

        medicine = str(
            data.get("medicine", "")
        ).strip()

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not phc_id:

            return jsonify({
                "success": False,
                "error": "PHC ID is required."
            }), 400

        if not medicine:

            return jsonify({
                "success": False,
                "error": "Medicine name is required."
            }), 400

        print("\n")
        print("==========================================")
        print("        AROGYAFLOW AI API ANALYSIS")
        print("==========================================")
        print(f"🏥 PHC: {phc_id}")
        print(f"💊 Medicine: {medicine}")

        # ====================================================
        # 1. DEMAND FORECAST
        # ====================================================

        print("\n📈 Running demand forecasting...")

        forecast_result = forecast_demand(
            phc_id=phc_id,
            medicine_name=medicine,
            days=7
        )

        forecast_result = make_json_safe(
            forecast_result
        )

        total_7_day_demand = round(
            sum(
                float(
                    item.get(
                        "predicted_demand",
                        0
                    )
                )
                for item in forecast_result
            ),
            2
        )

        # ====================================================
        # 2. STOCKOUT PREDICTION
        # ====================================================

        print("\n🚨 Running stockout prediction...")

        stockout_result = analyze_stockout(
            phc_id=phc_id,
            medicine_name=medicine
        )

        stockout_result = make_json_safe(
            stockout_result
        )

        # Extract ML probability safely
        ml_prediction = stockout_result.get(
            "ml_prediction",
            {}
        )

        stockout_probability = float(
            ml_prediction.get(
                "stockout_probability",
                0
            )
        )

        stockout_probability_percent = float(
            ml_prediction.get(
                "stockout_probability_percent",
                stockout_probability * 100
            )
        )

        risk = stockout_result.get(
            "risk",
            "LOW"
        )

        # ====================================================
        # 3. ANOMALY DETECTION
        # ====================================================

        print("\n🔎 Running anomaly detection...")

        anomaly_result = analyze_anomaly(
            phc_id=phc_id,
            medicine_name=medicine,
            save_result=True
        )

        anomaly_result = make_json_safe(
            anomaly_result
        )

        # ====================================================
        # 4. REDISTRIBUTION OPTIMIZATION
        # ====================================================

        print("\n🔄 Running redistribution optimization...")

        redistribution_result = find_source_phcs(
            destination_phc_id=phc_id,
            medicine_name=medicine
        )

        redistribution_result = make_json_safe(
            redistribution_result
        )

        # ----------------------------------------------------
        # Save recommendation if a transfer is recommended
        # ----------------------------------------------------

        recommendation_id = None

        if redistribution_result.get(
            "action"
        ) in (
            "REDISTRIBUTION_RECOMMENDED",
            "PARTIAL_REDISTRIBUTION"
        ):

            try:

                recommendation_id = (
                    save_recommendation(
                        redistribution_result
                    )
                )

            except Exception as save_error:

                print(
                    "⚠️ Could not save "
                    f"redistribution recommendation: "
                    f"{save_error}"
                )

        # ====================================================
        # FINAL AI RESULT
        # ====================================================

        result = {

            "success": True,

            "phc_id": phc_id,

            "medicine": medicine,

            # -----------------------------------------------
            # DEMAND
            # -----------------------------------------------

            "forecast": {

                "predictions":
                    forecast_result,

                "total_7_day_demand":
                    total_7_day_demand
            },

            # -----------------------------------------------
            # STOCKOUT
            # -----------------------------------------------

            "stockout": {

                "risk": risk,

                "probability":
                    stockout_probability,

                "probability_percent":
                    stockout_probability_percent,

                "ml_prediction":
                    ml_prediction,

                "current_stock":
                    stockout_result.get(
                        "current_stock",
                        0
                    ),

                "minimum_stock":
                    stockout_result.get(
                        "minimum_stock",
                        0
                    ),

                "will_stockout":
                    stockout_result.get(
                        "stockout",
                        {}
                    ).get(
                        "will_stockout",
                        False
                    ),

                "days_until_stockout":
                    stockout_result.get(
                        "stockout",
                        {}
                    ).get(
                        "days_until_stockout"
                    ),

                "stockout_date":
                    stockout_result.get(
                        "stockout",
                        {}
                    ).get(
                        "stockout_date"
                    ),

                "remaining_stock":
                    stockout_result.get(
                        "stockout",
                        {}
                    ).get(
                        "remaining_stock",
                        0
                    ),

                "cumulative_demand":
                    stockout_result.get(
                        "stockout",
                        {}
                    ).get(
                        "cumulative_demand",
                        total_7_day_demand
                    ),

                "replenishment_required":
                    stockout_result.get(
                        "replenishment_required",
                        False
                    ),

                "replenishment_priority":
                    stockout_result.get(
                        "replenishment_priority",
                        "NONE"
                    )
            },

            # -----------------------------------------------
            # ANOMALY
            # -----------------------------------------------

            "anomaly": {

                "is_anomaly":
                    anomaly_result.get(
                        "is_anomaly",
                        False
                    ),

                "status":
                    anomaly_result.get(
                        "anomaly_status",
                        "NORMAL"
                    ),

                "isolation_score":
                    anomaly_result.get(
                        "isolation_score",
                        0
                    ),

                "current_demand":
                    anomaly_result.get(
                        "current_demand",
                        0
                    ),

                "rolling_7_demand":
                    anomaly_result.get(
                        "rolling_7_demand",
                        0
                    ),

                "demand_change":
                    anomaly_result.get(
                        "demand_change",
                        0
                    ),

                "days_of_stock":
                    anomaly_result.get(
                        "days_of_stock",
                        0
                    )
            },

            # -----------------------------------------------
            # REDISTRIBUTION
            # -----------------------------------------------

            "redistribution": {

                "action":
                    redistribution_result.get(
                        "action",
                        "NO_ACTION_REQUIRED"
                    ),

                "shortage":
                    redistribution_result.get(
                        "shortage",
                        0
                    ),

                "current_stock":
                    redistribution_result.get(
                        "current_stock",
                        0
                    ),

                "minimum_stock":
                    redistribution_result.get(
                        "minimum_stock",
                        0
                    ),

                "forecast_remaining_stock":
                    redistribution_result.get(
                        "forecast_remaining_stock",
                        0
                    ),

                "total_transfer":
                    redistribution_result.get(
                        "total_transfer",
                        0
                    ),

                "remaining_shortage":
                    redistribution_result.get(
                        "remaining_shortage",
                        0
                    ),

                "transfer_plan":
                    redistribution_result.get(
                        "transfer_plan",
                        []
                    ),

                "optimization_method":
                    redistribution_result.get(
                        "optimization_method",
                        "Distance + Safe Surplus Weighted Optimization"
                    ),

                "recommendation_id":
                    recommendation_id
            }
        }

        # ====================================================
        # CONVERT EVERYTHING TO JSON SAFE VALUES
        # ====================================================

        result = make_json_safe(
            result
        )

        print("\n==========================================")
        print("        AI ANALYSIS COMPLETED")
        print("==========================================")
        print(
            f"📈 7-Day Demand: "
            f"{total_7_day_demand}"
        )
        print(
            f"🚨 Stockout Risk: "
            f"{risk}"
        )
        print(
            f"🔎 Anomaly: "
            f"{anomaly_result.get('anomaly_status')}"
        )
        print(
            f"🔄 Redistribution: "
            f"{redistribution_result.get('action')}"
        )
        print("==========================================\n")

        return jsonify(
            result
        )

    except Exception as error:

        print("\n==========================================")
        print("❌ AROGYAFLOW AI API ERROR")
        print("==========================================")
        print(
            f"{type(error).__name__}: {error}"
        )
        print("==========================================\n")

        return jsonify({

            "success": False,

            "error":
                str(error),

            "error_type":
                type(error).__name__
        }), 500


# ============================================================
# SIMPLE AI HEALTH CHECK
# ============================================================

@app.route(
    "/api/ai/health",
    methods=["GET"]
)
def ai_health():

    return jsonify({

        "success": True,

        "service":
            "ArogyaFlow AI",

        "status":
            "online",

        "modules": {

            "demand_forecasting":
                "Random Forest",

            "stockout_prediction":
                "Random Forest",

            "anomaly_detection":
                "Isolation Forest",

            "redistribution":
                "Distance + Safe Surplus Weighted Optimization"
        }
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )