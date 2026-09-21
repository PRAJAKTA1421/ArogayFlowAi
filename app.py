from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import datetime, date
import time

from firebase_config import db
from ai.forecasting import forecast_demand
from ai.stockout_prediction import analyze_stockout
from ai.anomaly_detection import analyze_anomaly
from ai.redistribution import find_source_phcs, save_recommendation

def safe_float(value, default=0.0):
    """Safely convert Firestore/numeric values to float."""
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if value == "":
                return default

        return float(value)

    except (TypeError, ValueError):
        return default

app = Flask(__name__)
app.config["SECRET_KEY"] = "arogyaflow-demo-key"

# Dashboard cache prevents repeated Firestore reads while the page is open.
DASHBOARD_CACHE_TTL = 300
AI_ANALYSIS_CACHE_TTL = 600
_dashboard_cache = {"timestamp": 0.0, "payload": None}
_ai_analysis_cache = {}


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



# ============================================================
# LIVE DASHBOARD OVERVIEW API
# ============================================================

@app.route("/api/dashboard/overview", methods=["GET"])
def dashboard_overview():
    """Return dashboard data without scanning demand_history."""
    try:
        now = time.time()
        if (_dashboard_cache["payload"] is not None and
                now - _dashboard_cache["timestamp"] < DASHBOARD_CACHE_TTL):
            return jsonify(_dashboard_cache["payload"])

        # Cheap reads only: PHCs + medicines + one summary + alerts + saved transfers.
        phc_docs = list(db.collection("phcs").stream())
        medicine_docs = list(db.collection("medicines").stream())
        summary = (
            db.collection("dashboard_summary")
            .document("network")
            .get()
            .to_dict()
            or {}
        )
        alert_docs = list(db.collection("alerts").stream())

        phcs = []
        phc_by_id = {}
        for doc in phc_docs:
            item = doc.to_dict() or {}
            item["id"] = doc.id
            phcs.append(item)
            phc_by_id[doc.id] = item

        medicines = []
        for doc in medicine_docs:
            item = doc.to_dict() or {}
            item["id"] = doc.id
            medicines.append(item)

        def num(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def date_key(value):
            if hasattr(value, "date") and not isinstance(value, str):
                try:
                    return value.date().isoformat()
                except Exception:
                    pass
            return str(value or "")[:10]

        total_phcs = len(phcs)
        total_beds = sum(num(p.get("total_beds")) for p in phcs)
        occupied_beds = sum(num(p.get("occupied_beds")) for p in phcs)
        available_beds = sum(num(p.get("available_beds")) for p in phcs)
        bed_occupancy = occupied_beds / total_beds * 100 if total_beds else 0

        total_staff = sum(num(p.get("total_staff")) for p in phcs)
        staff_on_duty = sum(num(p.get("present_staff")) for p in phcs)
        staff_availability = staff_on_duty / total_staff * 100 if total_staff else 0

        healthy_medicine_records = sum(
            1 for m in medicines
            if str(m.get("status", "")).lower() == "healthy"
        )
        medicine_availability = (
            healthy_medicine_records / len(medicines) * 100
            if medicines else 0
        )

        priority_rows = summary.get("priority_phcs", [])
        priority_count = int(summary.get("priority_count", len(priority_rows)))
        critical_phcs = int(summary.get("critical_phcs", 0))
        high_phcs = int(summary.get("high_phcs", 0))
        normal_phcs = max(total_phcs - priority_count, 0)

        top_shortage = summary.get("top_shortage")
        demand_trend = summary.get("demand_trend", [])
        patients_today = num(summary.get("patients_today", 0))
        latest_demand_date = summary.get("latest_demand_date", "")

        alerts = []
        for doc in alert_docs:
            item = doc.to_dict() or {}
            item["id"] = doc.id
            alerts.append(item)
        alerts.sort(key=lambda x: date_key(x.get("created_at")), reverse=True)
        active_alerts = sum(
            1 for a in alerts
            if str(a.get("status", "active")).lower() == "active"
        )

        top_occupancy = None
        if phcs:
            top = max(
                phcs,
                key=lambda p: num(p.get("occupied_beds")) / num(p.get("total_beds"), 1)
            )
            occ = num(top.get("occupied_beds")) / num(top.get("total_beds"), 1) * 100
            top_occupancy = {
                "name": top.get("name", top.get("id", "PHC")),
                "occupancy": occ,
            }

        # IMPORTANT: show only saved results from the actual redistribution optimizer.
        transfer_signals = []
        try:
            recommendation_docs = list(
                db.collection("transfer_recommendations")
                .order_by("created_at", direction="DESCENDING")
                .limit(5)
                .stream()
            )
            for doc in recommendation_docs:
                result = doc.to_dict() or {}
                for transfer in result.get("transfer_plan") or []:
                    transfer_signals.append({
                        "medicine": result.get("medicine", transfer.get("medicine", "Medicine")),
                        "source": transfer.get("source_name", transfer.get("source_phc", "Source PHC")),
                        "destination": transfer.get("destination_name", result.get("destination_name", transfer.get("destination_phc", "Destination PHC"))),
                        "quantity": round(num(transfer.get("recommended_quantity")), 1),
                        "source_surplus": round(num(transfer.get("source_safe_surplus")), 1),
                        "distance_km": round(num(transfer.get("distance_km")), 1),
                        "optimization_score": round(num(transfer.get("optimization_score")), 2),
                        "screening_note": "Saved redistribution optimizer result",
                    })
                    if len(transfer_signals) >= 5:
                        break
                if len(transfer_signals) >= 5:
                    break
        except Exception as transfer_error:
            print(f"Dashboard transfer signal query: {type(transfer_error).__name__}: {transfer_error}")

        health_score = 100
        health_score -= min(35, critical_phcs * 5)
        health_score -= min(25, high_phcs * 2)
        health_score -= min(20, max(0, 90 - medicine_availability) * 0.4)
        health_score -= min(10, max(0, 85 - staff_availability) * 0.25)
        health_score = max(0, min(100, round(health_score)))
        health_label = "Good" if health_score >= 75 else "Watch" if health_score >= 50 else "Needs Attention"

        payload = make_json_safe({
            "success": True,
            "generated_at_display": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "summary_source": "dashboard_summary/network",
            "kpis": {
                "total_phcs": total_phcs,
                "total_beds": total_beds,
                "available_beds": available_beds,
                "bed_occupancy": bed_occupancy,
                "patients_today": patients_today,
                "patients_today_display": (
                    f"{patients_today / 100000:.2f} Lakh" if patients_today >= 100000
                    else f"{patients_today / 1000:.1f}K" if patients_today >= 1000
                    else f"{patients_today:.0f}"
                ),
                "medicine_availability": medicine_availability,
                "healthy_medicine_records": healthy_medicine_records,
                "staff_on_duty": staff_on_duty,
                "staff_availability": staff_availability,
                "normal_phcs": normal_phcs,
                "at_risk_phcs": priority_count,
                "critical_phcs": critical_phcs,
                "active_alerts": active_alerts,
                "latest_demand_date": latest_demand_date,
                "health_score": health_score,
                "health_score_label": health_label,
                "top_occupancy": top_occupancy,
            },
            "priority_count": priority_count,
            "staff_availability": staff_availability,
            "top_shortage": top_shortage,
            "top_occupancy": top_occupancy,
            "demand_trend": demand_trend,
            "priority_phcs": priority_rows[:8],
            "transfer_signals": transfer_signals[:5],
            "alerts": alerts[:5],
        })

        _dashboard_cache["timestamp"] = time.time()
        _dashboard_cache["payload"] = payload
        return jsonify(payload)

    except Exception as error:
        print(f"Dashboard overview error: {type(error).__name__}: {error}")
        return jsonify({"success": False, "error": str(error), "error_type": type(error).__name__}), 500



# ============================================================
# PHC NETWORK API
# ============================================================

PHC_NETWORK_CACHE_TTL = 300
_phc_network_cache = {"timestamp": 0.0, "payload": None}


@app.route("/api/phc-network", methods=["GET"])
def phc_network_api():
    """Return PHC Network data directly from Firestore."""
    try:
        now = time.time()
        if (
            _phc_network_cache["payload"] is not None
            and now - _phc_network_cache["timestamp"] < PHC_NETWORK_CACHE_TTL
        ):
            return jsonify(_phc_network_cache["payload"])

        # One read of PHCs and one read of current medicine inventory.
        phc_docs = list(db.collection("phcs").stream())
        medicine_docs = list(db.collection("medicines").stream())

        phcs = []
        phc_by_id = {}
        for doc in phc_docs:
            item = doc.to_dict() or {}
            item["id"] = doc.id
            phcs.append(item)
            phc_by_id[doc.id] = item

        medicines_by_phc = {}
        for doc in medicine_docs:
            item = doc.to_dict() or {}
            phc_id = str(item.get("phc_id", "")).strip()
            if phc_id:
                medicines_by_phc.setdefault(phc_id, []).append(item)

        def num(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        # --------------------------------------------------------
        # PHC-level risk calculation
        # --------------------------------------------------------
        # This is a network-page operational risk classification.
        # It is NOT presented as a trained ML prediction.
        #
        # Critical: high bed occupancy OR severe medicine pressure
        #           OR severe staff shortage.
        # At risk: moderate bed/staff/inventory pressure.
        # Normal: no major pressure signal.
        # --------------------------------------------------------
        network_rows = []

        for phc in phcs:
            phc_id = phc["id"]
            name = phc.get("name", phc_id)
            total_beds = num(phc.get("total_beds"))
            occupied_beds = num(phc.get("occupied_beds"))
            total_staff = num(phc.get("total_staff"))
            present_staff = num(phc.get("present_staff"))

            occupancy = (
                occupied_beds / total_beds * 100
                if total_beds > 0 else 0
            )
            staff_availability = (
                present_staff / total_staff * 100
                if total_staff > 0 else 100
            )

            inventory = medicines_by_phc.get(phc_id, [])
            low_inventory = 0
            critical_inventory = 0
            shortage_units = 0.0

            for medicine in inventory:
                stock = num(
                    medicine.get(
                        "current_stock",
                        medicine.get("stock", medicine.get("quantity", 0))
                    )
                )
                minimum = num(
                    medicine.get(
                        "minimum_stock",
                        medicine.get("min_stock", 0)
                    )
                )
                status = str(medicine.get("status", "")).lower()

                if minimum > 0 and stock < minimum:
                    shortage_units += minimum - stock
                    low_inventory += 1
                    if stock <= minimum * 0.50:
                        critical_inventory += 1
                elif status in {"low", "critical", "out_of_stock", "stockout"}:
                    low_inventory += 1
                    if status in {"critical", "out_of_stock", "stockout"}:
                        critical_inventory += 1

            reasons = []
            risk_score = 0

            if occupancy >= 85:
                risk_score += 3
                reasons.append("High Bed Occupancy")
            elif occupancy >= 70:
                risk_score += 2
                reasons.append("Elevated Bed Occupancy")

            if staff_availability < 70:
                risk_score += 3
                reasons.append("Staff Shortage")
            elif staff_availability < 85:
                risk_score += 2
                reasons.append("Reduced Staff Availability")

            if critical_inventory > 0:
                risk_score += 3
                reasons.append("Critical Medicine Shortage")
            elif low_inventory > 0:
                risk_score += 2
                reasons.append("Medicine Shortage")

            if risk_score >= 4:
                risk = "critical"
            elif risk_score >= 2:
                risk = "warning"
            else:
                risk = "normal"

            # Preserve a stored critical/warning status when it is more severe
            # than the calculated operational score.
            stored_status = str(phc.get("status", "")).lower()
            if stored_status == "critical":
                risk = "critical"
            elif stored_status in {"warning", "at_risk", "at risk"} and risk == "normal":
                risk = "warning"

            if not reasons:
                reasons.append("Normal Operations")

            network_rows.append({
                "id": phc_id,
                "name": name,
                "district": phc.get("district", ""),
                "state": phc.get("state", "India"),
                "latitude": num(phc.get("latitude")),
                "longitude": num(phc.get("longitude")),
                "status": risk,
                "risk_score": risk_score,
                "risk_percent": min(99, max(1, risk_score * 20)),
                "primary_issue": reasons[0],
                "occupancy": round(occupancy, 1),
                "staff_availability": round(staff_availability, 1),
                "low_inventory": low_inventory,
                "critical_inventory": critical_inventory,
                "shortage_units": round(shortage_units, 1),
                "patients_today": num(phc.get("patients_today")),
            })

        total_phcs = len(network_rows)
        normal = sum(1 for row in network_rows if row["status"] == "normal")
        at_risk = sum(1 for row in network_rows if row["status"] == "warning")
        critical = sum(1 for row in network_rows if row["status"] == "critical")

        states = {}
        districts = set()
        for row in network_rows:
            state = row["state"] or "Unknown"
            states[state] = states.get(state, 0) + 1
            district = row["district"]
            if district:
                districts.add(str(district))

        state_rows = [
            {"state": state, "count": count}
            for state, count in states.items()
        ]
        state_rows.sort(key=lambda item: item["count"], reverse=True)

        # Highest-risk PHCs first. This is a network operational list, not ML output.
        critical_rows = [
            row for row in network_rows
            if row["status"] == "critical"
        ]
        critical_rows.sort(
            key=lambda row: (row["risk_score"], row["shortage_units"], row["occupancy"]),
            reverse=True,
        )

        # Include warning PHCs if fewer than five critical PHCs exist so the panel
        # remains useful with the current 50-PHC demo database.
        if len(critical_rows) < 5:
            remaining = [
                row for row in network_rows
                if row["status"] == "warning"
            ]
            remaining.sort(
                key=lambda row: (row["risk_score"], row["shortage_units"], row["occupancy"]),
                reverse=True,
            )
            critical_rows.extend(remaining[:5 - len(critical_rows)])

        critical_phcs = [
            {
                "id": row["id"],
                "name": row["name"],
                "state": row["state"],
                "issue": row["primary_issue"],
                "risk": row["risk_percent"],
                "status": row["status"],
                "occupancy": row["occupancy"],
                "staff_availability": row["staff_availability"],
                "shortage_units": row["shortage_units"],
            }
            for row in critical_rows[:5]
        ]

        # --------------------------------------------------------
        # PATIENTS TODAY
        # --------------------------------------------------------
        # Prefer the patients_today value stored directly on the
        # PHC documents.
        #
        # If those PHC records do not contain usable patient
        # counts, use the existing quota-safe network summary.
        #
        # We intentionally do NOT scan demand_history here.
        # --------------------------------------------------------

        total_patients_today = sum(
            row["patients_today"]
            for row in network_rows
            if row["patients_today"] > 0
        )

        # If PHC-level patient counts are unavailable, use the
        # existing Firestore dashboard summary.
        if total_patients_today <= 0:

            try:

                network_summary_doc = (
                    db.collection("dashboard_summary")
                    .document("network")
                    .get()
                )

                network_summary_data = (
                    network_summary_doc.to_dict()
                    or {}
                )

                total_patients_today = num(
                    network_summary_data.get(
                        "patients_today",
                        0
                    )
                )

            except Exception as summary_error:

                print(
                    "PHC Network patient summary fallback: "
                    f"{type(summary_error).__name__}: "
                    f"{summary_error}"
                )

                total_patients_today = 0

        total_beds = sum(
            num(phc.get("total_beds"))
            for phc in phcs
        )

        occupied_beds = sum(
            num(phc.get("occupied_beds"))
            for phc in phcs
        )

        average_occupancy = (
            occupied_beds / total_beds * 100
            if total_beds > 0 else 0
        )

        # India-map positioning uses the PHC latitude/longitude as an approximate
        # normalized position. It is intentionally simple so no new map library is required.
        map_points = []
        for row in network_rows:
            lat = row["latitude"]
            lon = row["longitude"]
            if lat == 0 and lon == 0:
                continue
            left = max(2, min(98, (lon - 68) / 30 * 100))
            top = max(2, min(98, (37 - lat) / 29 * 100))
            map_points.append({
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "latitude": lat,
                "longitude": lon,
                "left": round(left, 2),
                "top": round(top, 2),
            })

        payload = make_json_safe({
            "success": True,
            "source": "Firestore phcs + medicines",
            "generated_at": datetime.now(),
            "summary": {
                "total_phcs": total_phcs,
                "normal": normal,
                "at_risk": at_risk,
                "critical": critical,
                "total_districts": len(districts),
                "total_states": len(states),
                "normal_percent": round(normal / total_phcs * 100, 1) if total_phcs else 0,
                "at_risk_percent": round(at_risk / total_phcs * 100, 1) if total_phcs else 0,
                "critical_percent": round(critical / total_phcs * 100, 1) if total_phcs else 0,
            },
            "states": state_rows[:6],
            "critical_phcs": critical_phcs,
            "network_summary": {
                "patients_today": round(total_patients_today),
                "average_occupancy": round(average_occupancy, 1),
            },
            "map_points": map_points,
        })

        _phc_network_cache["timestamp"] = time.time()
        _phc_network_cache["payload"] = payload
        return jsonify(payload)

    except Exception as error:
        print(f"PHC Network API error: {type(error).__name__}: {error}")
        return jsonify({
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }), 500


@app.route("/phc-network")
def phc_network():
    return render_template("phc_network.html")



# ============================================================
# MEDICINE INVENTORY API
# ============================================================

MEDICINE_INVENTORY_CACHE_TTL = 300
_medicine_inventory_cache = {"timestamp": 0.0, "payload": None}


@app.route("/api/medicine-inventory", methods=["GET"])
def medicine_inventory_api():
    """Return quota-safe live medicine inventory statistics from Firestore."""
    try:
        now = time.time()
        if (
            _medicine_inventory_cache["payload"] is not None
            and now - _medicine_inventory_cache["timestamp"] < MEDICINE_INVENTORY_CACHE_TTL
        ):
            return jsonify(_medicine_inventory_cache["payload"])

        # One read of the PHC collection and one read of the medicine collection.
        # No demand_history scan is required for this page.
        phc_docs = list(db.collection("phcs").stream())
        medicine_docs = list(db.collection("medicines").stream())

        phcs_by_id = {}
        for doc in phc_docs:
            phc = doc.to_dict() or {}
            phcs_by_id[doc.id] = phc

        medicines = []
        for doc in medicine_docs:
            item = doc.to_dict() or {}
            item["id"] = doc.id
            medicines.append(item)

        def num(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def normalize_name(value):
            name = str(value or "").strip()
            replacements = {
                "Paracetamol 500mg": "Paracetamol",
                "Amoxicillin 500mg": "Amoxicillin",
                "Iron Folic Acid Tablet": "Iron Folic Acid",
                "Amlodipine 5mg": "Amlodipine",
                "Cetirizine 10mg": "Cetirizine",
                "Metformin 500mg": "Metformin",
                "Omeprazole 20mg": "Omeprazole",
                "ORS Sachet": "ORS",
            }
            return replacements.get(name, name)

        total = len(medicines)
        sufficient = 0
        low_stock = 0
        out_of_stock = 0

        medicine_stats = {}
        state_stats = {}
        latest_updated = None

        for medicine in medicines:
            stock = num(
                medicine.get(
                    "current_stock",
                    medicine.get("stock", medicine.get("quantity", 0)),
                )
            )
            minimum = num(
                medicine.get(
                    "minimum_stock",
                    medicine.get("min_stock", 0),
                )
            )

            if stock <= 0:
                inventory_status = "out_of_stock"
                out_of_stock += 1
            elif minimum > 0 and stock < minimum:
                inventory_status = "low"
                low_stock += 1
            else:
                inventory_status = "sufficient"
                sufficient += 1

            medicine_name = normalize_name(medicine.get("medicine_name", "Medicine"))
            stats = medicine_stats.setdefault(
                medicine_name,
                {"low_stock": 0, "out_of_stock": 0, "total": 0},
            )
            stats["total"] += 1
            if inventory_status == "low":
                stats["low_stock"] += 1
            elif inventory_status == "out_of_stock":
                stats["out_of_stock"] += 1

            phc_id = str(medicine.get("phc_id", "")).strip()
            phc = phcs_by_id.get(phc_id, {})
            state = str(phc.get("state", "Unknown")).strip() or "Unknown"

            state_entry = state_stats.setdefault(
                state,
                {"total": 0, "sufficient": 0, "low_stock": 0, "out_of_stock": 0},
            )
            state_entry["total"] += 1
            state_entry[inventory_status] = state_entry.get(inventory_status, 0) + 1

            updated_at = medicine.get("updated_at")
            if updated_at is not None:
                try:
                    if latest_updated is None or updated_at > latest_updated:
                        latest_updated = updated_at
                except TypeError:
                    pass

        availability_percent = (sufficient / total * 100) if total else 0

        top_medicines = []
        for name, stats in medicine_stats.items():
            affected = stats["low_stock"] + stats["out_of_stock"]
            if affected <= 0:
                continue

            if stats["out_of_stock"] > 0 and stats["out_of_stock"] >= stats["low_stock"]:
                status = "Critical"
            elif affected >= max(3, round(stats["total"] * 0.30)):
                status = "High"
            else:
                status = "Low"

            top_medicines.append({
                "medicine_name": name,
                "low_stock_phcs": stats["low_stock"],
                "out_of_stock_phcs": stats["out_of_stock"],
                "affected_phcs": affected,
                "status": status,
            })

        top_medicines.sort(
            key=lambda row: (
                row["out_of_stock_phcs"],
                row["low_stock_phcs"],
                row["affected_phcs"],
            ),
            reverse=True,
        )

        state_rows = []
        for state, stats in state_stats.items():
            state_total = stats["total"]
            state_availability = (
                stats["sufficient"] / state_total * 100
                if state_total else 0
            )
            state_rows.append({
                "state": state,
                "availability_percent": round(state_availability, 1),
                "total_records": state_total,
                "sufficient": stats["sufficient"],
                "low_stock": stats["low_stock"],
                "out_of_stock": stats["out_of_stock"],
            })

        state_rows.sort(
            key=lambda row: row["availability_percent"],
            reverse=True,
        )

        if latest_updated is None:
            updated_display = datetime.now().strftime("%d %b %Y, %I:%M %p")
        else:
            try:
                updated_display = latest_updated.strftime("%d %b %Y, %I:%M %p")
            except AttributeError:
                updated_display = str(latest_updated)

        payload = make_json_safe({
            "success": True,
            "source": "Firestore medicines + phcs",
            "generated_at": datetime.now(),
            "updated_display": updated_display,
            "summary": {
                "total_medicines": total,
                "sufficient_stock": sufficient,
                "low_stock": low_stock,
                "out_of_stock": out_of_stock,
                "availability_percent": round(availability_percent, 1),
                "sufficient_percent": round(sufficient / total * 100, 1) if total else 0,
                "low_stock_percent": round(low_stock / total * 100, 1) if total else 0,
                "out_of_stock_percent": round(out_of_stock / total * 100, 1) if total else 0,
            },
            "top_medicines": top_medicines[:5],
            "state_availability": state_rows[:8],
        })

        _medicine_inventory_cache["timestamp"] = time.time()
        _medicine_inventory_cache["payload"] = payload
        return jsonify(payload)

    except Exception as error:
        print(
            f"Medicine inventory API error: "
            f"{type(error).__name__}: {error}"
        )
        return jsonify({
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
        }), 500


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

# ============================================================
# BEDS OVERVIEW API
# ============================================================

BEDS_OVERVIEW_CACHE_TTL = 300

_beds_overview_cache = {
    "timestamp": 0.0,
    "payload": None,
}


@app.route("/api/beds-overview", methods=["GET"])
def beds_overview_api():

    try:
        now = time.time()

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if (
            _beds_overview_cache["payload"] is not None
            and
            now - _beds_overview_cache["timestamp"]
            < BEDS_OVERVIEW_CACHE_TTL
        ):
            return jsonify(
                _beds_overview_cache["payload"]
            )

        # ----------------------------------------------------
        # READ PHC DATA
        # ----------------------------------------------------

        phc_docs = list(
            db.collection("phcs").stream()
        )

        phcs = []

        for doc in phc_docs:

            data = doc.to_dict() or {}

            def safe_number(value, default=0):
                try:
                    if value is None:
                        return float(default)
                    return float(value)
                except (TypeError, ValueError):
                    return float(default)

            phc_id = str(
                data.get("phc_id", doc.id)
            ).strip()

            name = str(
                data.get(
                    "name",
                    data.get(
                        "phc_name",
                        phc_id
                    )
                )
            ).strip()

            state = str(
                data.get(
                    "state",
                    "Unknown"
                )
            ).strip()

            total_beds = max(
                0,
                safe_number(
                    data.get("total_beds")
                )
            )

            occupied_beds = max(
                0,
                safe_number(
                    data.get("occupied_beds")
                )
            )

            # Never allow occupied beds to exceed
            # the total beds for display calculations.
            if total_beds > 0:
                occupied_beds = min(
                    occupied_beds,
                    total_beds
                )

            available_beds = max(
                0,
                safe_number(
                    data.get("available_beds")
                )
            )

            # Recalculate available beds from the
            # actual total/occupied values.
            if total_beds > 0:
                available_beds = max(
                    0,
                    total_beds - occupied_beds
                )

            occupancy = (
                (occupied_beds / total_beds) * 100
                if total_beds > 0
                else 0
            )

            phcs.append({
                "id": phc_id,
                "name": name,
                "state": state,
                "total_beds": int(
                    round(total_beds)
                ),
                "occupied_beds": int(
                    round(occupied_beds)
                ),
                "available_beds": int(
                    round(available_beds)
                ),
                "occupancy": round(
                    occupancy,
                    1
                ),
            })

        # ----------------------------------------------------
        # NETWORK TOTALS
        # ----------------------------------------------------

        total_beds = sum(
            item["total_beds"]
            for item in phcs
        )

        occupied_beds = sum(
            item["occupied_beds"]
            for item in phcs
        )

        available_beds = sum(
            item["available_beds"]
            for item in phcs
        )

        average_occupancy = (
            (occupied_beds / total_beds) * 100
            if total_beds > 0
            else 0
        )

        # ----------------------------------------------------
        # STATE-LEVEL OCCUPANCY
        # ----------------------------------------------------

        state_data = {}

        for item in phcs:

            state = item["state"]

            if state not in state_data:
                state_data[state] = {
                    "state": state,
                    "total_beds": 0,
                    "occupied_beds": 0,
                    "available_beds": 0,
                }

            state_data[state]["total_beds"] += (
                item["total_beds"]
            )

            state_data[state]["occupied_beds"] += (
                item["occupied_beds"]
            )

            state_data[state]["available_beds"] += (
                item["available_beds"]
            )

        state_occupancy = []

        for state, item in state_data.items():

            state_total = item["total_beds"]
            state_occupied = item["occupied_beds"]

            occupancy = (
                (state_occupied / state_total) * 100
                if state_total > 0
                else 0
            )

            state_occupancy.append({
                "state": state,
                "total_beds": state_total,
                "occupied_beds": state_occupied,
                "available_beds":
                    item["available_beds"],
                "occupancy": round(
                    occupancy,
                    1
                ),
            })

        state_occupancy.sort(
            key=lambda x: x["occupancy"],
            reverse=True
        )

        # ----------------------------------------------------
        # HIGH OCCUPANCY PHCs
        # ----------------------------------------------------

        high_occupancy = sorted(
            [
                item
                for item in phcs
                if item["occupancy"] > 90
            ],
            key=lambda x: x["occupancy"],
            reverse=True
        )[:10]

        # ----------------------------------------------------
        # LOW OCCUPANCY PHCs
        # ----------------------------------------------------

        low_occupancy = sorted(
            [
                item
                for item in phcs
                if item["occupancy"] < 30
            ],
            key=lambda x: x["occupancy"]
        )[:10]

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = {
            "total_phcs":
                len(phcs),

            "total_beds":
                total_beds,

            "occupied_beds":
                occupied_beds,

            "available_beds":
                available_beds,

            "occupancy_percent":
                round(
                    average_occupancy,
                    1
                ),

            "occupied_percent":
                round(
                    (
                        occupied_beds /
                        total_beds *
                        100
                    )
                    if total_beds > 0
                    else 0,
                    1
                ),

            "available_percent":
                round(
                    (
                        available_beds /
                        total_beds *
                        100
                    )
                    if total_beds > 0
                    else 0,
                    1
                ),

            "high_occupancy_count":
                len([
                    p for p in phcs
                    if p["occupancy"] > 90
                ]),

            "low_occupancy_count":
                len([
                    p for p in phcs
                    if p["occupancy"] < 30
                ]),
        }

        # ----------------------------------------------------
        # FINAL PAYLOAD
        # ----------------------------------------------------

        payload = make_json_safe({

            "success":
                True,

            "source":
                "Firestore phcs",

            "generated_at":
                datetime.now().isoformat(),

            "updated_display":
                datetime.now().strftime(
                    "%d %b %Y, %I:%M %p"
                ),

            "summary":
                summary,

            "state_occupancy":
                state_occupancy,

            "high_occupancy_phcs":
                high_occupancy,

            "low_occupancy_phcs":
                low_occupancy,

        })

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        _beds_overview_cache[
            "timestamp"
        ] = time.time()

        _beds_overview_cache[
            "payload"
        ] = payload

        return jsonify(payload)

    except Exception as error:

        print(
            "Beds Overview API error: "
            f"{type(error).__name__}: {error}"
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error),

            "error_type":
                type(error).__name__,

        }), 500

@app.route("/medical-staff")
def medical_staff():
    return render_template(
        "medical_staff.html"
    )

# ============================================================
# MEDICAL STAFF API
# ============================================================

MEDICAL_STAFF_CACHE_TTL = 300
_medical_staff_cache = {
    "timestamp": 0.0,
    "payload": None
}


@app.route("/api/medical-staff")
def api_medical_staff():
    global _medical_staff_cache

    now = time.time()

    # Return cached response for 5 minutes
    if (
        _medical_staff_cache["payload"] is not None
        and now - _medical_staff_cache["timestamp"] < MEDICAL_STAFF_CACHE_TTL
    ):
        return jsonify(_medical_staff_cache["payload"])

    try:
        phc_docs = list(db.collection("phcs").stream())

        if not phc_docs:
            return jsonify({
                "success": True,
                "source": "Firestore phcs.staff",
                "summary": {
                    "total_staff": 0,
                    "present_staff": 0,
                    "required_staff": 0,
                    "staff_gap": 0,
                    "availability_percent": 0,
                    "doctors": 0,
                    "nurses": 0,
                    "paramedics": 0,
                    "on_leave": 0,
                    "unavailable": 0
                },
                "state_staff": [],
                "availability": {
                    "available": 0,
                    "on_leave": 0,
                    "unavailable": 0
                },
                "shortage_alerts": [],
                "generated_at": datetime.utcnow().isoformat()
            })

        total_staff = 0
        present_staff = 0
        required_staff = 0
        total_gap = 0

        total_doctors = 0
        total_nurses = 0
        total_paramedics = 0

        total_on_leave = 0
        total_unavailable = 0

        state_data = {}
        shortage_alerts = []

        for doc in phc_docs:
            phc = doc.to_dict() or {}

            phc_id = phc.get("id") or doc.id
            phc_name = (
                phc.get("name")
                or phc.get("phc_name")
                or f"{phc_id} PHC"
            )

            state = (
                phc.get("state")
                or phc.get("State")
                or "Unknown"
            )

            staff = phc.get("staff") or {}

            doctors = staff.get("doctors") or {}
            nurses = staff.get("nurses") or {}
            paramedics = staff.get("paramedics") or {}
            summary = staff.get("summary") or {}

            # ------------------------------------------------
            # Role-level values
            # ------------------------------------------------

            doctor_total = safe_float(doctors.get("total", 0))
            doctor_present = safe_float(doctors.get("present", 0))
            doctor_required = safe_float(doctors.get("required", 0))
            doctor_leave = safe_float(doctors.get("on_leave", 0))
            doctor_unavailable = safe_float(doctors.get("unavailable", 0))
            doctor_gap = safe_float(
                doctors.get(
                    "gap",
                    max(doctor_required - doctor_present, 0)
                )
            )

            nurse_total = safe_float(nurses.get("total", 0))
            nurse_present = safe_float(nurses.get("present", 0))
            nurse_required = safe_float(nurses.get("required", 0))
            nurse_leave = safe_float(nurses.get("on_leave", 0))
            nurse_unavailable = safe_float(nurses.get("unavailable", 0))
            nurse_gap = safe_float(
                nurses.get(
                    "gap",
                    max(nurse_required - nurse_present, 0)
                )
            )

            paramedic_total = safe_float(paramedics.get("total", 0))
            paramedic_present = safe_float(paramedics.get("present", 0))
            paramedic_required = safe_float(paramedics.get("required", 0))
            paramedic_leave = safe_float(paramedics.get("on_leave", 0))
            paramedic_unavailable = safe_float(paramedics.get("unavailable", 0))
            paramedic_gap = safe_float(
                paramedics.get(
                    "gap",
                    max(paramedic_required - paramedic_present, 0)
                )
            )

            # ------------------------------------------------
            # PHC totals
            # ------------------------------------------------

            phc_total = (
                doctor_total
                + nurse_total
                + paramedic_total
            )

            phc_present = (
                doctor_present
                + nurse_present
                + paramedic_present
            )

            phc_required = (
                doctor_required
                + nurse_required
                + paramedic_required
            )

            phc_gap = (
                doctor_gap
                + nurse_gap
                + paramedic_gap
            )

            phc_leave = (
                doctor_leave
                + nurse_leave
                + paramedic_leave
            )

            phc_unavailable = (
                doctor_unavailable
                + nurse_unavailable
                + paramedic_unavailable
            )

            # ------------------------------------------------
            # Global totals
            # ------------------------------------------------

            total_staff += phc_total
            present_staff += phc_present
            required_staff += phc_required
            total_gap += phc_gap

            total_doctors += doctor_total
            total_nurses += nurse_total
            total_paramedics += paramedic_total

            total_on_leave += phc_leave
            total_unavailable += phc_unavailable

            # ------------------------------------------------
            # State aggregation
            # ------------------------------------------------

            if state not in state_data:
                state_data[state] = {
                    "state": state,
                    "total_staff": 0,
                    "present_staff": 0,
                    "required_staff": 0,
                    "gap": 0,
                    "doctors": 0,
                    "nurses": 0,
                    "paramedics": 0,
                    "on_leave": 0,
                    "unavailable": 0
                }

            state_data[state]["total_staff"] += phc_total
            state_data[state]["present_staff"] += phc_present
            state_data[state]["required_staff"] += phc_required
            state_data[state]["gap"] += phc_gap

            state_data[state]["doctors"] += doctor_total
            state_data[state]["nurses"] += nurse_total
            state_data[state]["paramedics"] += paramedic_total

            state_data[state]["on_leave"] += phc_leave
            state_data[state]["unavailable"] += phc_unavailable

            # ------------------------------------------------
            # Shortage alerts
            # ------------------------------------------------

            role_data = [
                (
                    "Doctors",
                    doctor_required,
                    doctor_present,
                    doctor_gap
                ),
                (
                    "Nurses",
                    nurse_required,
                    nurse_present,
                    nurse_gap
                ),
                (
                    "Paramedics",
                    paramedic_required,
                    paramedic_present,
                    paramedic_gap
                )
            ]

            for role, required, present, gap in role_data:

                if gap > 0:
                    shortage_alerts.append({
                        "phc_id": phc_id,
                        "phc": phc_name,
                        "state": state,
                        "role": role,
                        "required": round(required, 1),
                        "available": round(present, 1),
                        "gap": round(gap, 1)
                    })

        # ----------------------------------------------------
        # State records
        # ----------------------------------------------------

        state_staff = []

        for item in state_data.values():

            state_total = item["total_staff"]
            state_present = item["present_staff"]

            availability = (
                (state_present / state_total) * 100
                if state_total > 0
                else 0
            )

            state_staff.append({
                "state": item["state"],
                "total_staff": round(item["total_staff"], 1),
                "present_staff": round(item["present_staff"], 1),
                "required_staff": round(item["required_staff"], 1),
                "gap": round(item["gap"], 1),
                "doctors": round(item["doctors"], 1),
                "nurses": round(item["nurses"], 1),
                "paramedics": round(item["paramedics"], 1),
                "on_leave": round(item["on_leave"], 1),
                "unavailable": round(item["unavailable"], 1),
                "availability_percent": round(availability, 1)
            })

        # Highest staffing gaps first
        state_staff.sort(
            key=lambda x: x["gap"],
            reverse=True
        )

        # Highest role shortages first
        shortage_alerts.sort(
            key=lambda x: x["gap"],
            reverse=True
        )

        # Keep dashboard response compact
        shortage_alerts = shortage_alerts[:10]

        # ----------------------------------------------------
        # Availability
        # ----------------------------------------------------

        available_staff = present_staff

        total_available_pool = (
            present_staff
            + total_on_leave
            + total_unavailable
        )

        availability_percent = (
            (present_staff / total_staff) * 100
            if total_staff > 0
            else 0
        )

        leave_percent = (
            (total_on_leave / total_available_pool) * 100
            if total_available_pool > 0
            else 0
        )

        unavailable_percent = (
            (total_unavailable / total_available_pool) * 100
            if total_available_pool > 0
            else 0
        )

        payload = {
            "success": True,
            "source": "Firestore phcs.staff",
            "data_type": "synthetic_demo",
            "schema_version": "1.0",

            "summary": {
                "total_staff": round(total_staff, 1),
                "present_staff": round(present_staff, 1),
                "required_staff": round(required_staff, 1),
                "staff_gap": round(total_gap, 1),

                "availability_percent": round(
                    availability_percent,
                    1
                ),

                "doctors": round(total_doctors, 1),
                "nurses": round(total_nurses, 1),
                "paramedics": round(total_paramedics, 1),

                "on_leave": round(total_on_leave, 1),
                "unavailable": round(total_unavailable, 1),

                "leave_percent": round(
                    leave_percent,
                    1
                ),

                "unavailable_percent": round(
                    unavailable_percent,
                    1
                )
            },

            "availability": {
                "available": round(available_staff, 1),
                "on_leave": round(total_on_leave, 1),
                "unavailable": round(total_unavailable, 1)
            },

            "state_staff": state_staff,

            "shortage_alerts": shortage_alerts,

            "generated_at": datetime.utcnow().isoformat(),
            "updated_display": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            )
        }

        payload = make_json_safe(payload)

        _medical_staff_cache = {
            "timestamp": now,
            "payload": payload
        }

        return jsonify(payload)

    except Exception as e:

        print(
            "MEDICAL STAFF API ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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

# ============================================================
# RESOURCE TRANSFER API
# ============================================================

RESOURCE_TRANSFER_CACHE_TTL = 300

_resource_transfer_cache = {
    "timestamp": 0.0,
    "payload": None,
}


@app.route("/api/resource-transfer", methods=["GET"])
def resource_transfer_api():
    """
    Return the latest saved redistribution optimizer results.

    Important:
    - Does NOT run find_source_phcs()
    - Reads only transfer_recommendations
    - Deduplicates older recommendations
    - Preserves all source PHCs from the latest optimizer run
    """

    try:

        # ====================================================
        # CACHE
        # ====================================================

        now = time.time()

        if (
            _resource_transfer_cache["payload"] is not None
            and
            now - _resource_transfer_cache["timestamp"]
            < RESOURCE_TRANSFER_CACHE_TTL
        ):
            return jsonify(
                _resource_transfer_cache["payload"]
            )

        # ====================================================
        # READ SAVED REDISTRIBUTION RESULTS
        # ====================================================

        recommendation_docs = list(
            db.collection(
                "transfer_recommendations"
            )
            .order_by(
                "created_at",
                direction="DESCENDING"
            )
            .limit(50)
            .stream()
        )

        # ====================================================
        # KEEP ONLY THE LATEST RECOMMENDATION FOR:
        #
        # destination PHC + medicine
        #
        # This removes repeated older optimizer runs.
        # ====================================================

        latest_recommendations = {}

        for doc in recommendation_docs:

            result = doc.to_dict() or {}

            destination_phc = str(
                result.get(
                    "destination_phc",
                    ""
                )
            ).strip()

            destination_name = str(
                result.get(
                    "destination_name",
                    destination_phc or "Destination PHC"
                )
            ).strip()

            medicine = str(
                result.get(
                    "medicine",
                    "Medicine"
                )
            ).strip()

            # Unique key
            recommendation_key = (
                destination_phc.lower(),
                medicine.lower()
            )

            # Because documents are ordered newest first,
            # the first document is the latest one.
            if recommendation_key not in latest_recommendations:

                latest_recommendations[
                    recommendation_key
                ] = {
                    "id": doc.id,
                    "result": result,
                    "destination_phc":
                        destination_phc,
                    "destination_name":
                        destination_name,
                    "medicine":
                        medicine,
                }

        # ====================================================
        # BUILD TRANSFER ROWS
        # ====================================================

        transfers = []

        medicine_totals = {}

        total_units = 0.0

        today_string = datetime.now().date().isoformat()

        transfers_today = 0

        pending_recommendations = 0
        transit_recommendations = 0
        completed_recommendations = 0

        # ====================================================
        # PROCESS LATEST RECOMMENDATIONS
        # ====================================================

        for recommendation in latest_recommendations.values():

            result = recommendation["result"]

            recommendation_id = recommendation["id"]

            destination_phc = (
                recommendation[
                    "destination_phc"
                ]
            )

            destination_name = (
                recommendation[
                    "destination_name"
                ]
            )

            medicine = (
                recommendation[
                    "medicine"
                ]
            )

            action = str(
                result.get(
                    "action",
                    "NO_ACTION_REQUIRED"
                )
            )

            raw_status = str(
                result.get(
                    "status",
                    "pending"
                )
            ).strip().lower()

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if raw_status in (
                "completed",
                "complete"
            ):

                display_status = "Completed"

                completed_recommendations += 1

            elif raw_status in (
                "in_transit",
                "in transit",
                "transit"
            ):

                display_status = "In Transit"

                transit_recommendations += 1

            else:

                display_status = "Pending"

                pending_recommendations += 1

            # ------------------------------------------------
            # Created date
            # ------------------------------------------------

            created_at = result.get(
                "created_at"
            )

            if created_at:

                created_string = str(
                    created_at
                )

                if created_string.startswith(
                    today_string
                ):
                    is_today = True
                else:
                    is_today = False

            else:

                is_today = False

            # ------------------------------------------------
            # Shortage
            # ------------------------------------------------

            shortage = float(
                result.get(
                    "shortage",
                    0
                )
                or 0
            )

            remaining_shortage = float(
                result.get(
                    "remaining_shortage",
                    0
                )
                or 0
            )

            # ------------------------------------------------
            # Priority
            # ------------------------------------------------

            if shortage >= 250:

                priority = "Critical"

            elif shortage >= 100:

                priority = "High"

            elif shortage > 0:

                priority = "Medium"

            else:

                priority = "Low"

            # ------------------------------------------------
            # Actual optimizer transfer plan
            # ------------------------------------------------

            transfer_plan = (
                result.get(
                    "transfer_plan"
                )
                or []
            )

            recommendation_had_transfer = False

            for transfer in transfer_plan:

                quantity = float(
                    transfer.get(
                        "recommended_quantity",
                        0
                    )
                    or 0
                )

                if quantity <= 0:
                    continue

                recommendation_had_transfer = True

                source_phc = str(
                    transfer.get(
                        "source_phc",
                        ""
                    )
                ).strip()

                source_name = str(
                    transfer.get(
                        "source_name",
                        source_phc or "Source PHC"
                    )
                ).strip()

                distance = transfer.get(
                    "distance_km"
                )

                optimization_score = (
                    transfer.get(
                        "optimization_score"
                    )
                )

                transfer_row = {

                    "id":
                        recommendation_id,

                    "source_phc":
                        source_phc,

                    "source_name":
                        source_name,

                    "destination_phc":
                        destination_phc,

                    "destination_name":
                        destination_name,

                    "medicine":
                        medicine,

                    "quantity":
                        round(
                            quantity,
                            1
                        ),

                    "distance_km":
                        (
                            round(
                                float(distance),
                                1
                            )
                            if distance is not None
                            else None
                        ),

                    "optimization_score":
                        (
                            round(
                                float(
                                    optimization_score
                                ),
                                2
                            )
                            if optimization_score is not None
                            else None
                        ),

                    "shortage":
                        round(
                            shortage,
                            1
                        ),

                    "remaining_shortage":
                        round(
                            remaining_shortage,
                            1
                        ),

                    "priority":
                        priority,

                    "status":
                        display_status,

                    "action":
                        action,

                    "reason":
                        (
                            "Forecasted stock shortage"
                            if shortage > 0
                            else
                            "Redistribution recommended"
                        ),

                    "created_at":
                        created_at,
                }

                transfers.append(
                    transfer_row
                )

                total_units += quantity

                # --------------------------------------------
                # Transfers today
                # --------------------------------------------

                if is_today:
                    transfers_today += 1

                # --------------------------------------------
                # Medicine summary
                # --------------------------------------------

                if medicine not in medicine_totals:

                    medicine_totals[
                        medicine
                    ] = {

                        "medicine":
                            medicine,

                        "quantity":
                            0.0,

                        "transfers":
                            0,
                    }

                medicine_totals[
                    medicine
                ]["quantity"] += quantity

                medicine_totals[
                    medicine
                ]["transfers"] += 1

        # ====================================================
        # SORT
        # ====================================================

        transfers.sort(
            key=lambda item: (
                item.get(
                    "optimization_score"
                )
                if item.get(
                    "optimization_score"
                ) is not None
                else -1
            ),
            reverse=True
        )

        # ====================================================
        # MEDICINE SUMMARY
        # ====================================================

        medicine_summary = sorted(
            medicine_totals.values(),
            key=lambda item:
                item["quantity"],
            reverse=True
        )

        for item in medicine_summary:

            item["quantity"] = round(
                item["quantity"],
                1
            )

        # ====================================================
        # RECENT TRANSFERS
        # ====================================================

        recent_transfers = transfers[:5]

        # ====================================================
        # SUMMARY
        # ====================================================

        summary = {

            "transfers_today":
                transfers_today,

            "in_transit":
                transit_recommendations,

            "completed":
                completed_recommendations,

            "pending_approval":
                pending_recommendations,

            "total_units":
                round(
                    total_units,
                    1
                ),

            "updated_display":
                datetime.now().strftime(
                    "%d %b %Y, %I:%M %p"
                ),

            "model":
                (
                    "Distance + Safe Surplus "
                    "Weighted Optimization"
                ),
        }

        # ====================================================
        # FINAL PAYLOAD
        # ====================================================

        payload = make_json_safe({

            "success":
                True,

            "source":
                "Firestore transfer_recommendations",

            "summary":
                summary,

            "transfers":
                transfers,

            "medicine_summary":
                medicine_summary,

            "recent_transfers":
                recent_transfers,

            "generated_at":
                datetime.now().isoformat(),

        })

        # ====================================================
        # SAVE CACHE
        # ====================================================

        _resource_transfer_cache[
            "timestamp"
        ] = time.time()

        _resource_transfer_cache[
            "payload"
        ] = payload

        return jsonify(
            payload
        )

    except Exception as error:

        print(
            "Resource Transfer API error: "
            f"{type(error).__name__}: {error}"
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error),

            "error_type":
                type(error).__name__,

        }), 500

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

        cache_key = (phc_id.upper(), medicine.lower())
        cached = _ai_analysis_cache.get(cache_key)
        if cached and time.time() - cached["timestamp"] < AI_ANALYSIS_CACHE_TTL:
            print(f"♻️ Returning cached AI analysis: {phc_id} / {medicine}")
            return jsonify(cached["result"])

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

        _ai_analysis_cache[cache_key] = {
            "timestamp": time.time(),
            "result": result,
        }

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