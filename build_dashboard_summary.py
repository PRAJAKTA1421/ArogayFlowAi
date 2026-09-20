from collections import defaultdict
from datetime import datetime
from firebase_config import db

print("=" * 60)
print("AROGYAFLOW AI - BUILD DASHBOARD SUMMARY")
print("=" * 60)

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

phc_docs = list(db.collection("phcs").stream())
phcs = {}
for doc in phc_docs:
    item = doc.to_dict() or {}
    item["id"] = doc.id
    phcs[doc.id] = item
print(f"PHCs loaded: {len(phcs)}")

medicine_docs = list(db.collection("medicines").stream())
medicines = []
for doc in medicine_docs:
    item = doc.to_dict() or {}
    item["id"] = doc.id
    medicines.append(item)
print(f"Medicine records loaded: {len(medicines)}")

print("Reading demand_history ONCE. Do not run this script repeatedly.")
demand_docs = list(db.collection("demand_history").stream())
print(f"Demand-history documents loaded: {len(demand_docs)}")

records = []
for doc in demand_docs:
    item = doc.to_dict() or {}
    phc_id = item.get("phc_id")
    medicine_name = item.get("medicine_name")
    d = date_key(item.get("date"))
    if not phc_id or not medicine_name or not d:
        continue
    records.append({
        "phc_id": phc_id,
        "medicine_name": medicine_name,
        "date": d,
        "demand": num(item.get("demand")),
        "patient_footfall": num(item.get("patient_footfall")),
    })

all_dates = sorted({r["date"] for r in records if r["date"]})
latest_date = all_dates[-1] if all_dates else ""
last_seven_dates = all_dates[-7:]

demand_by_date = defaultdict(float)
for r in records:
    if r["date"] in last_seven_dates:
        demand_by_date[r["date"]] += r["demand"]

demand_trend = []
for d in last_seven_dates:
    try:
        label = datetime.strptime(d, "%Y-%m-%d").strftime("%b %d")
    except ValueError:
        label = d
    demand_trend.append({"date": d, "label": label, "demand": round(demand_by_date[d], 2)})

latest_by_phc = {}
for r in records:
    if r["phc_id"] not in latest_by_phc or r["date"] > latest_by_phc[r["phc_id"]]["date"]:
        latest_by_phc[r["phc_id"]] = {"date": r["date"], "patient_footfall": r["patient_footfall"]}
patients_today = sum(x["patient_footfall"] for x in latest_by_phc.values())

demand_by_combo = defaultdict(list)
for r in records:
    demand_by_combo[(r["phc_id"], r["medicine_name"])].append((r["date"], r["demand"]))

phc_risk_map = {}
combo_risks = []
for med in medicines:
    phc_id = med.get("phc_id")
    medicine_name = med.get("medicine_name", "Medicine")
    current = num(med.get("current_stock"))
    minimum = num(med.get("minimum_stock"))
    history = sorted(demand_by_combo.get((phc_id, medicine_name), []), key=lambda x: x[0])[-7:]
    avg_demand = sum(x[1] for x in history) / len(history) if history else num(med.get("daily_demand"))
    remaining = current - avg_demand * 7
    shortage = max(minimum - remaining, 0)
    if current <= 0 or remaining < 0:
        risk = "CRITICAL"
    elif (shortage > minimum * 0.5 if minimum > 0 else shortage > 0):
        risk = "HIGH"
    elif shortage > 0 or current < minimum:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    score = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[risk]
    combo = {"phc_id": phc_id, "medicine": medicine_name, "shortage": round(shortage, 2), "risk": risk}
    combo_risks.append(combo)
    prev = phc_risk_map.get(phc_id)
    if prev is None or (score, shortage) > (prev["score"], prev["shortage"]):
        phc_risk_map[phc_id] = {"score": score, "shortage": shortage, "risk": risk, "medicine": medicine_name}

priority_rows = []
for phc_id, r in phc_risk_map.items():
    if r["score"] < 1:
        continue
    phc = phcs.get(phc_id, {})
    priority_rows.append({
        "name": phc.get("name", phc_id),
        "state": phc.get("state", "India"),
        "risk": r["risk"],
        "issue": f"{r['medicine']} inventory pressure",
        "action": "Review transfer" if r["score"] >= 2 else "Monitor / replenish",
        "shortage": round(r["shortage"], 2),
    })
priority_rows.sort(key=lambda x: ({"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}.get(x["risk"], 0), x["shortage"]), reverse=True)

priority_count = sum(1 for r in phc_risk_map.values() if r["score"] >= 1)
critical_phcs = sum(1 for r in phc_risk_map.values() if r["score"] >= 3)
high_phcs = sum(1 for r in phc_risk_map.values() if r["score"] == 2)

shortages = [x for x in combo_risks if x["shortage"] > 0]
shortages.sort(key=lambda x: x["shortage"], reverse=True)
top_shortage = None
if shortages:
    x = shortages[0]
    top_shortage = {"medicine": x["medicine"], "shortage": x["shortage"], "phc": phcs.get(x["phc_id"], {}).get("name", x["phc_id"])}

summary = {
    "generated_at": datetime.utcnow().isoformat(),
    "latest_demand_date": latest_date,
    "patients_today": round(patients_today, 2),
    "demand_trend": demand_trend,
    "priority_count": priority_count,
    "critical_phcs": critical_phcs,
    "high_phcs": high_phcs,
    "priority_phcs": priority_rows[:8],
    "top_shortage": top_shortage,
    "source": "Aggregated once from demand_history for dashboard use",
}

db.collection("dashboard_summary").document("network").set(summary)

print("=" * 60)
print("SUMMARY CREATED: dashboard_summary/network")
print(f"Latest demand date: {latest_date}")
print(f"Patients: {patients_today:.0f}")
print(f"Trend points: {len(demand_trend)}")
print(f"Priority PHCs: {priority_count}")
print(f"Critical PHCs: {critical_phcs}")
print(f"High PHCs: {high_phcs}")
print("Do NOT run this script repeatedly.")
print("=" * 60)
