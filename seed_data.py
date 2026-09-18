"""
ArogyaFlow AI - Firebase-friendly demo data seeder

Seeds only a small operational dataset into Firestore:
- 50 Maharashtra PHCs
- 8 medicines per PHC
- 14 days of demand history
- 5 demo alerts

Large ML/training datasets remain local under data_pipeline/temporal/.

IMPORTANT:
The demand, inventory, outbreak, anomaly, and supply-disruption values created
by this script are synthetic demonstration data, NOT real PHC consumption data.
"""

import math
import random
from datetime import datetime, timedelta

from firebase_config import db


# ============================================================
# CONFIGURATION
# ============================================================

NUM_PHCS = 50

# Only 14 days are stored in Firestore.
# This is enough for the current feature-building logic.
HISTORY_DAYS = 30

RANDOM_SEED = 42


# ============================================================
# MEDICINES
# ============================================================

MEDICINES = [
    {
        "id": "MED-001",
        "name": "Paracetamol",
        "category": "Analgesic/Antipyretic",
        "unit": "tablets",
        "base_demand": 35,
        "minimum_stock": 360,
        "maximum_stock": 1500,
    },
    {
        "id": "MED-002",
        "name": "Amoxicillin",
        "category": "Antibiotic",
        "unit": "capsules",
        "base_demand": 18,
        "minimum_stock": 180,
        "maximum_stock": 900,
    },
    {
        "id": "MED-003",
        "name": "ORS Sachet",
        "category": "Rehydration",
        "unit": "sachets",
        "base_demand": 22,
        "minimum_stock": 220,
        "maximum_stock": 1050,
    },
    {
        "id": "MED-004",
        "name": "Metformin",
        "category": "Antidiabetic",
        "unit": "tablets",
        "base_demand": 14,
        "minimum_stock": 210,
        "maximum_stock": 900,
    },
    {
        "id": "MED-005",
        "name": "Amlodipine",
        "category": "Antihypertensive",
        "unit": "tablets",
        "base_demand": 12,
        "minimum_stock": 180,
        "maximum_stock": 750,
    },
    {
        "id": "MED-006",
        "name": "Cetirizine",
        "category": "Antihistamine",
        "unit": "tablets",
        "base_demand": 16,
        "minimum_stock": 112,
        "maximum_stock": 700,
    },
    {
        "id": "MED-007",
        "name": "Iron Folic Acid Tablet",
        "category": "Nutritional Supplement",
        "unit": "tablets",
        "base_demand": 20,
        "minimum_stock": 300,
        "maximum_stock": 900,
    },
    {
        "id": "MED-008",
        "name": "Omeprazole",
        "category": "Gastrointestinal",
        "unit": "capsules",
        "base_demand": 10,
        "minimum_stock": 100,
        "maximum_stock": 600,
    },
]


# ============================================================
# 50 DEMO PHCs
# ============================================================

PHCS = [
    ("Palghar", 19.697107, 72.763725),
    ("Vasai", 19.3919, 72.8397),
    ("Virar", 19.4559, 72.8111),
    ("Boisar", 19.8030, 72.7550),
    ("Dahanu", 19.9900, 72.7430),
    ("Nalasopara", 19.4150, 72.8050),
    ("Thane Rural", 19.2183, 72.9781),
    ("Bhiwandi", 19.2813, 73.0483),
    ("Shahapur", 19.4526, 73.3257),
    ("Murbad", 19.2550, 73.3900),
    ("Nashik Rural", 20.0059, 73.7897),
    ("Sinnar", 19.8451, 73.9986),
    ("Igatpuri", 19.6950, 73.5620),
    ("Dindori", 20.2020, 73.8320),
    ("Yeola", 20.0420, 74.4890),
    ("Pune Rural", 18.5204, 73.8567),
    ("Baramati", 18.1517, 74.5777),
    ("Shirur", 18.8270, 74.3740),
    ("Junnar", 19.2080, 73.8750),
    ("Khed", 18.8400, 73.8800),
    ("Satara", 17.6805, 74.0183),
    ("Karad", 17.2890, 74.1810),
    ("Wai", 17.9530, 73.8900),
    ("Phaltan", 17.9910, 74.4320),
    ("Kolhapur Rural", 16.7050, 74.2430),
    ("Ichalkaranji", 16.6910, 74.4600),
    ("Sangli Rural", 16.8524, 74.5815),
    ("Miraj", 16.8270, 74.6420),
    ("Solapur Rural", 17.6599, 75.9064),
    ("Barshi", 18.2345, 75.6928),
    ("Ahmednagar", 19.0952, 74.7496),
    ("Kopargaon", 19.8820, 74.4760),
    ("Jalgaon Rural", 21.0077, 75.5626),
    ("Bhusawal", 21.0450, 75.8010),
    ("Dhule", 20.9042, 74.7749),
    ("Nandurbar", 21.3650, 74.2400),
    ("Aurangabad Rural", 19.8762, 75.3433),
    ("Paithan", 19.4760, 75.3860),
    ("Beed Rural", 18.9891, 75.7600),
    ("Latur Rural", 18.4088, 76.5604),
    ("Osmanabad", 18.1860, 76.0419),
    ("Nanded Rural", 19.1383, 77.3210),
    ("Parbhani", 19.2600, 76.7700),
    ("Akola Rural", 20.7000, 77.0100),
    ("Amravati Rural", 20.9320, 77.7520),
    ("Nagpur Rural", 21.1458, 79.0882),
    ("Wardha", 20.7453, 78.6022),
    ("Bhandara", 21.1700, 79.6500),
    ("Gondia", 21.4624, 80.2209),
    ("Chandrapur", 19.9615, 79.2961),
]


# ============================================================
# ID HELPERS
# ============================================================

def medicine_document_id(phc_id, medicine_name):
    safe_name = (
        medicine_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"{phc_id}_{safe_name}"


def demand_document_id(phc_id, medicine_name, date_value):
    safe_name = (
        medicine_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return (
        f"{phc_id}_{safe_name}_"
        f"{date_value.strftime('%Y%m%d')}"
    )


# ============================================================
# FEATURE HELPERS
# ============================================================

def seasonal_factor(day_of_year):
    return (
        1.0
        + 0.12
        * math.sin(
            2.0 * math.pi * day_of_year / 365.0
        )
    )


def weekday_factor(day_of_week):
    factors = {
        0: 1.05,  # Monday
        1: 1.08,
        2: 1.00,
        3: 1.04,
        4: 1.10,
        5: 0.92,
        6: 0.78,  # Sunday
    }

    return factors.get(day_of_week, 1.0)


def population_factor(index):
    return round(
        0.80 + ((index * 17) % 51) / 100.0,
        3,
    )


def capacity_factor(index):
    return round(
        0.80 + ((index * 13) % 41) / 100.0,
        3,
    )


def urban_rural(index):
    urban_indices = {
        1, 2, 5, 7, 15,
        27, 28, 30, 34,
        36, 45,
    }

    if index in urban_indices:
        return "Urban"

    return "Rural"


def outbreak_flag(phc_index, date_value):
    if phc_index % 10 != 0:
        return 0

    start_date = datetime(
        2026,
        8,
        1,
    ).date() + timedelta(days=5)

    end_date = start_date + timedelta(days=8)

    return int(
        start_date <= date_value <= end_date
    )


# ============================================================
# DEMAND GENERATION
# ============================================================

def generate_demand(
    medicine,
    phc_index,
    date_value,
    rng,
):
    demand = (
        medicine["base_demand"]
        * population_factor(phc_index)
        * capacity_factor(phc_index)
        * weekday_factor(
            date_value.weekday()
        )
        * seasonal_factor(
            date_value.timetuple().tm_yday
        )
        * rng.uniform(
            0.88,
            1.12,
        )
    )

    outbreak = outbreak_flag(
        phc_index,
        date_value,
    )

    # Outbreak effect on common acute medicines.
    if (
        outbreak
        and medicine["id"]
        in {
            "MED-001",
            "MED-002",
            "MED-003",
            "MED-006",
        }
    ):
        demand *= 1.75

    return (
        max(
            1,
            int(round(demand)),
        ),
        outbreak,
    )


def patient_footfall(
    demand,
    phc_index,
    rng,
):
    footfall = (
        demand
        * rng.uniform(
            2.8,
            4.4,
        )
        * population_factor(phc_index)
    )

    return max(
        demand,
        int(round(footfall)),
    )


# ============================================================
# INVENTORY GENERATION
# ============================================================

def generate_inventory(
    phc_id,
    medicine,
    phc_index,
    rng,
):
    minimum = medicine["minimum_stock"]
    base = medicine["base_demand"]

    # Every 10th PHC gets tighter inventory so that
    # the redistribution demo has meaningful cases.
    if phc_index % 10 == 0:
        current_stock = int(
            minimum
            * rng.uniform(
                0.85,
                1.45,
            )
        )
    else:
        current_stock = int(
            base
            * rng.uniform(
                18,
                35,
            )
        )

    if current_stock <= minimum:
        status = "low"
    elif current_stock <= minimum * 1.5:
        status = "medium"
    else:
        status = "healthy"

    daily_demand = max(
        1,
        round(
            base
            * population_factor(phc_index)
            * 0.95,
            2,
        ),
    )

    return {
        "phc_id": phc_id,
        "medicine_id": medicine["id"],
        "medicine_name": medicine["name"],
        "category": medicine["category"],
        "unit": medicine["unit"],
        "current_stock": float(
            current_stock
        ),
        "daily_demand": float(
            daily_demand
        ),
        "minimum_stock": float(
            minimum
        ),
        "maximum_stock": float(
            medicine["maximum_stock"]
        ),
        "status": status,
        "updated_at": datetime.utcnow(),
    }


# ============================================================
# SEED PHCs
# ============================================================

def seed_phcs():
    print(
        "\n🏥 Creating 50 Maharashtra demo PHCs..."
    )

    phc_ids = []

    for index, (
        name,
        latitude,
        longitude,
    ) in enumerate(
        PHCS,
        start=1,
    ):
        phc_id = f"PHC{index:03d}"

        phc_ids.append(
            phc_id
        )

        data = {
            "phc_id": phc_id,
            "name": f"{name} PHC",
            "state": "Maharashtra",
            "district": (
                name
                .replace(
                    " Rural",
                    "",
                )
            ),
            "phc_type": "PHC",
            "urban_rural": urban_rural(
                index
            ),
            "latitude": latitude,
            "longitude": longitude,
            "population_factor": population_factor(
                index
            ),
            "capacity_factor": capacity_factor(
                index
            ),
            "created_at": datetime.utcnow(),
        }

        db.collection(
            "phcs"
        ).document(
            phc_id
        ).set(data)

        print(
            f"  ✓ {phc_id} - {name} PHC"
        )

    return phc_ids


# ============================================================
# SEED MEDICINES
# ============================================================

def seed_medicines(
    phc_ids,
    rng,
):
    print(
        "\n💊 Creating medicine inventory..."
    )

    count = 0

    for phc_index, phc_id in enumerate(
        phc_ids,
        start=1,
    ):
        for medicine in MEDICINES:

            document_id = medicine_document_id(
                phc_id,
                medicine["name"],
            )

            data = generate_inventory(
                phc_id,
                medicine,
                phc_index,
                rng,
            )

            db.collection(
                "medicines"
            ).document(
                document_id
            ).set(data)

            count += 1

    print(
        f"  ✓ Created {count} medicine records"
    )


# ============================================================
# SEED SMALL DEMAND HISTORY
# ============================================================

def seed_demand_history(
    phc_ids,
    rng,
):
    print(
        f"\n📊 Creating SMALL AI-compatible "
        f"demand history ({HISTORY_DAYS} days)..."
    )

    # Fixed date window so repeated runs produce
    # the same document IDs.
    end_date = datetime(
        2026,
        8,
        23,
    ).date()

    start_date = (
        end_date
        - timedelta(
            days=HISTORY_DAYS - 1
        )
    )

    count = 0

    for phc_index, phc_id in enumerate(
        phc_ids,
        start=1,
    ):

        for medicine in MEDICINES:

            for offset in range(
                HISTORY_DAYS
            ):

                date_value = (
                    start_date
                    + timedelta(
                        days=offset
                    )
                )

                demand, outbreak = (
                    generate_demand(
                        medicine,
                        phc_index,
                        date_value,
                        rng,
                    )
                )

                anomaly = int(
                    phc_index % 17 == 0
                    and offset == 10
                    and medicine["id"]
                    in {
                        "MED-001",
                        "MED-003",
                    }
                )

                if anomaly:
                    demand = int(
                        round(
                            demand * 2.2
                        )
                    )

                supply_disruption = int(
                    phc_index % 13 == 0
                    and offset in {
                        4,
                        11,
                    }
                )

                data = {
                    "phc_id": phc_id,
                    "medicine_name": medicine["name"],
                    "medicine_id": medicine["id"],
                    "category": medicine["category"],
                    "date": datetime.combine(
                        date_value,
                        datetime.min.time(),
                    ),
                    "demand": float(
                        demand
                    ),
                    "patient_footfall": float(
                        patient_footfall(
                            demand,
                            phc_index,
                            rng,
                        )
                    ),
                    "outbreak_flag": outbreak,
                    "supply_disruption_flag": (
                        supply_disruption
                    ),
                    "seasonal_factor": float(
                        seasonal_factor(
                            date_value.timetuple().tm_yday
                        )
                    ),
                    "anomaly_flag": anomaly,
                    "created_at": datetime.utcnow(),
                }

                record_id = demand_document_id(
                    phc_id,
                    medicine["name"],
                    date_value,
                )

                db.collection(
                    "demand_history"
                ).document(
                    record_id
                ).set(data)

                count += 1

    print(
        f"  ✓ Created {count} demand-history records"
    )

    print(
        f"    = {NUM_PHCS} PHCs × "
        f"{len(MEDICINES)} medicines × "
        f"{HISTORY_DAYS} days"
    )


# ============================================================
# SEED ALERTS
# ============================================================

def seed_alerts():
    print(
        "\n🚨 Creating demo alerts..."
    )

    alerts = [
        (
            "ALERT001",
            "PHC001",
            "Paracetamol",
            "STOCKOUT_RISK",
            "HIGH",
            "Paracetamol stock is approaching the minimum threshold.",
        ),
        (
            "ALERT002",
            "PHC011",
            "Amoxicillin",
            "DEMAND_SPIKE",
            "MEDIUM",
            "Unusual demand increase detected for Amoxicillin.",
        ),
        (
            "ALERT003",
            "PHC021",
            "ORS Sachet",
            "REDISTRIBUTION",
            "MEDIUM",
            "Redistribution from a nearby PHC may reduce shortage risk.",
        ),
        (
            "ALERT004",
            "PHC031",
            "Metformin",
            "ANOMALY",
            "MEDIUM",
            "Unusual medicine behavior detected.",
        ),
        (
            "ALERT005",
            "PHC041",
            "Amlodipine",
            "LOW_STOCK",
            "HIGH",
            "Current Amlodipine inventory is below the preferred buffer.",
        ),
    ]

    for (
        alert_id,
        phc_id,
        medicine,
        alert_type,
        severity,
        message,
    ) in alerts:

        db.collection(
            "alerts"
        ).document(
            alert_id
        ).set(
            {
                "alert_id": alert_id,
                "phc_id": phc_id,
                "medicine_name": medicine,
                "type": alert_type,
                "severity": severity,
                "message": message,
                "created_at": datetime.utcnow(),
            }
        )

    print(
        f"  ✓ Created {len(alerts)} alerts"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "\n============================================"
    )
    print(
        "       AROGYAFLOW AI DATA SEEDER"
    )
    print(
        "============================================"
    )

    print(
        "\n🔥 Connected to Firebase Firestore"
    )

    print(
        "\nℹ️ Firebase-friendly mode:"
    )

    print(
        f"   {NUM_PHCS} PHCs"
    )

    print(
        f"   {len(MEDICINES)} medicines"
    )

    print(
        f"   {HISTORY_DAYS} days of demand history"
    )

    print(
        "\n⚠️ Demand history is SYNTHETIC demo data."
    )

    print(
        "   Large ML/training datasets stay local."
    )

    rng = random.Random(
        RANDOM_SEED
    )

    phc_ids = seed_phcs()

    seed_medicines(
        phc_ids,
        rng,
    )

    seed_demand_history(
        phc_ids,
        rng,
    )

    seed_alerts()

    print(
        "\n============================================"
    )

    print(
        "             SEEDING COMPLETE"
    )

    print(
        "============================================"
    )

    print(
        "\n📦 Firestore demo data:"
    )

    print(
        f"   PHCs:              {NUM_PHCS}"
    )

    print(
        f"   Medicines:         "
        f"{NUM_PHCS * len(MEDICINES)}"
    )

    print(
        f"   Demand history:    "
        f"{NUM_PHCS * len(MEDICINES) * HISTORY_DAYS}"
    )

    print(
        "   Alerts:             5"
    )

    print(
        "\n🧠 ML training datasets remain local."
    )

    print(
        "🚀 You can now run: python app.py"
    )


if __name__ == "__main__":
    main()