import random
from datetime import datetime, timedelta

from firebase_config import db


# ============================================================
# CONFIGURATION
# ============================================================

NUM_PHCS = 50
HISTORY_DAYS = 30


# ============================================================
# SAMPLE DATA
# ============================================================

PHC_NAMES = [
    "Palghar PHC",
    "Vasai PHC",
    "Virar PHC",
    "Boisar PHC",
    "Dahanu PHC",
    "Nalasopara PHC",
    "Thane Rural PHC",
    "Bhiwandi PHC",
    "Shahapur PHC",
    "Murbad PHC",
    "Nashik Rural PHC",
    "Sinnar PHC",
    "Igatpuri PHC",
    "Dindori PHC",
    "Yeola PHC",
    "Pune Rural PHC",
    "Baramati PHC",
    "Shirur PHC",
    "Junnar PHC",
    "Khed PHC",
    "Satara PHC",
    "Karad PHC",
    "Wai PHC",
    "Phaltan PHC",
    "Kolhapur Rural PHC",
    "Ichalkaranji PHC",
    "Sangli Rural PHC",
    "Miraj PHC",
    "Solapur Rural PHC",
    "Barshi PHC",
    "Ahmednagar PHC",
    "Kopargaon PHC",
    "Jalgaon Rural PHC",
    "Bhusawal PHC",
    "Dhule PHC",
    "Nandurbar PHC",
    "Aurangabad Rural PHC",
    "Paithan PHC",
    "Beed Rural PHC",
    "Latur Rural PHC",
    "Osmanabad PHC",
    "Nanded Rural PHC",
    "Parbhani PHC",
    "Akola Rural PHC",
    "Amravati Rural PHC",
    "Nagpur Rural PHC",
    "Wardha PHC",
    "Bhandara PHC",
    "Gondia PHC",
    "Chandrapur PHC",
]

# ============================================================
# REAL MAHARASHTRA LOCATION COORDINATES
# ============================================================

PHC_COORDINATES = {

    "Palghar PHC": (19.697107, 72.763725),
    "Vasai PHC": (19.342820, 72.805440),
    "Virar PHC": (19.455900, 72.811400),
    "Boisar PHC": (19.803000, 72.755000),
    "Dahanu PHC": (19.990000, 72.740000),
    "Nalasopara PHC": (19.415000, 72.805000),

    "Thane Rural PHC": (19.218300, 72.978100),
    "Bhiwandi PHC": (19.281300, 73.048300),
    "Shahapur PHC": (19.452600, 73.325700),
    "Murbad PHC": (19.254600, 73.390700),

    "Nashik Rural PHC": (19.997500, 73.789800),
    "Sinnar PHC": (19.845100, 73.998700),
    "Igatpuri PHC": (19.695900, 73.562100),
    "Dindori PHC": (20.200000, 73.833300),
    "Yeola PHC": (20.042700, 74.489300),

    "Pune Rural PHC": (18.520400, 73.856700),
    "Baramati PHC": (18.151700, 74.577700),
    "Shirur PHC": (18.827600, 74.375800),
    "Junnar PHC": (19.207700, 73.875200),
    "Khed PHC": (18.440900, 73.860300),

    "Satara PHC": (17.680500, 74.018300),
    "Karad PHC": (17.289600, 74.181100),
    "Wai PHC": (17.952700, 73.890700),
    "Phaltan PHC": (17.991100, 74.431800),

    "Kolhapur Rural PHC": (16.705000, 74.243300),
    "Ichalkaranji PHC": (16.691200, 74.460500),
    "Sangli Rural PHC": (16.852400, 74.581500),
    "Miraj PHC": (16.827800, 74.644200),

    "Solapur Rural PHC": (17.659900, 75.906400),
    "Barshi PHC": (18.234500, 75.692800),

    "Ahmednagar PHC": (19.094800, 74.748000),
    "Kopargaon PHC": (19.882600, 74.476300),

    "Jalgaon Rural PHC": (21.007700, 75.562600),
    "Bhusawal PHC": (21.045500, 75.780400),
    "Dhule PHC": (20.904200, 74.774900),
    "Nandurbar PHC": (21.366700, 74.233300),

    "Aurangabad Rural PHC": (19.876200, 75.343300),
    "Paithan PHC": (19.477700, 75.381100),
    "Beed Rural PHC": (18.989100, 75.760100),

    "Latur Rural PHC": (18.408800, 76.560400),
    "Osmanabad PHC": (18.186000, 76.041900),
    "Nanded Rural PHC": (19.138300, 77.321000),
    "Parbhani PHC": (19.260800, 76.776700),

    "Akola Rural PHC": (20.700200, 77.008200),
    "Amravati Rural PHC": (20.937400, 77.779600),
    "Nagpur Rural PHC": (21.145800, 79.088200),
    "Wardha PHC": (20.745300, 78.602200),
    "Bhandara PHC": (21.170000, 79.650000),
    "Gondia PHC": (21.462400, 80.220900),
    "Chandrapur PHC": (19.961500, 79.296100),
}

MEDICINES = [
    {
        "name": "Paracetamol",
        "unit": "tablets",
        "base_demand": 120,
    },
    {
        "name": "Amoxicillin",
        "unit": "tablets",
        "base_demand": 90,
    },
    {
        "name": "Azithromycin",
        "unit": "tablets",
        "base_demand": 70,
    },
    {
        "name": "ORS",
        "unit": "packets",
        "base_demand": 100,
    },
    {
        "name": "Ibuprofen",
        "unit": "tablets",
        "base_demand": 80,
    },
    {
        "name": "Metformin",
        "unit": "tablets",
        "base_demand": 60,
    },
    {
        "name": "Cetirizine",
        "unit": "tablets",
        "base_demand": 75,
    },
    {
        "name": "Ciprofloxacin",
        "unit": "tablets",
        "base_demand": 55,
    },
    {
        "name": "Omeprazole",
        "unit": "capsules",
        "base_demand": 85,
    },
    {
        "name": "Doxycycline",
        "unit": "tablets",
        "base_demand": 50,
    },
]


def generate_phc_data(index, name):
    """
    Generate one PHC using real geographic
    coordinates for the named Maharashtra location.
    """

    if name not in PHC_COORDINATES:
        raise ValueError(
            f"No coordinates found for {name}"
        )

    latitude, longitude = PHC_COORDINATES[name]

    total_beds = random.randint(20, 120)
    occupied_beds = random.randint(
        int(total_beds * 0.35),
        int(total_beds * 0.92)
    )

    total_staff = random.randint(12, 40)
    present_staff = random.randint(
        max(5, int(total_staff * 0.65)),
        total_staff
    )

    patients_today = random.randint(80, 450)

    return {
        "name": name,
        "district": name.replace(" PHC", ""),
        "state": "Maharashtra",
        "country": "India",

        "latitude": latitude,
        "longitude": longitude,

        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": total_beds - occupied_beds,

        "total_staff": total_staff,
        "present_staff": present_staff,

        "patients_today": patients_today,

        "status": (
            "critical"
            if occupied_beds / total_beds > 0.85
            else "warning"
            if occupied_beds / total_beds > 0.70
            else "normal"
        ),

        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


def generate_medicine_data(phc_id, medicine):
    """
    Generate current medicine inventory.
    """

    base = medicine["base_demand"]

    current_stock = random.randint(
        base * 2,
        base * 12
    )

    minimum_stock = base * 3

    return {
        "phc_id": phc_id,
        "medicine_name": medicine["name"],
        "unit": medicine["unit"],

        "current_stock": current_stock,
        "minimum_stock": minimum_stock,

        "daily_demand": base,

        "status": (
            "critical"
            if current_stock < minimum_stock * 0.5
            else "low"
            if current_stock < minimum_stock
            else "healthy"
        ),

        "updated_at": datetime.utcnow(),
    }


def generate_demand(
    phc_index,
    medicine,
    day_index
):
    """
    Generate realistic historical demand.

    Demand changes based on:

    - base medicine demand
    - weekly pattern
    - seasonal/random variation
    - occasional demand spikes
    """

    base = medicine["base_demand"]

    date = datetime.utcnow() - timedelta(days=day_index)

    # Weekly variation
    weekday_factor = {
        0: 1.05,
        1: 1.00,
        2: 1.08,
        3: 0.95,
        4: 1.12,
        5: 1.18,
        6: 0.82,
    }

    factor = weekday_factor[date.weekday()]

    # Random variation
    noise = random.uniform(0.85, 1.15)

    # Simulated outbreak at some PHCs
    outbreak_factor = 1.0

    if phc_index % 10 == 0 and day_index < 10:
        outbreak_factor = random.uniform(1.4, 1.9)

    demand = (
        base
        * factor
        * noise
        * outbreak_factor
    )

    return max(1, int(demand))


# ============================================================
# SEED PHCs
# ============================================================

def seed_phcs():

    print("\n🏥 Creating PHCs...")

    phc_ids = []

    for index, name in enumerate(PHC_NAMES):

        phc_id = f"PHC{index + 1:03d}"

        data = generate_phc_data(
            index,
            name
        )

        db.collection("phcs").document(phc_id).set(data)

        phc_ids.append(phc_id)

        print(
            f"  ✓ {phc_id} - {name}"
        )

    return phc_ids


# ============================================================
# SEED MEDICINES
# ============================================================

def seed_medicines(phc_ids):

    print("\n💊 Creating medicine inventory...")

    count = 0

    for phc_id in phc_ids:

        for medicine in MEDICINES:

            medicine_id = (
                f"{phc_id}_"
                f"{medicine['name'].lower().replace(' ', '_')}"
            )

            data = generate_medicine_data(
                phc_id,
                medicine
            )

            db.collection("medicines") \
                .document(medicine_id) \
                .set(data)

            count += 1

    print(
        f"  ✓ Created {count} medicine records"
    )


# ============================================================
# SEED DEMAND HISTORY
# ============================================================

def seed_demand_history(phc_ids):

    print(
        "\n📊 Creating historical demand data..."
    )

    count = 0

    for phc_index, phc_id in enumerate(phc_ids):

        for medicine in MEDICINES:

            for day_index in range(HISTORY_DAYS):

                demand = generate_demand(
                    phc_index,
                    medicine,
                    day_index
                )

                date = (
                    datetime.utcnow()
                    - timedelta(days=day_index)
                )

                record_id = (
                    f"{phc_id}_"
                    f"{medicine['name']}_"
                    f"{date.strftime('%Y%m%d')}"
                )

                data = {
                    "phc_id": phc_id,

                    "medicine_name":
                        medicine["name"],

                    "date": date.strftime(
                        "%Y-%m-%d"
                    ),

                    "demand": demand,

                    "patient_footfall":
                        int(demand * random.uniform(
                            2.0,
                            4.0
                        )),

                    "created_at":
                        datetime.utcnow(),
                }

                db.collection(
                    "demand_history"
                ).document(record_id).set(data)

                count += 1

    print(
        f"  ✓ Created {count} demand records"
    )


# ============================================================
# SEED ALERTS
# ============================================================

def seed_alerts(phc_ids):

    print("\n🚨 Creating sample alerts...")

    alerts = []

    for phc_id in phc_ids[:10]:

        alert = {
            "phc_id": phc_id,

            "type": "medicine",

            "medicine_name":
                random.choice(MEDICINES)["name"],

            "severity":
                random.choice([
                    "high",
                    "medium",
                    "critical"
                ]),

            "message":
                "Potential medicine stock-out predicted.",

            "status": "active",

            "created_at":
                datetime.utcnow(),
        }

        alerts.append(alert)

    for index, alert in enumerate(alerts):

        db.collection("alerts") \
            .document(f"ALERT{index + 1:03d}") \
            .set(alert)

    print(
        f"  ✓ Created {len(alerts)} alerts"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "============================================\n"
        "       AROGYAFLOW AI DATA SEEDER\n"
        "============================================\n"
    )

    print(
        "🔥 Connected to Firebase Firestore"
    )

    # PHCs
    phc_ids = seed_phcs()

    # Medicines
    seed_medicines(phc_ids)

    # Demand history
    seed_demand_history(phc_ids)

    # Alerts
    seed_alerts(phc_ids)

    print(
        "\n"
        "============================================\n"
        "       ✅ DATA SEEDING COMPLETE\n"
        "============================================\n"
    )

    print(
        f"\n🏥 PHCs: {len(phc_ids)}"
    )

    print(
        f"💊 Medicines per PHC: {len(MEDICINES)}"
    )

    print(
        f"📅 Historical days: {HISTORY_DAYS}"
    )

    print(
        "\n🔥 ArogyaFlow Firestore is ready!\n"
    )


if __name__ == "__main__":
    main()