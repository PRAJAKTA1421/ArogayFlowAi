import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PHASE B3 - TEMPORAL HEALTHCARE DATASET GENERATOR
# ============================================================

SEED = 42
DAYS = 180

BASE_DIR = Path(__file__).resolve().parent
PHC_FILE = BASE_DIR / "temporal" / "phc_master.csv"
INVENTORY_FILE = BASE_DIR / "temporal" / "medicine_inventory_baseline.csv"
OUTPUT_DIR = BASE_DIR / "temporal"


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate ArogyaFlow AI temporal healthcare dataset"
)

parser.add_argument(
    "--dev",
    action="store_true",
    help="Generate a 500-PHC development dataset"
)

args = parser.parse_args()

OUTPUT_FILE = (
    OUTPUT_DIR / "temporal_healthcare_data_dev.csv"
    if args.dev
    else OUTPUT_DIR / "temporal_healthcare_data.csv"
)

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

print("=" * 70)
print("PHASE B3 - TEMPORAL HEALTHCARE DATASET GENERATOR")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD PHC MASTER
# ------------------------------------------------------------

if not PHC_FILE.exists():
    raise FileNotFoundError(f"PHC master not found: {PHC_FILE}")

phc_df = pd.read_csv(PHC_FILE)

print("\n✓ Loaded PHC master")
print(f"  PHC records: {len(phc_df):,}")
print(f"  Unique PHCs: {phc_df['PHC_ID'].nunique():,}")


# ------------------------------------------------------------
# 2. LOAD INVENTORY BASELINE
# ------------------------------------------------------------

if not INVENTORY_FILE.exists():
    raise FileNotFoundError(
        f"Inventory baseline not found: {INVENTORY_FILE}"
    )

inventory_df = pd.read_csv(INVENTORY_FILE)

print("\n✓ Loaded medicine inventory baseline")
print(f"  Inventory records: {len(inventory_df):,}")
print(
    f"  Unique medicines: "
    f"{inventory_df['Medicine_ID'].nunique():,}"
)


# ------------------------------------------------------------
# 3. DEVELOPMENT MODE
# ------------------------------------------------------------

if args.dev:

    dev_phc_count = min(500, len(phc_df))

    selected_phcs = (
        phc_df[
            [
                "PHC_ID",
                "State",
                "PHC_Type",
                "Urban_Rural",
                "Population_Factor",
                "Capacity_Factor",
            ]
        ]
        .sample(
            n=dev_phc_count,
            random_state=SEED
        )
        .copy()
    )

    phc_df = selected_phcs

    print("\n⚙ DEVELOPMENT MODE ENABLED")
    print(f"  PHCs selected: {len(phc_df):,}")
    print(f"  Days: {DAYS}")
    print("  Medicines: 8")

else:

    phc_df = phc_df[
        [
            "PHC_ID",
            "State",
            "PHC_Type",
            "Urban_Rural",
            "Population_Factor",
            "Capacity_Factor",
        ]
    ].copy()

    print("\n⚙ FULL INDIA MODE")
    print(f"  PHCs: {len(phc_df):,}")
    print(f"  Days: {DAYS}")
    print("  Medicines: 8")


# ------------------------------------------------------------
# 4. MEDICINE MASTER
# ------------------------------------------------------------

medicine_columns = [
    "Medicine_ID",
    "Medicine_Name",
    "Category",
    "Daily_Demand",
    "Current_Stock",
]

medicine_df = (
    inventory_df[medicine_columns]
    .drop_duplicates(
        subset=["Medicine_ID"]
    )
    .copy()
)


medicine_df = medicine_df.rename(
    columns={
        "Daily_Demand": "Baseline_Demand",
        "Current_Stock": "Baseline_Stock",
    }
)

# Fast medicine lookup
medicine_lookup = (
    inventory_df[
        [
            "Medicine_ID",
            "Medicine_Name",
            "Category"
        ]
    ]
    .drop_duplicates(
        subset=["Medicine_ID"]
    )
    .set_index("Medicine_ID")
    .to_dict("index")
)

print("\n✓ Medicine master created")


# ------------------------------------------------------------
# 5. CREATE PHC × MEDICINE BASELINE
# ------------------------------------------------------------

baseline = phc_df.merge(
    inventory_df[
        [
            "PHC_ID",
            "Medicine_ID",
            "Daily_Demand",
            "Current_Stock",
        ]
    ],
    on="PHC_ID",
    how="inner",
)

baseline = baseline.rename(
    columns={
        "Daily_Demand": "Initial_Daily_Demand",
        "Current_Stock": "Initial_Stock",
    }
)

print("\n✓ Created PHC × Medicine baseline")
print(f"  Records: {len(baseline):,}")


# ------------------------------------------------------------
# 6. MEDICINE EFFECTS
# ------------------------------------------------------------

medicine_effects = {
    "MED-001": 1.00,  # Paracetamol
    "MED-002": 0.75,  # Amoxicillin
    "MED-003": 1.10,  # ORS
    "MED-004": 0.60,  # Metformin
    "MED-005": 0.55,  # Amlodipine
    "MED-006": 0.70,  # Cetirizine
    "MED-007": 0.90,  # Iron Folic Acid
    "MED-008": 0.50,  # Omeprazole
}

baseline["Medicine_Factor"] = (
    baseline["Medicine_ID"]
    .map(medicine_effects)
    .fillna(1.0)
)


# ------------------------------------------------------------
# 7. DATE RANGE
# ------------------------------------------------------------

dates = pd.date_range(
    start="2026-01-01",
    periods=DAYS,
    freq="D"
)

print(
    f"\n✓ Temporal period: "
    f"{dates.min().date()} → {dates.max().date()}"
)


# ------------------------------------------------------------
# 8. GENERATE DATA IN DAILY CHUNKS
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

header_written = False

total_rows = 0

# Initial inventory state
inventory_state = {
    row.PHC_ID + "||" + row.Medicine_ID: float(row.Initial_Stock)
    for row in baseline.itertuples(index=False)
}


# ------------------------------------------------------------
# 9. STATE-LEVEL OUTBREAK ASSIGNMENT
# ------------------------------------------------------------

states = baseline["State"].dropna().unique()

outbreak_states = rng.choice(
    states,
    size=max(1, int(len(states) * 0.10)),
    replace=False
)

outbreak_states = set(outbreak_states)

print(
    f"\n✓ Outbreak states selected: "
    f"{len(outbreak_states)}"
)


# ------------------------------------------------------------
# 10. DAILY SIMULATION
# ------------------------------------------------------------

for day_index, current_date in enumerate(dates):

    day_records = []

    day_of_week = current_date.dayofweek

    # Weekend effect
    if day_of_week >= 5:
        weekday_factor = 0.88
    else:
        weekday_factor = 1.00

    # Slow seasonal component
    seasonal_factor = (
        1.0
        + 0.12
        * np.sin(
            2
            * np.pi
            * day_index
            / 180
        )
    )

    # --------------------------------------------------------
    # PHC × MEDICINE SIMULATION
    # --------------------------------------------------------

    for row in baseline.itertuples(index=False):

        key = (
            row.PHC_ID
            + "||"
            + row.Medicine_ID
        )

        opening_stock = max(
            0,
            inventory_state[key]
        )

        # --------------------------------------------
        # PATIENT FOOTFALL
        # --------------------------------------------

        base_footfall = (
            35
            * row.Population_Factor
            * row.Capacity_Factor
        )

        footfall_noise = rng.uniform(
            0.85,
            1.15
        )

        patient_footfall = max(
            1,
            int(
                base_footfall
                * weekday_factor
                * seasonal_factor
                * footfall_noise
            )
        )

        # --------------------------------------------
        # OUTBREAK
        # --------------------------------------------

        outbreak_flag = 0

        outbreak_factor = 1.0

        # Certain states receive outbreak events
        # during different windows.
        if (
            row.State in outbreak_states
            and 45 <= day_index <= 65
            and row.Medicine_ID
            in ["MED-001", "MED-002", "MED-003", "MED-006"]
        ):

            outbreak_flag = 1

            outbreak_factor = rng.uniform(
                1.35,
                1.85
            )

        # --------------------------------------------
        # DEMAND
        # --------------------------------------------

        demand_noise = rng.uniform(
            0.85,
            1.15
        )

        demand = (
            row.Initial_Daily_Demand
            * (
                0.90
                + 0.20 * row.Population_Factor
            )
            * weekday_factor
            * seasonal_factor
            * outbreak_factor
            * demand_noise
        )

        # Footfall relationship
        footfall_multiplier = (
            patient_footfall
            / max(
                1,
                base_footfall
            )
        )

        demand *= (
            0.80
            + 0.20 * footfall_multiplier
        )

        demand = max(
            1,
            int(round(demand))
        )

        # --------------------------------------------
        # ANOMALY
        # --------------------------------------------

        anomaly_flag = 0

        # About 1% anomalous observations
        if rng.random() < 0.01:

            anomaly_flag = 1

            anomaly_multiplier = rng.choice(
                [
                    rng.uniform(1.8, 2.8),
                    rng.uniform(0.25, 0.50)
                ]
            )

            demand = max(
                1,
                int(
                    round(
                        demand
                        * anomaly_multiplier
                    )
                )
            )

        # --------------------------------------------------------
        # REPLENISHMENT
        # --------------------------------------------------------

        received_stock = 0
        supply_disruption_flag = 0

        # Regular replenishment every 14 days
        if day_index > 0 and day_index % 14 == 0:

            target_stock = (
                row.Initial_Daily_Demand
                * 21
            )

            received_stock = max(
                0,
                int(
                    round(
                        target_stock
                        - opening_stock
                    )
                )
            )

            # Occasionally create delayed supply
            if rng.random() < 0.08:

                supply_disruption_flag = 1

                received_stock = int(
                    received_stock * 0.40
            )

        # --------------------------------------------
        # STOCK CONSUMPTION
        # --------------------------------------------

        available_stock = (
            opening_stock
            + received_stock
        )

        issued_stock = min(
            available_stock,
            demand
        )

        closing_stock = max(
            0,
            available_stock
            - issued_stock
        )

        stockout_flag = int(
            closing_stock == 0
            and demand > 0
        )

        # --------------------------------------------
        # DAYS OF STOCK REMAINING
        # --------------------------------------------

        days_of_stock = (
            closing_stock
            / max(
                demand,
                1
            )
        )

        # --------------------------------------------
        # RISK
        # --------------------------------------------

        if stockout_flag:
            risk_level = "CRITICAL"
        elif days_of_stock <= 2:
            risk_level = "HIGH"
        elif days_of_stock <= 7:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # --------------------------------------------
        # SAVE RECORD
        # --------------------------------------------

        day_records.append(
            {
                "Date": current_date.date(),

                "PHC_ID": row.PHC_ID,

                "State": row.State,

                "PHC_Type": row.PHC_Type,

                "Urban_Rural": row.Urban_Rural,

                "Medicine_ID": row.Medicine_ID,

                "Medicine_Name": medicine_lookup[
                    row.Medicine_ID
                ]["Medicine_Name"],

                "Category": medicine_lookup[
                    row.Medicine_ID
                ]["Category"],

                "Patient_Footfall": patient_footfall,

                "Demand": demand,

                "Opening_Stock": int(
                    round(opening_stock)
                ),

                "Received_Stock": received_stock,

                "Issued_Stock": int(
                    issued_stock
                ),

                "Closing_Stock": int(
                    round(closing_stock)
                ),

                "Days_of_Stock": round(
                    days_of_stock,
                    2
                ),

                "Stockout_Flag": stockout_flag,

                "Risk_Level": risk_level,

                "Outbreak_Flag": outbreak_flag,

                "Anomaly_Flag": anomaly_flag,

                "Supply_Disruption_Flag": supply_disruption_flag,

                "Seasonal_Factor": round(
                        seasonal_factor,
                        4
                ),

                "Population_Factor": row.Population_Factor,

                "Capacity_Factor": row.Capacity_Factor,
            }
        )

        inventory_state[key] = closing_stock

    # --------------------------------------------------------
    # WRITE DAILY CHUNK
    # --------------------------------------------------------

    daily_df = pd.DataFrame(day_records)

    daily_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=not header_written
    )

    header_written = True

    total_rows += len(daily_df)

    if (
        day_index == 0
        or (day_index + 1) % 10 == 0
        or day_index == DAYS - 1
    ):
        print(
            f"  Day {day_index + 1:3d}/{DAYS} "
            f"| {current_date.date()} "
            f"| rows written: {total_rows:,}"
        )


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE B3 VALIDATION")
print("=" * 70)

print(f"\nOutput file: {OUTPUT_FILE}")

# Read dataset back for validation
df = pd.read_csv(OUTPUT_FILE)

print(f"\nDataset shape: {df.shape}")

print(
    f"Unique PHCs: "
    f"{df['PHC_ID'].nunique():,}"
)

print(
    f"Unique medicines: "
    f"{df['Medicine_ID'].nunique():,}"
)

print(
    f"Unique dates: "
    f"{df['Date'].nunique():,}"
)

expected_days = DAYS

print(
    f"Expected days: "
    f"{expected_days}"
)

# ------------------------------------------------------------
# DUPLICATE CHECK
# ------------------------------------------------------------

duplicates = df.duplicated(
    subset=[
        "Date",
        "PHC_ID",
        "Medicine_ID"
    ]
).sum()

print(
    f"\nDuplicate "
    f"(Date + PHC + Medicine): {duplicates}"
)

# ------------------------------------------------------------
# NEGATIVE VALUE CHECK
# ------------------------------------------------------------

negative_demand = (
    df["Demand"] < 0
).sum()

negative_stock = (
    df[
        [
            "Opening_Stock",
            "Received_Stock",
            "Issued_Stock",
            "Closing_Stock"
        ]
    ]
    < 0
).sum().sum()

print(
    f"Negative demand values: "
    f"{negative_demand}"
)

print(
    f"Negative stock values: "
    f"{negative_stock}"
)

# ------------------------------------------------------------
# STOCKOUT
# ------------------------------------------------------------

stockout_count = df[
    "Stockout_Flag"
].sum()

print(
    f"\nStockout records: "
    f"{stockout_count:,}"
)

print(
    f"Stockout rate: "
    f"{stockout_count / len(df) * 100:.2f}%"
)

# ------------------------------------------------------------
# OUTBREAK
# ------------------------------------------------------------

outbreak_count = df[
    "Outbreak_Flag"
].sum()

print(
    f"Outbreak records: "
    f"{outbreak_count:,}"
)

# ------------------------------------------------------------
# ANOMALIES
# ------------------------------------------------------------

anomaly_count = df[
    "Anomaly_Flag"
].sum()

print(
    f"Anomaly records: "
    f"{anomaly_count:,}"
)

# ------------------------------------------------------------
# RISK DISTRIBUTION
# ------------------------------------------------------------

print("\nRisk distribution:")

print(
    df["Risk_Level"]
    .value_counts()
)

# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")

missing = df.isna().sum()

print(
    missing[
        missing > 0
    ]
)

# ------------------------------------------------------------
# DEMAND STATISTICS
# ------------------------------------------------------------

print("\nDemand statistics:")

print(
    df["Demand"].describe()
)

# ------------------------------------------------------------
# SAMPLE
# ------------------------------------------------------------

print("\nFirst 10 records:")

print(
    df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)

if (
    duplicates == 0
    and negative_demand == 0
    and negative_stock == 0
    and stockout_count > 0
    and outbreak_count > 0
    and anomaly_count > 0
):

    print("✓ PHASE B3 VALIDATION PASSED")

else:

    print("⚠ PHASE B3 VALIDATION REQUIRES REVIEW")

print("=" * 70)

print(
    f"\nTemporal dataset saved to:\n"
    f"{OUTPUT_FILE}"
)