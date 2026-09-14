import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data_pipeline/processed/india_healthcare_context.csv"
OUTPUT_DIR = "data_pipeline/temporal"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phc_master.csv")

# Reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# LOAD PHASE A DATA
# ============================================================

print("=" * 70)
print("PHASE B1 - PHC MASTER DATASET GENERATOR")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Phase A dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\n✓ Loaded Phase A dataset")
print(f"  Shape: {df.shape}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "State",
    "PHC_Count",
    "PHC_Public",
    "PHC_Private",
    "PHC_Urban",
    "PHC_Rural",
    "Doctors_Required",
    "Doctors_In_Position",
    "AYUSH_Hospitals",
    "AYUSH_Beds"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("✓ Required columns validated")


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "PHC_Count",
    "PHC_Public",
    "PHC_Private",
    "PHC_Urban",
    "PHC_Rural",
    "Doctors_Required",
    "Doctors_In_Position",
    "AYUSH_Hospitals",
    "AYUSH_Beds"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# REMOVE INVALID STATES
# ============================================================

invalid_states = [
    "All India",
    "India",
    "Total"
]

df = df[
    ~df["State"].astype(str).str.strip().isin(invalid_states)
].copy()

df = df[df["PHC_Count"].notna()].copy()

df["PHC_Count"] = df["PHC_Count"].astype(int)

print(f"✓ States/UTs with usable PHC count: {len(df)}")

all_states = set(
    pd.read_csv(INPUT_FILE)["State"]
    .astype(str)
    .str.strip()
)

included_states = set(df["State"].astype(str).str.strip())

excluded_states = sorted(
    all_states - included_states
)

if excluded_states:
    print(
        f"  Excluded because PHC_Count is unavailable/aggregate: "
        f"{excluded_states}"
    )


# ============================================================
# CREATE PHC RECORDS
# ============================================================

records = []

for _, row in df.iterrows():

    state = row["State"]

    total_phcs = int(row["PHC_Count"])

    public_phcs = (
        int(row["PHC_Public"])
        if pd.notna(row["PHC_Public"])
        else total_phcs
    )

    urban_phcs = (
        int(row["PHC_Urban"])
        if pd.notna(row["PHC_Urban"])
        else 0
    )

    rural_phcs = (
        int(row["PHC_Rural"])
        if pd.notna(row["PHC_Rural"])
        else 0
    )

    # --------------------------------------------------------
    # PHC DISTRIBUTION
    # --------------------------------------------------------

    # Use reported urban/rural values where available.
    # If they don't add up exactly to total PHCs,
    # distribute the remainder proportionally.
    reported_total = urban_phcs + rural_phcs

    if reported_total > 0:

        urban_ratio = urban_phcs / reported_total

        estimated_urban = round(total_phcs * urban_ratio)

    else:

        estimated_urban = round(total_phcs * 0.20)

    estimated_urban = max(
        0,
        min(total_phcs, estimated_urban)
    )

    estimated_rural = total_phcs - estimated_urban


    # --------------------------------------------------------
    # DOCTOR CAPACITY
    # --------------------------------------------------------

    doctors_required = row["Doctors_Required"]
    doctors_in_position = row["Doctors_In_Position"]

    if pd.notna(doctors_required) and doctors_required > 0:

        doctors_per_phc = (
            doctors_required / total_phcs
        )

    else:

        doctors_per_phc = 1.0


    if (
        pd.notna(doctors_in_position)
        and doctors_in_position >= 0
    ):

        doctor_position_per_phc = (
            doctors_in_position / total_phcs
        )

    else:

        doctor_position_per_phc = doctors_per_phc


    # --------------------------------------------------------
    # GENERATE INDIVIDUAL PHCs
    # --------------------------------------------------------

    for i in range(total_phcs):

        phc_number = i + 1

        state_code = (
            state.upper()
            .replace(" ", "")
            .replace("&", "")
            .replace("-", "")
        )[:6]

        # ----------------------------------------------------
        # GENERATE UNIQUE NATIONWIDE PHC ID
        # ----------------------------------------------------

        phc_id = f"PHC-{len(records) + 1:05d}"

        # ----------------------------------------------------
        # URBAN / RURAL
        # ----------------------------------------------------

        if i < estimated_urban:

            urban_rural = "Urban"

        else:

            urban_rural = "Rural"


        # ----------------------------------------------------
        # PHC TYPE
        # ----------------------------------------------------

        phc_type = "Public PHC"


        # ----------------------------------------------------
        # POPULATION FACTOR
        # ----------------------------------------------------

        # Synthetic modelling factor.
        # It represents relative demand intensity,
        # NOT an actual population measurement.

        if urban_rural == "Urban":

            population_factor = rng.uniform(
                1.10,
                1.60
            )

        else:

            population_factor = rng.uniform(
                0.70,
                1.30
            )


        # ----------------------------------------------------
        # CAPACITY FACTOR
        # ----------------------------------------------------

        # Based partly on doctor availability.
        # This is a modelling feature, not a measured PHC value.

        if doctors_per_phc > 0:

            capacity_factor = (
                doctor_position_per_phc /
                doctors_per_phc
            )

        else:

            capacity_factor = 1.0


        capacity_factor = np.clip(
            capacity_factor,
            0.50,
            1.50
        )


        # ----------------------------------------------------
        # STATE-LEVEL HEALTHCARE CONTEXT
        # ----------------------------------------------------

        ayush_hospitals = (
            row["AYUSH_Hospitals"]
            if pd.notna(row["AYUSH_Hospitals"])
            else np.nan
        )

        ayush_beds = (
            row["AYUSH_Beds"]
            if pd.notna(row["AYUSH_Beds"])
            else np.nan
        )


        # ----------------------------------------------------
        # STORE RECORD
        # ----------------------------------------------------

        records.append({

            "PHC_ID": phc_id,

            "State": state,

            "PHC_Type": phc_type,

            "Urban_Rural": urban_rural,

            # Synthetic modelling variables
            "Population_Factor": round(
                population_factor,
                4
            ),

            "Capacity_Factor": round(
                capacity_factor,
                4
            ),

            # State-level real/public-data context
            "State_PHC_Count": total_phcs,

            "State_PHC_Public": public_phcs,

            "State_PHC_Urban": urban_phcs,

            "State_PHC_Rural": rural_phcs,

            "Doctors_Required_State": doctors_required,

            "Doctors_In_Position_State": doctors_in_position,

            "AYUSH_Hospitals_State": ayush_hospitals,

            "AYUSH_Beds_State": ayush_beds
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

phc_master = pd.DataFrame(records)


# ============================================================
# SAVE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

phc_master.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PHASE B1 VALIDATION")
print("=" * 70)

print(f"\nPHC master shape: {phc_master.shape}")

print(
    f"Unique PHCs: "
    f"{phc_master['PHC_ID'].nunique()}"
)

print(
    f"Unique states/UTs: "
    f"{phc_master['State'].nunique()}"
)

print("\nUrban/Rural distribution:")

print(
    phc_master["Urban_Rural"]
    .value_counts()
    .to_string()
)

print("\nPHCs by state:")

print(
    phc_master
    .groupby("State")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .to_string()
)

print("\nMissing values:")

print(
    phc_master.isna()
    .sum()
    .to_string()
)

print("\nFirst 10 PHCs:")

print(
    phc_master.head(10)
    .to_string(index=False)
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("✓ PHASE B1 COMPLETED")
print("=" * 70)

print(
    f"\nOutput saved to:\n{OUTPUT_FILE}"
)