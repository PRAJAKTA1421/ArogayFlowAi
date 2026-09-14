import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PHASE B2 - MEDICINE INVENTORY BASELINE GENERATOR
# ============================================================

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
PHC_FILE = BASE_DIR / "temporal" / "phc_master.csv"
OUTPUT_DIR = BASE_DIR / "temporal"
OUTPUT_FILE = OUTPUT_DIR / "medicine_inventory_baseline.csv"

print("=" * 70)
print("PHASE B2 - MEDICINE INVENTORY BASELINE GENERATOR")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD PHC MASTER DATASET
# ------------------------------------------------------------

if not PHC_FILE.exists():
    raise FileNotFoundError(
        f"PHC master file not found: {PHC_FILE}"
    )

phc_df = pd.read_csv(PHC_FILE)

print("\n✓ Loaded PHC master dataset")
print(f"  Shape: {phc_df.shape}")
print(f"  Unique PHCs: {phc_df['PHC_ID'].nunique()}")

# ------------------------------------------------------------
# 2. DEFINE ESSENTIAL MEDICINES
# ------------------------------------------------------------

medicines = [
    {
        "Medicine_ID": "MED-001",
        "Medicine_Name": "Paracetamol 500mg",
        "Category": "Analgesic",
        "Base_Daily_Demand": 35,
        "Min_Stock_Days": 7,
        "Max_Stock_Days": 30
    },
    {
        "Medicine_ID": "MED-002",
        "Medicine_Name": "Amoxicillin 500mg",
        "Category": "Antibiotic",
        "Base_Daily_Demand": 18,
        "Min_Stock_Days": 7,
        "Max_Stock_Days": 30
    },
    {
        "Medicine_ID": "MED-003",
        "Medicine_Name": "ORS Sachet",
        "Category": "Rehydration",
        "Base_Daily_Demand": 22,
        "Min_Stock_Days": 10,
        "Max_Stock_Days": 35
    },
    {
        "Medicine_ID": "MED-004",
        "Medicine_Name": "Metformin 500mg",
        "Category": "Diabetes",
        "Base_Daily_Demand": 14,
        "Min_Stock_Days": 15,
        "Max_Stock_Days": 45
    },
    {
        "Medicine_ID": "MED-005",
        "Medicine_Name": "Amlodipine 5mg",
        "Category": "Hypertension",
        "Base_Daily_Demand": 12,
        "Min_Stock_Days": 15,
        "Max_Stock_Days": 45
    },
    {
        "Medicine_ID": "MED-006",
        "Medicine_Name": "Cetirizine 10mg",
        "Category": "Antihistamine",
        "Base_Daily_Demand": 16,
        "Min_Stock_Days": 7,
        "Max_Stock_Days": 30
    },
    {
        "Medicine_ID": "MED-007",
        "Medicine_Name": "Iron Folic Acid Tablet",
        "Category": "Supplement",
        "Base_Daily_Demand": 20,
        "Min_Stock_Days": 15,
        "Max_Stock_Days": 45
    },
    {
        "Medicine_ID": "MED-008",
        "Medicine_Name": "Omeprazole 20mg",
        "Category": "Gastrointestinal",
        "Base_Daily_Demand": 10,
        "Min_Stock_Days": 10,
        "Max_Stock_Days": 30
    }
]

medicine_df = pd.DataFrame(medicines)

print(f"\n✓ Defined medicines: {len(medicine_df)}")
print(medicine_df[
    ["Medicine_ID", "Medicine_Name", "Category"]
].to_string(index=False))

# ------------------------------------------------------------
# 3. CREATE PHC × MEDICINE COMBINATIONS
# ------------------------------------------------------------

phc_df["_key"] = 1
medicine_df["_key"] = 1

inventory_df = phc_df.merge(
    medicine_df,
    on="_key",
    how="inner"
).drop(columns="_key")

print("\n✓ Created PHC × Medicine combinations")
print(f"  Total inventory records: {len(inventory_df)}")

# ------------------------------------------------------------
# 4. CALCULATE DAILY DEMAND
# ------------------------------------------------------------

# Population factor represents variation between PHCs
demand_variation = np.random.uniform(0.80, 1.20, len(inventory_df))

inventory_df["Daily_Demand"] = (
    inventory_df["Base_Daily_Demand"]
    * inventory_df["Population_Factor"]
    * demand_variation
).round().clip(lower=1).astype(int)

# ------------------------------------------------------------
# 5. GENERATE INITIAL STOCK
# ------------------------------------------------------------

stock_days = np.random.uniform(
    inventory_df["Min_Stock_Days"],
    inventory_df["Max_Stock_Days"]
)

inventory_df["Stock_Days"] = stock_days.round(1)

inventory_df["Current_Stock"] = (
    inventory_df["Daily_Demand"]
    * inventory_df["Stock_Days"]
).round().astype(int)

# ------------------------------------------------------------
# 6. DEFINE REORDER LEVEL
# ------------------------------------------------------------

inventory_df["Reorder_Level"] = (
    inventory_df["Daily_Demand"]
    * inventory_df["Min_Stock_Days"]
).round().astype(int)

# ------------------------------------------------------------
# 7. ADD STOCK STATUS
# ------------------------------------------------------------

inventory_df["Stock_Status"] = np.select(
    [
        inventory_df["Current_Stock"] <= inventory_df["Reorder_Level"],
        inventory_df["Stock_Days"] <= 14
    ],
    [
        "LOW",
        "MEDIUM"
    ],
    default="HEALTHY"
)

# ------------------------------------------------------------
# 8. ADD EXPIRY INFORMATION
# ------------------------------------------------------------

inventory_df["Expiry_Days_Remaining"] = np.random.randint(
    30,
    730,
    len(inventory_df)
)

inventory_df["Expiry_Risk"] = np.select(
    [
        inventory_df["Expiry_Days_Remaining"] <= 60,
        inventory_df["Expiry_Days_Remaining"] <= 120
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)

# ------------------------------------------------------------
# 9. SELECT FINAL COLUMNS
# ------------------------------------------------------------

final_columns = [
    "PHC_ID",
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Medicine_Name",
    "Category",
    "Daily_Demand",
    "Stock_Days",
    "Current_Stock",
    "Reorder_Level",
    "Stock_Status",
    "Expiry_Days_Remaining",
    "Expiry_Risk"
]

inventory_df = inventory_df[final_columns]

# ------------------------------------------------------------
# 10. VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE B2 VALIDATION")
print("=" * 70)

print(f"\nInventory shape: {inventory_df.shape}")
print(f"Unique PHCs: {inventory_df['PHC_ID'].nunique()}")
print(f"Unique medicines: {inventory_df['Medicine_ID'].nunique()}")

print("\nExpected records:")
print(f"{len(phc_df) * len(medicine_df):,}")

print("\nActual records:")
print(f"{len(inventory_df):,}")

print("\nStock status distribution:")
print(inventory_df["Stock_Status"].value_counts())

print("\nExpiry risk distribution:")
print(inventory_df["Expiry_Risk"].value_counts())

print("\nMissing values:")
print(inventory_df.isna().sum())

print("\nSample records:")
print(
    inventory_df.head(10).to_string(index=False)
)

# ------------------------------------------------------------
# 11. SAVE DATASET
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inventory_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("✓ PHASE B2 COMPLETED")
print("=" * 70)

print(f"\nOutput saved to:\n{OUTPUT_FILE}")