import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data_pipeline/temporal/temporal_healthcare_data_dev.csv"

OUTPUT_FILE = (
    "data_pipeline/temporal/"
    "temporal_healthcare_data_features_dev.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("B3.1 - TEMPORAL FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Input shape: {df.shape}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "Date",
    "PHC_ID",
    "Medicine_ID",
    "Demand",
    "Received_Stock",
    "Outbreak_Flag",
    "Anomaly_Flag",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Date"] = pd.to_datetime(df["Date"])


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting data...")

df = df.sort_values(
    ["PHC_ID", "Medicine_ID", "Date"]
).reset_index(drop=True)


# ============================================================
# GROUP BY PHC + MEDICINE
# ============================================================

group_cols = ["PHC_ID", "Medicine_ID"]

grouped = df.groupby(group_cols, sort=False)


# ============================================================
# 1. LAG FEATURES
# ============================================================

print("Creating lag features...")

# Demand from previous day
df["Lag_1_Demand"] = grouped["Demand"].shift(1)

# Demand from previous 7 days
df["Lag_7_Demand"] = grouped["Demand"].shift(7)


# ============================================================
# 2. ROLLING DEMAND
# ============================================================

print("Creating rolling demand feature...")

df["Rolling_7_Demand"] = (
    grouped["Demand"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=7,
            min_periods=3
        ).mean()
    )
)


# ============================================================
# 3. DEMAND CHANGE
# ============================================================

print("Creating demand change feature...")

df["Demand_Change"] = (
    df["Demand"] - df["Lag_1_Demand"]
)


# ============================================================
# 6. DEMAND SPIKE FLAG
# ============================================================

print("Creating demand spike flag...")

# A demand spike is detected when current demand is
# significantly higher than the recent rolling average.

df["Demand_Spike_Flag"] = (
    (
        df["Rolling_7_Demand"].notna()
        & (
            df["Demand"]
            > df["Rolling_7_Demand"] * 1.50
        )
    )
    | (df["Outbreak_Flag"] == 1)
    | (df["Anomaly_Flag"] == 1)
).astype(int)


# ============================================================
# 7. FUTURE STOCKOUT TARGET
# ============================================================

print("Creating future stockout target...")

# IMPORTANT:
# We don't want the model to use today's Stockout_Flag
# to predict today's stockout.
#
# Instead, this target represents whether a stockout occurs
# during the NEXT 7 DAYS.

df["Stockout_Next_7_Days"] = (
    grouped["Stockout_Flag"]
    .transform(
        lambda x: (
            x.shift(-1)
            .rolling(
                window=7,
                min_periods=1
            )
            .max()
        )
    )
)

df["Stockout_Next_7_Days"] = (
    df["Stockout_Next_7_Days"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 8. REMOVE ROWS WITHOUT SUFFICIENT HISTORY
# ============================================================

print("\nRemoving rows without sufficient historical features...")

before = len(df)

df = df[
    df["Lag_1_Demand"].notna()
    & df["Lag_7_Demand"].notna()
    & df["Rolling_7_Demand"].notna()
].copy()

after = len(df)

print(
    f"Removed {before - after:,} rows "
    f"without sufficient history."
)


# ============================================================
# 9. HANDLE REMAINING NUMERIC MISSING VALUES
# ============================================================

numeric_features = [
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Seasonal_Factor",
]

for col in numeric_features:
    df[col] = df[col].fillna(0)


# ============================================================
# 10. FINAL SORT
# ============================================================

df = df.sort_values(
    ["Date", "PHC_ID", "Medicine_ID"]
).reset_index(drop=True)


# ============================================================
# 11. SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

print("\nSaving engineered dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("B3.1 VALIDATION")
print("=" * 70)

print(f"\nFinal shape: {df.shape}")

print(f"Unique PHCs: {df['PHC_ID'].nunique():,}")
print(f"Unique medicines: {df['Medicine_ID'].nunique()}")
print(f"Unique dates: {df['Date'].nunique()}")

print("\nNew features:")

new_features = [
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
    "Demand_Spike_Flag",
    "Stockout_Next_7_Days",
]

for col in new_features:
    print(
        f"  {col:<25} "
        f"missing={df[col].isna().sum():,}"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = df.duplicated(
    subset=["Date", "PHC_ID", "Medicine_ID"]
).sum()

print(f"\nDuplicate records: {duplicates:,}")


# ============================================================
# STOCKOUT TARGET DISTRIBUTION
# ============================================================

print("\nStockout_Next_7_Days distribution:")

print(
    df["Stockout_Next_7_Days"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SPIKE DISTRIBUTION
# ============================================================

print("\nDemand_Spike_Flag distribution:")

print(
    df["Demand_Spike_Flag"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SUPPLY DISRUPTION DISTRIBUTION
# ============================================================

print("\nSupply_Disruption_Flag distribution:")

print(
    df["Supply_Disruption_Flag"]
    .value_counts()
    .sort_index()
)


# ============================================================
# DEMAND STATISTICS
# ============================================================

print("\nDemand statistics:")

print(
    df["Demand"].describe()
)


# ============================================================
# FINAL STATUS
# ============================================================

if (
    duplicates == 0
    and df[new_features].isna().sum().sum() == 0
):

    print("\n" + "=" * 70)
    print("✓ B3.1 VALIDATION PASSED")
    print("=" * 70)

    print(
        f"\nOutput saved to:\n{OUTPUT_FILE}"
    )

else:

    print("\n" + "=" * 70)
    print("⚠ B3.1 VALIDATION FAILED")
    print("=" * 70)