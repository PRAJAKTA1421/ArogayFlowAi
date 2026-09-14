import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data_pipeline/temporal/temporal_healthcare_data_features_dev.csv"
)

OUTPUT_DIR = Path("data_pipeline/temporal")

DEMAND_OUTPUT = OUTPUT_DIR / "ml_demand_forecasting_dev.csv"
STOCKOUT_OUTPUT = OUTPUT_DIR / "ml_stockout_prediction_dev.csv"
ANOMALY_OUTPUT = OUTPUT_DIR / "ml_anomaly_detection_dev.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_dataset(df, name):
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    numeric_df = df.select_dtypes(include=[np.number])

    infinite_values = np.isinf(numeric_df.to_numpy()).sum()

    print(f"Infinite values: {infinite_values}")

    if df.isna().sum().sum() != 0:
        raise ValueError(f"{name}: Missing values detected!")

    if df.duplicated().sum() != 0:
        raise ValueError(f"{name}: Duplicate rows detected!")

    if infinite_values != 0:
        raise ValueError(f"{name}: Infinite values detected!")

    print("")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("B3.2 - LEAKAGE-FREE ML DATASET PREPARATION")
print("=" * 70)

print("\nLoading B3.1 dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Input shape: {df.shape}")


# ============================================================
# SORT TEMPORAL DATA
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["PHC_ID", "Medicine_ID", "Date"]
).reset_index(drop=True)


# ============================================================
# COMMON DATE FEATURES
# ============================================================

print("\nCreating common ML features...")

df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Day_of_Month"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)


# ============================================================
# COMMON IDENTIFIERS
# ============================================================

# Keep categorical information in the datasets.
# Encoding will be handled consistently during model training.

common_context_features = [
    "PHC_ID",
    "State",
    "PHC_Type",
    "Urban_Rural",
    "Medicine_ID",
    "Medicine_Name",
    "Category",
]

date_features = [
    "Day_of_Week",
    "Day_of_Month",
    "Month",
    "Week_of_Year",
]


# ============================================================
# B3.2-A
# DEMAND FORECASTING DATASET
# ============================================================

print("Preparing demand forecasting dataset...")

demand_df = df.copy()

# ------------------------------------------------------------
# TARGET = NEXT DAY DEMAND
# ------------------------------------------------------------

demand_df["Target_Demand_Next_Day"] = (
    demand_df
    .groupby(["PHC_ID", "Medicine_ID"])["Demand"]
    .shift(-1)
)

# Last day of each PHC + medicine group has no next-day target.
demand_df = demand_df.dropna(
    subset=["Target_Demand_Next_Day"]
).copy()

demand_df["Target_Demand_Next_Day"] = (
    demand_df["Target_Demand_Next_Day"].astype(float)
)


# ------------------------------------------------------------
# DEMAND FEATURES
# ------------------------------------------------------------

demand_features = common_context_features + date_features + [
    "Patient_Footfall",
    "Population_Factor",
    "Capacity_Factor",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Outbreak_Flag",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
]

demand_columns = demand_features + [
    "Date",
    "Target_Demand_Next_Day",
]

demand_ml = demand_df[demand_columns].copy()


# ============================================================
# B3.2-B
# STOCKOUT PREDICTION DATASET
# ============================================================

print("Preparing stockout prediction dataset...")

stockout_df = df.copy()


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

stockout_target = "Stockout_Next_7_Days"


# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

# These are intentionally excluded:
#
# Closing_Stock       -> future/current outcome leakage
# Issued_Stock        -> derived from actual demand/stock movement
# Stockout_Flag       -> direct outcome
# Risk_Level          -> directly derived from stockout state
# Anomaly_Flag        -> synthetic ground-truth anomaly label
# Demand_Spike_Flag   -> derived signal closely tied to anomaly
#
# Current Demand is also excluded so that the model represents
# an early-warning prediction using historical information.

stockout_features = common_context_features + date_features + [
    "Patient_Footfall",
    "Population_Factor",
    "Capacity_Factor",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Opening_Stock",
    "Received_Stock",
    "Days_of_Stock",
    "Outbreak_Flag",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
]

stockout_columns = stockout_features + [
    "Date",
    stockout_target,
]

stockout_ml = stockout_df[stockout_columns].copy()


# ============================================================
# B3.2-C
# ANOMALY DETECTION DATASET
# ============================================================

print("Preparing anomaly detection dataset...")

anomaly_df = df.copy()


# ------------------------------------------------------------
# IMPORTANT:
# DO NOT USE Anomaly_Flag AS A MODEL FEATURE.
#
# It is the synthetic ground-truth label that we will use
# later to evaluate Isolation Forest.
# ------------------------------------------------------------

anomaly_features = common_context_features + date_features + [
    "Patient_Footfall",
    "Population_Factor",
    "Capacity_Factor",
    "Demand",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Rolling_7_Demand",
    "Demand_Change",
    "Opening_Stock",
    "Days_of_Stock",
    "Outbreak_Flag",
    "Supply_Disruption_Flag",
    "Seasonal_Factor",
]

anomaly_columns = anomaly_features + [
    "Date",
    "Anomaly_Flag",
]

anomaly_ml = anomaly_df[anomaly_columns].copy()


# ============================================================
# NUMERIC CLEANING
# ============================================================

print("Cleaning numeric values...")

for dataset in [demand_ml, stockout_ml, anomaly_ml]:

    numeric_columns = dataset.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        dataset[column] = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )

    # Remove rows with invalid numeric values
    dataset.dropna(
        subset=numeric_columns,
        inplace=True
    )

    # Replace infinite values if any
    dataset.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    dataset.dropna(
        subset=numeric_columns,
        inplace=True
    )


# ============================================================
# SAVE DATASETS
# ============================================================

print("\nSaving ML datasets...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

demand_ml.to_csv(
    DEMAND_OUTPUT,
    index=False
)

stockout_ml.to_csv(
    STOCKOUT_OUTPUT,
    index=False
)

anomaly_ml.to_csv(
    ANOMALY_OUTPUT,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

validate_dataset(
    demand_ml,
    "DEMAND FORECASTING DATASET"
)

print(
    "Target: Target_Demand_Next_Day"
)

print(
    demand_ml["Target_Demand_Next_Day"].describe()
)

print("")


validate_dataset(
    stockout_ml,
    "STOCKOUT PREDICTION DATASET"
)

print(
    "Target: Stockout_Next_7_Days"
)

print(
    stockout_ml["Stockout_Next_7_Days"].value_counts()
)

positive_rate = (
    stockout_ml["Stockout_Next_7_Days"].mean() * 100
)

print(
    f"\nPositive stockout rate: {positive_rate:.2f}%"
)

print("")


validate_dataset(
    anomaly_ml,
    "ANOMALY DETECTION DATASET"
)

print(
    "Ground-truth Anomaly_Flag distribution:"
)

print(
    anomaly_ml["Anomaly_Flag"].value_counts()
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)

# Demand
assert "Demand" not in demand_features
assert "Target_Demand_Next_Day" not in demand_features

# Stockout
assert "Stockout_Next_7_Days" not in stockout_features
assert "Anomaly_Flag" not in stockout_features
assert "Stockout_Flag" not in stockout_features

# Anomaly
assert "Anomaly_Flag" not in anomaly_features

print("✓ Demand target leakage check passed")
print("✓ Stockout target leakage check passed")
print("✓ Anomaly label leakage check passed")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✓ B3.2 VALIDATION PASSED")
print("=" * 70)

print("\nFiles created:")

print(f"1. {DEMAND_OUTPUT}")
print(f"2. {STOCKOUT_OUTPUT}")
print(f"3. {ANOMALY_OUTPUT}")

print("\nDataset sizes:")
print(f"Demand   : {len(demand_ml):,}")
print(f"Stockout : {len(stockout_ml):,}")
print(f"Anomaly  : {len(anomaly_ml):,}")

print("\nReady for B4 - Model Training.")