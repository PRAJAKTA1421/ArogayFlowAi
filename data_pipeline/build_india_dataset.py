import os
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)


PHC_STAFF_FILE = os.path.join(
    RAW_DIR,
    "allo-doc-PHCS_2017.csv"
)

PHC_STAFF_XLS_FILE = os.path.join(
    RAW_DIR,
    "datafile.xls"
)

AYUSH_FILE = os.path.join(
    RAW_DIR,
    "AYUSHHospitals.csv"
)

ALL_INDIA_FILE = os.path.join(
    RAW_DIR,
    "_All_India_DataUploadStatus.xls"
)


# ============================================================
# STATE NAME CLEANING
# ============================================================

def clean_state_name(name):
    """
    Normalize Indian State/UT names across all source datasets.

    Handles:
    - spaces
    - &, /, -, brackets
    - compressed names
    - old state names
    - duplicate naming formats
    - aggregate/header rows
    """

    if pd.isna(name):
        return None

    name = str(name).strip()

    # Remove footnotes and line breaks
    name = name.replace("*", "")
    name = name.replace("†", "")
    name = name.replace("\n", " ")
    name = " ".join(name.split())

    if not name:
        return None

    # --------------------------------------------------------
    # Create a comparison key
    # Removes spaces and punctuation so:
    #
    # "Andhra Pradesh"
    # "AndhraPradesh"
    # "Andhra-Pradesh"
    #
    # all become:
    # "andhrapradesh"
    # --------------------------------------------------------

    key = re.sub(r"[^a-z0-9]", "", name.lower())

    # --------------------------------------------------------
    # Invalid / aggregate / garbage rows
    # --------------------------------------------------------

    invalid_keys = {
        "",
        "nan",
        "none",
        "state",
        "stateut",
        "stateuts",
        "total",
        "allindia",
        "allindiatotal",
        "india",
        "allindiatotal",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    }

    # Handle -1 to -10 and similar garbage values
    if re.fullmatch(r"-?\d+", name):
        try:
            number = int(name)
            if -10 <= number <= 10:
                return None
        except ValueError:
            pass

    if key in invalid_keys:
        return None

    # --------------------------------------------------------
    # Canonical State / UT mapping
    # --------------------------------------------------------

    state_map = {

        # -------------------------
        # Andaman & Nicobar
        # -------------------------
        "anisland": "Andaman & Nicobar Islands",
        "anislands": "Andaman & Nicobar Islands",
        "andamanandnicobarislands": "Andaman & Nicobar Islands",
        "andamanandnicobar": "Andaman & Nicobar Islands",
        "andamannicobarislands": "Andaman & Nicobar Islands",

        # -------------------------
        # Andhra Pradesh
        # -------------------------
        "andhrapradesh": "Andhra Pradesh",

        # -------------------------
        # Arunachal Pradesh
        # -------------------------
        "arunachalpradesh": "Arunachal Pradesh",

        # -------------------------
        # Assam
        # -------------------------
        "assam": "Assam",

        # -------------------------
        # Bihar
        # -------------------------
        "bihar": "Bihar",

        # -------------------------
        # Chandigarh
        # -------------------------
        "chandigarh": "Chandigarh",

        # -------------------------
        # Chhattisgarh
        # -------------------------
        "chhattisgarh": "Chhattisgarh",

        # -------------------------
        # Dadra & Nagar Haveli and Daman & Diu
        # -------------------------
        "dadranagarhaveli":
            "Dadra & Nagar Haveli and Daman & Diu",

        "dadranagarhavelianddamananddiu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "dadranagarhavelianddamanndiu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "damananddiu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "damanndiu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "damandiu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "daman&diu":
            "Dadra & Nagar Haveli and Daman & Diu",

        # -------------------------
        # Delhi
        # -------------------------
        "delhi": "Delhi",
        "nctofdelhi": "Delhi",
        "delhinct": "Delhi",

        # -------------------------
        # Goa
        # -------------------------
        "goa": "Goa",

        # -------------------------
        # Gujarat
        # -------------------------
        "gujarat": "Gujarat",

        # -------------------------
        # Haryana
        # -------------------------
        "haryana": "Haryana",

        # -------------------------
        # Himachal Pradesh
        # -------------------------
        "himachalpradesh": "Himachal Pradesh",

        # -------------------------
        # Jammu & Kashmir
        # -------------------------
        "jammuandkashmir": "Jammu & Kashmir",
        "jammukashmir": "Jammu & Kashmir",

        # -------------------------
        # Jharkhand
        # -------------------------
        "jharkhand": "Jharkhand",

        # -------------------------
        # Karnataka
        # -------------------------
        "karnataka": "Karnataka",

        # -------------------------
        # Kerala
        # -------------------------
        "kerala": "Kerala",

        # -------------------------
        # Ladakh
        # -------------------------
        "ladakh": "Ladakh",

        # -------------------------
        # Lakshadweep
        # -------------------------
        "lakshadweep": "Lakshadweep",

        # -------------------------
        # Madhya Pradesh
        # -------------------------
        "madhyapradesh": "Madhya Pradesh",

        # -------------------------
        # Maharashtra
        # -------------------------
        "maharashtra": "Maharashtra",

        # -------------------------
        # Manipur
        # -------------------------
        "manipur": "Manipur",

        # -------------------------
        # Meghalaya
        # -------------------------
        "meghalaya": "Meghalaya",

        # -------------------------
        # Mizoram
        # -------------------------
        "mizoram": "Mizoram",

        # -------------------------
        # Nagaland
        # -------------------------
        "nagaland": "Nagaland",

        # -------------------------
        # Odisha
        # -------------------------
        "odisha": "Odisha",
        "orissa": "Odisha",

        # -------------------------
        # Puducherry
        # -------------------------
        "puducherry": "Puducherry",
        "pondicherry": "Puducherry",

        # -------------------------
        # Punjab
        # -------------------------
        "punjab": "Punjab",

        # -------------------------
        # Rajasthan
        # -------------------------
        "rajasthan": "Rajasthan",

        # -------------------------
        # Sikkim
        # -------------------------
        "sikkim": "Sikkim",

        # -------------------------
        # Tamil Nadu
        # -------------------------
        "tamilnadu": "Tamil Nadu",

        # -------------------------
        # Telangana
        # -------------------------
        "telangana": "Telangana",

        # -------------------------
        # Tripura
        # -------------------------
        "tripura": "Tripura",

        # -------------------------
        # Uttarakhand
        # -------------------------
        "uttarakhand": "Uttarakhand",
        "uttaranchal": "Uttarakhand",

        # -------------------------
        # Uttar Pradesh
        # -------------------------
        "uttarpradesh": "Uttar Pradesh",

        # -------------------------
        # West Bengal
        # -------------------------
        "westbengal": "West Bengal",
    }

    # Return canonical state name
    if key in state_map:
        return state_map[key]

    # --------------------------------------------------------
    # If not found, return cleaned original
    # --------------------------------------------------------

    return name


# ============================================================
# NUMERIC CLEANING
# ============================================================

def clean_numeric(series):

    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("*", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# ============================================================
# 1. PHC STAFF DATA
# ============================================================

def load_phc_staff_data():

    print("\n========================================")
    print("Loading PHC staff data")
    print("========================================")

    # --------------------------------------------------------
    # Main CSV
    # --------------------------------------------------------

    df = pd.read_csv(PHC_STAFF_FILE)

    print(f"CSV rows: {len(df)}")

    # Rename columns
    df = df.rename(columns={
        "State/ UT": "State",
        "Required - [R]": "Doctors_Required",
        "Sanctioned - [S]": "Doctors_Sanctioned",
        "In Position - [P]": "Doctors_In_Position",
        "Vacant - [S-P]": "Doctor_Vacancy",
        "Shortfall - [R-P]": "Doctor_Shortfall"
    })

    required_columns = [
        "State",
        "Doctors_Required",
        "Doctors_Sanctioned",
        "Doctors_In_Position",
        "Doctor_Vacancy",
        "Doctor_Shortfall"
    ]

    df = df[
        [
            column
            for column in required_columns
            if column in df.columns
        ]
    ]

    df["State"] = df["State"].apply(clean_state_name)

    numeric_columns = [
        "Doctors_Required",
        "Doctors_Sanctioned",
        "Doctors_In_Position",
        "Doctor_Vacancy",
        "Doctor_Shortfall"
    ]

    for column in numeric_columns:

        if column in df.columns:
            df[column] = clean_numeric(df[column])

    # --------------------------------------------------------
    # Second XLS dataset
    # --------------------------------------------------------

    print("\nLoading second PHC staff dataset...")

    try:

        df_xls = pd.read_excel(
            PHC_STAFF_XLS_FILE,
            engine="xlrd"
        )

        print(f"XLS rows: {len(df_xls)}")

        df_xls = df_xls.rename(columns={
            "State/UT": "State",
            "State/ UT": "State",
            "Required1 - [R]": "Doctors_Required",
            "Required - [R]": "Doctors_Required",
            "Sanctioned - [S]": "Doctors_Sanctioned",
            "In Position - [P]": "Doctors_In_Position",
            "Vacant - [S-P]": "Doctor_Vacancy",
            "Shortfall - [R-P]": "Doctor_Shortfall"
        })

        xls_columns = [
            "State",
            "Doctors_Required",
            "Doctors_Sanctioned",
            "Doctors_In_Position",
            "Doctor_Vacancy",
            "Doctor_Shortfall"
        ]

        df_xls = df_xls[
            [
                column
                for column in xls_columns
                if column in df_xls.columns
            ]
        ]

        df_xls["State"] = (
            df_xls["State"]
            .apply(clean_state_name)
        )

        for column in numeric_columns:

            if column in df_xls.columns:
                df_xls[column] = clean_numeric(
                    df_xls[column]
                )

        # We use the CSV as primary source.
        # XLS is retained for cross-checking.
        df_xls.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "phc_staff_secondary.csv"
            ),
            index=False
        )

    except Exception as e:

        print(
            "Warning: Could not process second "
            f"PHC dataset: {e}"
        )

    # Remove invalid rows
    df = df.dropna(
        subset=["State"]
    )

    # Remove duplicate states
    df = df.drop_duplicates(
        subset=["State"],
        keep="first"
    )

    return df


# ============================================================
# 2. AYUSH HOSPITAL DATA
# ============================================================

def load_ayush_data():

    print("\n========================================")
    print("Loading AYUSH hospital data")
    print("========================================")

    # The first row contains the actual column names
    df = pd.read_csv(
        AYUSH_FILE,
        header=0
    )

    print(f"Rows loaded: {len(df)}")

    print("\nOriginal AYUSH columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # The file contains multi-level headers.
    #
    # We primarily need:
    # State
    # Total hospitals
    # Total beds
    # --------------------------------------------------------

    # Show first few rows for verification
    print("\nFirst AYUSH rows:")
    print(
        df.head(5).to_string(
            index=False
        )
    )

    # The first data row contains sub-header information.
    # Remove it if it is not a state.
    if len(df) > 0:

        first_value = str(
            df.iloc[0, 1]
        ).lower()

        if (
            "state" in first_value
            or "govt" in first_value
            or "government" in first_value
        ):
            df = df.iloc[1:].copy()

    # State column
    state_column = df.columns[1]

    df = df.rename(
        columns={
            state_column: "State"
        }
    )

    df["State"] = (
        df["State"]
        .apply(clean_state_name)
    )

    # --------------------------------------------------------
    # Find total hospital and bed columns
    # --------------------------------------------------------

    hospital_candidates = [
        column
        for column in df.columns
        if "Number of Hospitals" in str(column)
    ]

    bed_candidates = [
        column
        for column in df.columns
        if "Number of Beds" in str(column)
    ]

    # Because the file has unnamed columns,
    # inspect positions instead of relying only on names.
    #
    # Known structure:
    #
    # State
    # Number of Hospitals
    #   Govt
    #   Local Body
    #   Others
    #   Total
    #
    # Number of Beds
    #   Govt
    #   Local Body
    #   Others
    #   Total

    # Total hospital count is normally column 5
    # Total bed count is normally column 9.
    #
    # We check that these columns exist.

    if len(df.columns) >= 10:

        hospital_total_column = df.columns[5]
        bed_total_column = df.columns[9]

        df["AYUSH_Hospitals"] = clean_numeric(
            df[hospital_total_column]
        )

        df["AYUSH_Beds"] = clean_numeric(
            df[bed_total_column]
        )

    else:

        df["AYUSH_Hospitals"] = None
        df["AYUSH_Beds"] = None

    df = df[
        [
            "State",
            "AYUSH_Hospitals",
            "AYUSH_Beds"
        ]
    ]

    df = df.dropna(
        subset=["State"]
    )

    df = df.drop_duplicates(
        subset=["State"],
        keep="first"
    )

    return df


# ============================================================
# 3. ALL INDIA FACILITY DATA
# ============================================================

def load_all_india_data():

    print("\n========================================")
    print("Loading All-India facility data")
    print("========================================")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # _All_India_DataUploadStatus.xls is actually
    # an HTML-based Excel export.
    #
    # Therefore we read it using pandas read_html.
    # --------------------------------------------------------

    try:

        tables = pd.read_html(
            ALL_INDIA_FILE
        )

    except Exception as e:

        raise RuntimeError(
            "Could not read All-India dataset. "
            f"Error: {e}"
        )

    print(
        f"HTML tables detected: {len(tables)}"
    )

    if len(tables) == 0:

        raise RuntimeError(
            "No tables found in All-India dataset."
        )

    # The main table is generally the largest table.
    df = max(
        tables,
        key=lambda table: table.shape[0]
        * table.shape[1]
    )

    print(
        f"Main table shape: {df.shape}"
    )

    # --------------------------------------------------------
    # The source has repeated facility rows:
    #
    # State
    #   SC
    #   PHC
    #   CHC
    #   SDH
    #   DH
    #
    # We only need PHC.
    # --------------------------------------------------------

    print("\nColumns detected:")
    print(df.columns.tolist())

    # Reset columns if multi-level
    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = [
            "_".join(
                str(part)
                for part in column
                if str(part) != "nan"
            ).strip()
            for column in df.columns
        ]

    # Locate state and facility columns
    state_column = df.columns[0]

    facility_column = None

    for column in df.columns:

        values = (
            df[column]
            .astype(str)
            .str.upper()
        )

        if values.eq("PHC").any():

            facility_column = column
            break

    if facility_column is None:

        # Sometimes facility type is column 2
        facility_column = df.columns[2]

    print(
        f"State column: {state_column}"
    )

    print(
        f"Facility column: {facility_column}"
    )

    # --------------------------------------------------------
    # Forward-fill state names.
    # --------------------------------------------------------

    df[state_column] = (
        df[state_column]
        .replace(
            ["", "nan", "None"],
            pd.NA
        )
        .ffill()
    )

    # --------------------------------------------------------
    # Keep only PHC rows
    # --------------------------------------------------------

    phc_df = df[
        df[facility_column]
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("PHC")
    ].copy()

    print(
        f"PHC rows found: {len(phc_df)}"
    )

    # --------------------------------------------------------
    # Based on the source structure:
    #
    # column 3 = Total Facility
    # column 4 = Public
    # column 5 = Private
    # column 6 = Urban
    # column 7 = Rural
    #
    # column 8 onwards are active/reporting statistics.
    # --------------------------------------------------------

    if len(phc_df.columns) >= 8:

        phc_df["PHC_Count"] = clean_numeric(
            phc_df.iloc[:, 3]
        )

        phc_df["PHC_Public"] = clean_numeric(
            phc_df.iloc[:, 4]
        )

        phc_df["PHC_Private"] = clean_numeric(
            phc_df.iloc[:, 5]
        )

        phc_df["PHC_Urban"] = clean_numeric(
            phc_df.iloc[:, 6]
        )

        phc_df["PHC_Rural"] = clean_numeric(
            phc_df.iloc[:, 7]
        )

    else:

        phc_df["PHC_Count"] = None
        phc_df["PHC_Public"] = None
        phc_df["PHC_Private"] = None
        phc_df["PHC_Urban"] = None
        phc_df["PHC_Rural"] = None

    phc_df = phc_df.rename(
        columns={
            state_column: "State"
        }
    )

    phc_df["State"] = (
        phc_df["State"]
        .apply(clean_state_name)
    )

    phc_df = phc_df[
        [
            "State",
            "PHC_Count",
            "PHC_Public",
            "PHC_Private",
            "PHC_Urban",
            "PHC_Rural"
        ]
    ]

    # Remove All India aggregate
    phc_df = phc_df[
        phc_df["State"].notna()
    ]

    phc_df = phc_df[
        ~phc_df["State"].str.upper().isin([
            "ALL INDIA",
            "ALL INDIA/ TOTAL",
            "TOTAL",
            "INDIA"
        ])
    ]

    phc_df = phc_df.dropna(
        subset=["State"]
    )

    phc_df = phc_df.drop_duplicates(
        subset=["State"],
        keep="first"
    )

    return phc_df


# ============================================================
# 4. BUILD MASTER DATASET
# ============================================================

def build_master_dataset():

    print("\n")
    print("========================================")
    print("      AROGYAFLOW AI")
    print("      INDIA DATA PIPELINE")
    print("========================================")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    staff_df = load_phc_staff_data()

    ayush_df = load_ayush_data()

    all_india_df = load_all_india_data()

    # --------------------------------------------------------
    # Save individual cleaned datasets
    # --------------------------------------------------------

    staff_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clean_phc_staff.csv"
        ),
        index=False
    )

    ayush_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clean_ayush.csv"
        ),
        index=False
    )

    all_india_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "clean_india_phc_facilities.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\n========================================")
    print("Merging datasets")
    print("========================================")

    master_df = pd.merge(
        all_india_df,
        staff_df,
        on="State",
        how="outer"
    )

    master_df = pd.merge(
        master_df,
        ayush_df,
        on="State",
        how="outer"
    )

    # --------------------------------------------------------
    # FINAL MASTER DATA CLEANING
    # --------------------------------------------------------

    # Normalize state names again after merging
    master_df["State"] = (
        master_df["State"]
        .apply(clean_state_name)
    )

    # --------------------------------------------------------
    # FINAL STATE NAME STANDARDIZATION
    # --------------------------------------------------------

    master_df["State"] = master_df["State"].replace({

        "Daman & Diu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "Daman &Diu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "Daman and Diu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "Daman&Diu":
            "Dadra & Nagar Haveli and Daman & Diu",

        "Dadra & Nagar Haveli":
            "Dadra & Nagar Haveli and Daman & Diu",
    })

    # Remove invalid/header/aggregate rows
    master_df = master_df[
        master_df["State"].notna()
    ]

    master_df = master_df[
        ~master_df["State"].str.upper().isin([
            "ALL INDIA",
            "ALL INDIA/ TOTAL",
            "TOTAL",
            "INDIA"
        ])
    ]

    # --------------------------------------------------------
    # CHECK FOR DUPLICATE STATES
    # --------------------------------------------------------

    duplicate_states = master_df[
        master_df["State"].duplicated(keep=False)
    ]["State"].unique()

    if len(duplicate_states) > 0:

        print("\nWARNING: Duplicate states detected:")
        print(duplicate_states)

    else:

        print("\n✓ No duplicate states detected")

    # --------------------------------------------------------
    # Data source information
    # --------------------------------------------------------

    master_df["Data_Source"] = (
        "Government public datasets"
    )

    master_df["Data_Year"] = (
        "2017-2020"
    )

    # --------------------------------------------------------
    # Derived features
    # --------------------------------------------------------

    master_df["Doctor_Vacancy_Rate"] = (
        master_df["Doctor_Vacancy"]
        .div(master_df["Doctors_Sanctioned"].replace(0, pd.NA))
    )

    master_df["Doctor_Shortfall_Rate"] = (
        master_df["Doctor_Shortfall"]
        .div(master_df["Doctors_Required"].replace(0, pd.NA))
    )

    # Avoid infinity
    master_df = master_df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    master_df = master_df.sort_values(
        by="State"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "india_healthcare_context.csv"
    )

    master_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("DATA PIPELINE COMPLETE")
    print("========================================")

    print(
        f"\nStates in master dataset: "
        f"{len(master_df)}"
    )

    print(
        f"Columns: "
        f"{len(master_df.columns)}"
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    print("\nFinal columns:")

    for column in master_df.columns:

        print(
            f"  ✓ {column}"
        )

    print("\nSample data:")

    print(
        master_df.head(10).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_master_dataset()