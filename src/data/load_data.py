import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Path Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "india_city_aqi_2015_2023.csv"


# --------------------------------------------------
# Load Raw Dataset
# --------------------------------------------------
def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Raw dataset loaded successfully.")
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


# --------------------------------------------------
# Filter Ahmedabad Data
# --------------------------------------------------
def filter_ahmedabad(df):
    df_city = df[df["city"].str.strip().str.lower() == "ahmedabad"].copy()
    print(f"[INFO] Ahmedabad data shape: {df_city.shape}")
    return df_city

def prepare_time_series(df):
    """
    Prepare dataframe for time-series modeling.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Create next-day AQI target
    df["aqi_next_day"] = df["aqi"].shift(-1)

    # Drop last row (no next-day target)
    df = df.dropna()

    print(f"[INFO] Time-series prepared.")
    print(f"[INFO] Final dataset shape after shift: {df.shape}")

    return df




# --------------------------------------------------
# Basic Inspection
# --------------------------------------------------
def inspect_data(df):
    print("\n[INFO] First 5 rows:")
    print(df.head())

    print("\n[INFO] Data Info:")
    print(df.info())

    print("\n[INFO] Missing Values per Column:")
    print(df.isna().sum())


# --------------------------------------------------
# Run Test
# --------------------------------------------------
if __name__ == "__main__":
    raw_df = load_raw_data()
    ahmedabad_df = filter_ahmedabad(raw_df)
    ts_df = prepare_time_series(ahmedabad_df)
    inspect_data(ts_df)

