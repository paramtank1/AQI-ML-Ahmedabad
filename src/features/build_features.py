import pandas as pd
from src.data.load_data import load_raw_data, filter_ahmedabad, prepare_time_series


# --------------------------------------------------
# Lag Features
# --------------------------------------------------
def create_lag_features(df):
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_lag_7"] = df["aqi"].shift(7)
    return df


# --------------------------------------------------
# Rolling Features
# --------------------------------------------------
def create_rolling_features(df):
    df["rolling_mean_3"] = df["aqi"].rolling(window=3).mean()
    df["rolling_mean_7"] = df["aqi"].rolling(window=7).mean()
    df["rolling_std_7"] = df["aqi"].rolling(window=7).std()
    return df


# --------------------------------------------------
# Main Feature Builder
# --------------------------------------------------
def build_features(df):
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Drop rows created due to lag/rolling operations
    df = df.dropna().reset_index(drop=True)

    print(f"[INFO] Feature engineering complete.")
    print(f"[INFO] Final dataset shape after features: {df.shape}")

    return df


# --------------------------------------------------
# Test Run
# --------------------------------------------------
if __name__ == "__main__":
    raw = load_raw_data()
    ahmedabad = filter_ahmedabad(raw)
    df = prepare_time_series(ahmedabad)

    df_features = build_features(df)

    print(df_features.head())