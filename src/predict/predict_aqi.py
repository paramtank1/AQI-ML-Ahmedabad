import joblib
import json
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Load Model
# --------------------------------------------------
def load_model():

    model_path = Path("models/rf_aqi_model.joblib")
    feature_path = Path("models/feature_columns.json")

    model = joblib.load(model_path)

    with open(feature_path, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns


# --------------------------------------------------
# Create Input Data
# --------------------------------------------------
def prepare_input(feature_columns):

    # Example input values
    input_data = {
        "pm25": 80,
        "pm10": 120,
        "no2": 40,
        "so2": 15,
        "co": 0.8,
        "o3": 45,
        "aqi_lag_1": 90,
        "aqi_lag_3": 95,
        "aqi_lag_7": 88,
        "rolling_mean_3": 92,
        "rolling_mean_7": 90,
        "rolling_std_7": 12
    }

    df = pd.DataFrame([input_data])

    # Ensure correct feature order
    df = df[feature_columns]

    return df


# --------------------------------------------------
# Predict AQI
# --------------------------------------------------
def predict():

    model, feature_columns = load_model()

    input_df = prepare_input(feature_columns)

    prediction = model.predict(input_df)

    print("\nPredicted Next-Day AQI:", round(prediction[0], 2))


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    predict()