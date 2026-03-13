import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():

    model_path = Path("models/rf_aqi_model.joblib")
    feature_path = Path("models/feature_columns.json")

    model = joblib.load(model_path)

    with open(feature_path, "r") as f:
        feature_columns = json.load(f)

    return model, feature_columns


model, feature_columns = load_model()


# --------------------------------------------------
# Page Title
# --------------------------------------------------
st.title("🌍 AQI Prediction Dashboard")
st.write("Predict next-day Air Quality Index using Machine Learning")


# --------------------------------------------------
# Input Fields
# --------------------------------------------------
st.subheader("Pollution Measurements")

pm25 = st.slider("PM2.5", 0.0, 300.0, 80.0)
pm10 = st.slider("PM10", 0.0, 400.0, 120.0)
no2 = st.slider("NO2", 0.0, 200.0, 40.0)
so2 = st.slider("SO2", 0.0, 100.0, 15.0)
co = st.slider("CO", 0.0, 5.0, 0.8)
o3 = st.slider("O3", 0.0, 200.0, 45.0)


st.subheader("Historical AQI")

aqi_lag_1 = st.number_input("AQI Yesterday", value=90)
aqi_lag_3 = st.number_input("AQI 3 Days Ago", value=95)
aqi_lag_7 = st.number_input("AQI 7 Days Ago", value=88)

rolling_mean_3 = st.number_input("3 Day AQI Average", value=92)
rolling_mean_7 = st.number_input("7 Day AQI Average", value=90)
rolling_std_7 = st.number_input("7 Day AQI Volatility", value=12)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("Predict AQI"):

    input_data = {
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "so2": so2,
        "co": co,
        "o3": o3,
        "aqi_lag_1": aqi_lag_1,
        "aqi_lag_3": aqi_lag_3,
        "aqi_lag_7": aqi_lag_7,
        "rolling_mean_3": rolling_mean_3,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7
    }

    df = pd.DataFrame([input_data])

    df = df[feature_columns]

    prediction = model.predict(df)[0]

    st.subheader("Prediction Result")

    st.metric("Predicted AQI", round(prediction, 2))

    # AQI Category
    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 200:
        category = "Moderate"
    elif prediction <= 300:
        category = "Poor"
    elif prediction <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.write(f"Air Quality Category: **{category}**")