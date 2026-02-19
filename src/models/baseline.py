import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data.load_data import load_raw_data, filter_ahmedabad, prepare_time_series


# --------------------------------------------------
# Time-based Train/Test Split
# --------------------------------------------------
def time_series_split(df, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)
    
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    print(f"[INFO] Train size: {train.shape}")
    print(f"[INFO] Test size:  {test.shape}")
    
    return train, test


# --------------------------------------------------
# Naive Forecast (Tomorrow = Today)
# --------------------------------------------------
def naive_forecast(train, test):
    # Prediction = today's AQI
    y_pred = test["aqi"].values
    y_true = test["aqi_next_day"].values
    
    return y_true, y_pred


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\n[BASELINE PERFORMANCE]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    
    return mae, rmse


# --------------------------------------------------
# Run Baseline
# --------------------------------------------------
if __name__ == "__main__":
    raw = load_raw_data()
    ahmedabad = filter_ahmedabad(raw)
    df = prepare_time_series(ahmedabad)

    train, test = time_series_split(df)

    y_true, y_pred = naive_forecast(train, test)

    evaluate_model(y_true, y_pred)
