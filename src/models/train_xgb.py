import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.load_data import load_raw_data, filter_ahmedabad, prepare_time_series
from src.features.build_features import build_features


# --------------------------------------------------
# Time-based Train/Test Split
# --------------------------------------------------
def time_series_split(df, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


# --------------------------------------------------
# Prepare Feature Matrix
# --------------------------------------------------
def prepare_features(train, test):
    drop_columns = ["city", "date", "aqi", "aqi_category", "aqi_next_day"]

    X_train = train.drop(columns=drop_columns)
    y_train = train["aqi_next_day"]

    X_test = test.drop(columns=drop_columns)
    y_test = test["aqi_next_day"]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# Train XGBoost
# --------------------------------------------------
def train_xgb(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("[INFO] XGBoost training complete.")
    return model


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n[XGBOOST PERFORMANCE]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return mae, rmse


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
def print_feature_importance(model, X_train):
    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)

    print("\n[FEATURE IMPORTANCE]")
    print(importance)


# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    raw = load_raw_data()
    ahmedabad = filter_ahmedabad(raw)
    df = prepare_time_series(ahmedabad)
    df = build_features(df)

    train, test = time_series_split(df)
    X_train, X_test, y_train, y_test = prepare_features(train, test)

    model = train_xgb(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    print_feature_importance(model, X_train)