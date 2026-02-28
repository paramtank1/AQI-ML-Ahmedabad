import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
# Manual Hyperparameter Tuning
# --------------------------------------------------
def manual_rf_tuning(X_train, y_train, X_test, y_test):

    n_estimators_list = [100, 200, 300]
    max_depth_list = [None, 10, 15, 20]
    min_samples_split_list = [2, 5, 10]

    best_mae = float("inf")
    best_params = None

    print("\n[STARTING MANUAL RF TUNING]\n")

    for n in n_estimators_list:
        for depth in max_depth_list:
            for min_split in min_samples_split_list:

                model = RandomForestRegressor(
                    n_estimators=n,
                    max_depth=depth,
                    min_samples_split=min_split,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)

                print(f"n_estimators={n}, max_depth={depth}, min_samples_split={min_split} --> MAE: {mae:.4f}")

                if mae < best_mae:
                    best_mae = mae
                    best_params = (n, depth, min_split)

    print("\n[BEST CONFIGURATION]")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Best Params: n_estimators={best_params[0]}, max_depth={best_params[1]}, min_samples_split={best_params[2]}")

    return best_params


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

    manual_rf_tuning(X_train, y_train, X_test, y_test)