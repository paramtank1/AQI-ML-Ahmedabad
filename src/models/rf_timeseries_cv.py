import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from src.data.load_data import load_raw_data, filter_ahmedabad, prepare_time_series
from src.features.build_features import build_features


# --------------------------------------------------
# Prepare Feature Matrix
# --------------------------------------------------
def prepare_features(df):
    drop_columns = ["city", "date", "aqi", "aqi_category", "aqi_next_day"]

    X = df.drop(columns=drop_columns)
    y = df["aqi_next_day"]

    return X, y


# --------------------------------------------------
# TimeSeries Cross-Validation
# --------------------------------------------------
def run_timeseries_cv(X, y, n_splits=5):

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_mae = []

    print("\n[TimeSeriesSplit Cross-Validation]\n")

    fold_number = 1

    for train_index, test_index in tscv.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        fold_mae.append(mae)

        print(f"Fold {fold_number} MAE: {mae:.4f}")
        fold_number += 1

    print("\n[Cross-Validation Summary]")
    print(f"Mean MAE: {np.mean(fold_mae):.4f}")
    print(f"Std  MAE: {np.std(fold_mae):.4f}")

    return fold_mae


# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":

    raw = load_raw_data()
    ahmedabad = filter_ahmedabad(raw)
    df = prepare_time_series(ahmedabad)
    df = build_features(df)

    X, y = prepare_features(df)

    run_timeseries_cv(X, y, n_splits=5)