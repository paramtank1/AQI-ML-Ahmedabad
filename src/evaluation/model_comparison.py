import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data.load_data import load_raw_data, filter_ahmedabad, prepare_time_series
from src.features.build_features import build_features


# --------------------------------------------------
# Prepare Features
# --------------------------------------------------
def prepare_features(df):

    drop_columns = ["city", "date", "aqi", "aqi_category", "aqi_next_day"]

    X = df.drop(columns=drop_columns)
    y = df["aqi_next_day"]

    return X, y


# --------------------------------------------------
# Model Dictionary
# --------------------------------------------------
def get_models():

    models = {

        "Linear Regression": LinearRegression(),

        "kNN Regressor": KNeighborsRegressor(n_neighbors=5),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    }

    return models


# --------------------------------------------------
# TimeSeries Cross Validation
# --------------------------------------------------
def evaluate_models(X, y):

    models = get_models()

    tscv = TimeSeriesSplit(n_splits=5)

    results = []

    for name, model in models.items():

        mae_scores = []
        rmse_scores = []
        r2_scores = []

        print(f"\nEvaluating: {name}")

        for train_index, test_index in tscv.split(X):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)

        results.append({
            "Model": name,
            "MAE": np.mean(mae_scores),
            "RMSE": np.mean(rmse_scores),
            "R2": np.mean(r2_scores)
        })

    results_df = pd.DataFrame(results)

    return results_df


# --------------------------------------------------
# Save Results
# --------------------------------------------------
def save_results(results_df):

    results_df.to_csv("reports/model_results.csv", index=False)

    print("\nResults saved to reports/model_results.csv")


# --------------------------------------------------
# Plot Graphs
# --------------------------------------------------
def plot_results(results_df):

    # MAE Plot
    plt.figure()
    plt.bar(results_df["Model"], results_df["MAE"])
    plt.title("Model Comparison - MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/model_mae_comparison.png")

    # RMSE Plot
    plt.figure()
    plt.bar(results_df["Model"], results_df["RMSE"])
    plt.title("Model Comparison - RMSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("reports/figures/model_rmse_comparison.png")

    print("Graphs saved to reports/figures/")


# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":

    raw = load_raw_data()
    ahmedabad = filter_ahmedabad(raw)

    df = prepare_time_series(ahmedabad)
    df = build_features(df)

    X, y = prepare_features(df)

    results = evaluate_models(X, y)

    print("\nModel Comparison Results:")
    print(results)

    save_results(results)

    plot_results(results)