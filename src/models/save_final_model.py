import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

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
# Train Final Model
# --------------------------------------------------
def train_final_model(X, y):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model


# --------------------------------------------------
# Save Model
# --------------------------------------------------
def save_model(model, feature_columns):

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "rf_aqi_model.joblib"
    feature_path = model_dir / "feature_columns.json"

    joblib.dump(model, model_path)

    with open(feature_path, "w") as f:
        json.dump(list(feature_columns), f)

    print("\nModel saved successfully!")
    print(f"Model path: {model_path}")
    print(f"Feature columns saved: {feature_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    raw = load_raw_data()

    ahmedabad = filter_ahmedabad(raw)

    df = prepare_time_series(ahmedabad)

    df = build_features(df)

    X, y = prepare_features(df)

    model = train_final_model(X, y)

    save_model(model, X.columns)