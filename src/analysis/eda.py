import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import loader functions
from src.data.load_data import load_raw_data, filter_ahmedabad, prepare_time_series


# --------------------------------------------------
# Basic Summary
# --------------------------------------------------
def basic_summary(df):
    print("\n[INFO] ===== BASIC SUMMARY =====")
    print(df.describe())


# --------------------------------------------------
# Date Range Check
# --------------------------------------------------
def check_date_range(df):
    print("\n[INFO] ===== DATE RANGE =====")
    print(f"Start Date: {df['date'].min()}")
    print(f"End Date:   {df['date'].max()}")
    print(f"Total Days: {df.shape[0]}")


# --------------------------------------------------
# AQI Distribution
# --------------------------------------------------
def plot_aqi_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["aqi"], bins=40, kde=True)
    plt.title("Distribution of AQI (Ahmedabad)")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Boxplot for Outliers
# --------------------------------------------------
def plot_boxplots(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df["aqi"])
    plt.title("AQI Boxplot (Outlier Check)")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------
def plot_correlation(df):
    plt.figure(figsize=(10, 6))
    corr = df[["pm25", "pm10", "no2", "so2", "co", "o3", "aqi"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# AQI Time Series Plot
# --------------------------------------------------
def plot_time_series(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["aqi"])
    plt.title("AQI Over Time (Ahmedabad)")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Run EDA
# --------------------------------------------------
if __name__ == "__main__":
    raw = load_raw_data()
    ahmedabad = filter_ahmedabad(raw)
    df = prepare_time_series(ahmedabad)

    basic_summary(df)
    check_date_range(df)

    plot_aqi_distribution(df)
    plot_boxplots(df)
    plot_correlation(df)
    plot_time_series(df)
