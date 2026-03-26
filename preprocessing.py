"""
Preprocessing pipeline for Pakistan Air Quality time-series data.
Handles loading, daily aggregation, missing values, and feature engineering.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# AQI category mapping (1-5 to standard labels)
AQI_CATEGORIES = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy for Sensitive",
    4: "Unhealthy",
    5: "Very Unhealthy",
}

DATA_DIR = Path(__file__).parent / "Testing"
POLLUTANT_COLS = [
    "components_co", "components_no", "components_no2", "components_o3",
    "components_so2", "components_pm2_5", "components_pm10", "components_nh3",
]
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m", "shortwave_radiation",
]
LAG_DAYS = [1, 2, 3]
ROLLING_WINDOWS = [3, 7]


def load_city_data(data_dir=DATA_DIR):
    """Load all city CSVs and combine with city column."""
    dfs = []
    for f in Path(data_dir).glob("*_complete_data_*.csv"):
        city = f.stem.split("_")[0].title()
        df = pd.read_csv(f)
        # Handle mixed formats: "1/7/2024 0:00" vs "13/07/2024 00:00:00"
        try:
            df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", dayfirst=True)
        except (ValueError, TypeError, AttributeError):
            df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
        df["city"] = city
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def aggregate_to_daily(hourly_df):
    """Aggregate hourly to daily: mode for main_aqi, mean for numeric columns."""
    hourly_df = hourly_df.copy()
    hourly_df["date"] = hourly_df["datetime"].dt.date

    # For AQI category use mode (most frequent value that day)
    daily_aqi = (
        hourly_df.groupby(["city", "date"])["main_aqi"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.median())
        .reset_index()
        .rename(columns={"main_aqi": "aqi_category"})
    )

    # Mean of pollutants and weather per day
    agg_cols = POLLUTANT_COLS + WEATHER_COLS
    daily_numeric = (
        hourly_df.groupby(["city", "date"])[agg_cols]
        .mean()
        .reset_index()
    )

    daily = daily_aqi.merge(daily_numeric, on=["city", "date"])
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["city", "date"]).reset_index(drop=True)


def add_lag_rolling_features(daily_df):
    """Add lag and rolling-window features per city."""
    out = []
    for city, g in daily_df.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)
        for lag in LAG_DAYS:
            g[f"aqi_lag_{lag}"] = g["aqi_category"].shift(lag)
            g[f"pm25_lag_{lag}"] = g["components_pm2_5"].shift(lag)
        for w in ROLLING_WINDOWS:
            g[f"pm25_roll_mean_{w}"] = g["components_pm2_5"].rolling(w, min_periods=1).mean().shift(1)
            g[f"aqi_roll_mean_{w}"] = g["aqi_category"].rolling(w, min_periods=1).mean().shift(1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def prepare_features(daily_df, target="aqi_category"):
    """Create feature matrix and target. Drops rows with NaN from lags."""
    feature_cols = (
        POLLUTANT_COLS + WEATHER_COLS +
        [c for c in daily_df.columns if c.startswith("aqi_lag_") or c.startswith("pm25_lag_") or
         c.startswith("pm25_roll_") or c.startswith("aqi_roll_")]
    )
    feature_cols = [c for c in feature_cols if c in daily_df.columns]
    X = daily_df[feature_cols].copy()
    y = daily_df[target]
    meta = daily_df[["city", "date"]].copy()
    # Drop rows where we don't have enough history (first few days)
    valid = X.notna().all(axis=1)
    X = X[valid].astype(float)
    y = y[valid]
    meta = meta[valid].reset_index(drop=True)
    return X, y, meta, feature_cols


def fit_preprocessor(X, scaler_path="scaler.joblib", feature_path="feature_columns.joblib"):
    """Fit StandardScaler and save with feature column list."""
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), feature_path)
    return scaler


def transform_with_preprocessor(X, scaler_path="scaler.joblib", feature_path="feature_columns.joblib"):
    """Load scaler and feature list, align columns, transform."""
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_path)
    X = X.reindex(columns=feature_cols)
    X = X.fillna(X.median())
    return scaler.transform(X), feature_cols


def run_pipeline(data_dir=DATA_DIR, save_daily_path="daily_aggregated.csv"):
    """Full pipeline: load -> daily agg -> lag/rolling -> return (X, y, meta, feature_cols)."""
    raw = load_city_data(data_dir)
    daily = aggregate_to_daily(raw)
    daily = add_lag_rolling_features(daily)
    daily.to_csv(save_daily_path, index=False)
    X, y, meta, feature_cols = prepare_features(daily)
    return X, y, meta, feature_cols, daily


if __name__ == "__main__":
    X, y, meta, feature_cols, daily = run_pipeline()
    print("Daily shape:", daily.shape)
    print("Feature matrix shape:", X.shape)
    print("Target value counts:\n", y.value_counts())
