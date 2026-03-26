"""
Prediction module: load model and preprocessing, forecast AQI category for next N days.
Supports RF, XGBoost (1-row) and LSTM, Temporal CNN (7-day sequence).
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

from preprocessing import (
    AQI_CATEGORIES,
    POLLUTANT_COLS,
    WEATHER_COLS,
    LAG_DAYS,
    ROLLING_WINDOWS,
    add_lag_rolling_features,
)

SEQ_LEN = 7  # for LSTM / Temporal CNN

try:
    from tensorflow.keras.models import load_model as keras_load_model
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


def load_artifacts(save_dir="."):
    save_dir = Path(save_dir)
    bundle = joblib.load(save_dir / "model.joblib")
    if bundle.get("model_path") and HAS_KERAS:
        bundle["model"] = keras_load_model(bundle["model_path"])
    scaler = joblib.load(save_dir / "scaler.joblib")
    feature_cols = joblib.load(save_dir / "feature_columns.joblib")
    return bundle, scaler, feature_cols


def build_feature_row_for_date(city_df, for_date, feature_cols):
    """
    Build one row of features for (city, for_date) using prior rows in city_df.
    city_df should include real data + any previously predicted rows (with aqi_category set).
    """
    for_d = pd.Timestamp(for_date)
    past = city_df[city_df["date"] < for_d].sort_values("date")
    if past.empty:
        return None
    last = past.iloc[-1]
    row = {
        "city": last["city"],
        "date": for_d,
        "aqi_category": np.nan,
        **{c: last.get(c, np.nan) for c in POLLUTANT_COLS + WEATHER_COLS},
    }
    for lag in LAG_DAYS:
        if len(past) >= lag:
            prev = past.iloc[-lag]
            row[f"aqi_lag_{lag}"] = prev["aqi_category"]
            row[f"pm25_lag_{lag}"] = prev["components_pm2_5"]
        else:
            row[f"aqi_lag_{lag}"] = np.nan
            row[f"pm25_lag_{lag}"] = np.nan
    for w in ROLLING_WINDOWS:
        window = past.tail(w)
        row[f"pm25_roll_mean_{w}"] = window["components_pm2_5"].mean() if len(window) else np.nan
        row[f"aqi_roll_mean_{w}"] = window["aqi_category"].mean() if len(window) else np.nan
    return row


def _get_seq_matrix(city_data, feature_cols, scaler):
    """Last SEQ_LEN rows as (1, SEQ_LEN, n_features) float matrix, scaled."""
    if len(city_data) < SEQ_LEN:
        return None
    block = city_data.tail(SEQ_LEN).reindex(columns=feature_cols)
    for c in block.columns:
        try:
            block[c] = pd.to_numeric(block[c], errors="coerce")
        except Exception:
            block[c] = 0
    block = block.fillna(0).astype(float)
    X = scaler.transform(block)
    return X[np.newaxis, :, :]  # (1, SEQ_LEN, n_features)


def forecast_next_n_days(daily_df, bundle, scaler, feature_cols, n_days=7):
    """
    For each city, forecast AQI category for the next n_days.
    RF/XGBoost: 1-row features. LSTM/TCN: 7-day sequence.
    """
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    model = bundle["model"]
    is_xgb = bundle.get("is_xgb", False)
    is_seq = bundle.get("is_seq", False)
    inv_map = bundle.get("inv_map")
    classes = bundle.get("classes", [1, 2, 3, 4, 5])
    rows = []

    cities = daily_df["city"].unique()
    last_date = daily_df["date"].max()

    for city in cities:
        city_data = daily_df[daily_df["city"] == city].sort_values("date").copy()
        min_len = SEQ_LEN if is_seq else max(LAG_DAYS)
        if len(city_data) < min_len:
            continue
        last_row = city_data.iloc[-1].to_dict()

        for step in range(1, n_days + 1):
            for_date = last_date + timedelta(days=step)
            if is_seq:
                X_seq = _get_seq_matrix(city_data, feature_cols, scaler)
                if X_seq is None:
                    continue
                pred_idx = np.argmax(model.predict(X_seq, verbose=0)[0])
                pred_label = int(classes[pred_idx])
            else:
                row_dict = build_feature_row_for_date(city_data, for_date, feature_cols)
                if row_dict is None:
                    continue
                X_row = pd.DataFrame([row_dict]).reindex(columns=feature_cols)
                for c in X_row.columns:
                    try:
                        X_row[c] = pd.to_numeric(X_row[c], errors="coerce")
                    except Exception:
                        X_row[c] = 0
                X_row = X_row.fillna(0).astype(float)
                X_scaled = scaler.transform(X_row)
                pred = model.predict(X_scaled)[0]
                pred_label = int(inv_map.get(int(pred), pred)) if (is_xgb and inv_map) else int(pred)
            rows.append({
                "City": city,
                "Date": for_date.strftime("%Y-%m-%d"),
                "Predicted_AQI_Category": AQI_CATEGORIES.get(pred_label, str(pred_label)),
                "Predicted_AQI_Value": pred_label,
            })
            new_row = {**last_row, "date": for_date, "aqi_category": pred_label}
            for c in POLLUTANT_COLS + WEATHER_COLS:
                new_row[c] = last_row.get(c, np.nan)
            city_data = pd.concat([city_data, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(rows)


def run_forecast(save_dir=".", daily_path=None, n_days=7, output_csv="predictions.csv"):
    save_dir = Path(save_dir)
    daily_path = daily_path or save_dir / "daily_aggregated.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Run training first to create {daily_path}")
    daily_df = pd.read_csv(daily_path)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    # Ensure we have lag/rolling columns
    if "aqi_lag_1" not in daily_df.columns:
        from preprocessing import add_lag_rolling_features
        daily_df = add_lag_rolling_features(daily_df)
    bundle, scaler, feature_cols = load_artifacts(save_dir)
    pred_df = forecast_next_n_days(daily_df, bundle, scaler, feature_cols, n_days=n_days)
    pred_df.to_csv(save_dir / output_csv, index=False)
    print(f"Saved predictions to {save_dir / output_csv}")
    return pred_df
