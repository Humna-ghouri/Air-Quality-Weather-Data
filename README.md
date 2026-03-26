# Pakistan Air Quality Level Prediction (ML)

Machine learning project to **predict daily AQI category** for Pakistani cities using historical air quality and weather data. The goal is to forecast pollution levels a few days ahead so authorities can issue early warnings.

---

## 1. Dataset Used

- **Source**: Hourly air quality and weather data for **5 cities** (Islamabad, Karachi, Lahore, Peshawar, Quetta), **July–December 2024**.
- **Location**: CSV files in the `Testing/` folder:
  - `islamabad_complete_data_july_to_dec_2024.csv`
  - `karachi_complete_data_july_to_dec_2024.csv`
  - `lahore_complete_data_july_to_dec_2024.csv`
  - `peshawar_complete_data_july_to_dec_2024.csv`
  - `quetta_complete_data_july_to_dec_2024.csv`
- **Variables**:
  - **Target**: `main_aqi` (1–5) → mapped to AQI categories: Good, Moderate, Unhealthy for Sensitive, Unhealthy, Very Unhealthy.
  - **Pollutants**: CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃.
  - **Weather**: temperature, humidity, dew point, precipitation, pressure, wind speed/direction, shortwave radiation.

---

## 2. Preprocessing & Feature Engineering

- **Cleaning**: Hourly data aggregated to **daily** (mode for AQI category, mean for numeric columns). Missing values handled via median fill where needed.
- **Feature engineering**:
  - **Lag features**: AQI and PM2.5 at lags 1, 2, 3 days.
  - **Rolling features**: 3- and 7-day rolling means of AQI and PM2.5 (shifted by 1 day to avoid leakage).
  - **Environmental features**: All pollutant and weather columns (daily means).
- **Train/test split**: Temporal — last 20% of days held out for evaluation (simulating real forecast conditions).

---

## 3. Models & Evaluation

- **Models trained**: **Random Forest** and **XGBoost** (if installed). The best model by **F1-score (weighted)** is saved.
- **Evaluation metrics**:
  - **Accuracy** and **F1-score (weighted)** for AQI category prediction.
  - Classification report and confusion matrix (see training output).
- **Prediction lead time**: Forecasts for the **next 1–7 days** (configurable). Multi-step forecasting is done iteratively: day+1 uses last real data; day+2 uses predicted day+1 in lags, etc.

---

## 4. Repository Structure

```
air-quality-ml/
├── Testing/                          # City CSVs (hourly data)
├── preprocessing.py                 # Load, aggregate, lag/rolling features
├── train.py                          # Train RF/XGBoost, save model & pipeline
├── predict.py                        # Forecast next N days
├── generate_predictions.py           # CLI to produce prediction CSV
├── app.py                            # Streamlit app
├── requirements.txt
├── README.md
├── model.joblib                      # (after training) saved model bundle
├── scaler.joblib                     # (after training) feature scaler
├── feature_columns.joblib            # (after training) feature list
├── daily_aggregated.csv              # (after training) daily data with features
├── predictions.csv                   # (after generate_predictions) forecast CSV
└── evaluation.csv                   # (after training) accuracy/F1 per model
```

---

## 5. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model and save artifacts

From the project root (folder containing `Testing/`):

```bash
python train.py --save-dir .
```

Optional: `--data-dir path/to/csvs` if data is not in `Testing/`, `--test-size 0.2`, `--seed 42`.

### Generate prediction CSV (next N days)

```bash
python generate_predictions.py --save-dir . --n-days 3 --output predictions.csv
```

Output: `predictions.csv` with columns **City**, **Date**, **Predicted_AQI_Category**, **Predicted_AQI_Value**.

### Run Streamlit app

```bash
streamlit run app.py
```

- Select **city** and **forecast horizon** (1–7 days).
- View AQI forecast and **alerts** for Unhealthy / Very Unhealthy.
- All cities’ next-day summary (city-wise risk) is shown at the bottom.

---

## 6. Final Submission Checklist

| Item | Description |
|------|-------------|
| **Code & model** | `train.py`, `preprocessing.py`, `predict.py`; saved `model.joblib`, `scaler.joblib`, `feature_columns.joblib` |
| **Prediction output** | `predictions.csv`: City, Date, Predicted AQI category (and value) for next N days |
| **Streamlit app** | `app.py`: city selection, AQI forecast, unhealthy air alerts; loads trained model |
| **Report** | This README: dataset, features, model choice, evaluation, limitations |

---

## 7. Limitations

- **Data**: Only July–Dec 2024 for 5 cities; no seasonal coverage across years.
- **Forecast horizon**: Accuracy drops for longer lead times (e.g. day+3) due to iterative use of predicted values in lags.
- **Features**: Future days use last-known pollutant/weather (no separate weather forecast).
- **Classes**: Imbalanced AQI categories may bias metrics; weighted F1 is used to mitigate.

---

## 8. Optional / Bonus

- **Visual forecast**: Streamlit table and per-city forecast section.
- **City-wise risk**: “All cities — next day” table sorted by predicted AQI.
- **Explainability**: Random Forest feature importances could be added in `train.py` (e.g. `model.feature_importances_`).
