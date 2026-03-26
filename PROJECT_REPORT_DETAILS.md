# Pakistan Air Quality Prediction — Full Project Report (Detail)

**Is project mein kya use kiya, kaise kiya, kya output aaya, kitni accuracy hai, kaunsa best model hai — sab detail.**

---

## 1. Dataset — Kya Use Kiya?

| Item | Detail |
|------|--------|
| **Source** | `Testing/` folder mein 5 Pakistani cities ke CSV files |
| **Cities** | Islamabad, Karachi, Lahore, Peshawar, Quetta |
| **Period** | July 2024 – December 2024 (hourly data) |
| **Files** | `islamabad_complete_data_july_to_dec_2024.csv`, `karachi_...`, `lahore_...`, `peshawar_...`, `quetta_...` |
| **Variables** | **Target:** `main_aqi` (1–5) = AQI category. **Pollutants:** CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃. **Weather:** temperature, humidity, dew point, precipitation, pressure, wind speed/direction, radiation. |

**Summary:** Historical hourly air quality + weather data 5 cities ke liye use kiya, daily level par aggregate karke model train kiya.

---

## 2. Preprocessing — Kaise Clean & Prepare Kiya?

| Step | Kya kiya |
|------|----------|
| **Load** | Sab city CSVs read kiye, `datetime` parse kiya (mixed format handle: "1/7/2024 0:00" aur "13/07/2024 00:00:00" dono). |
| **Missing** | Jahan NaN tha, median se fill kiya; daily aggregate ke baad jo rows lag/rolling ki wajah se incomplete thi unhe drop kiya. |
| **Daily aggregation** | Hourly → daily: AQI category ke liye **mode** (sabse zyada aane wala category), pollutants/weather ke liye **mean**. |
| **Feature engineering** | **Lag features:** AQI aur PM2.5 ke 1, 2, 3 din pehle ke values. **Rolling:** 3-day aur 7-day rolling mean (shift 1 day taake future leak na ho). **Environmental:** Saare pollutant aur weather columns (daily mean). |
| **Scaling** | `StandardScaler` se features scale kiye (model consistency ke liye). |

**Output:** `daily_aggregated.csv` — har city, har date ke liye ek row + saare features.

---

## 3. Models — Konsa Use Kiya, Kaunsa Best?

| Model | Use kiya? | Accuracy | F1-Score (weighted) | Best? |
|-------|-----------|----------|----------------------|-------|
| **Random Forest** | Haan | **75.00%** | **0.726** | No |
| **XGBoost** | Haan | **76.67%** | **0.749** | **Yes (saved)** |

- **Best model:** **XGBoost** — F1-score zyada honay ki wajah se isi ko save kiya (`model.joblib`).
- **Training:** Last 20% days ko test set banaya (time-order) taake real forecast jaisa evaluation ho.
- **Evaluation:** Accuracy aur F1 (weighted) dono use kiye; F1 se best model choose kiya.

**Exact numbers (evaluation.csv):**

- Random Forest: accuracy = 0.75, f1_weighted ≈ 0.726  
- XGBoost: accuracy = 0.7667, f1_weighted ≈ 0.749  

---

## 4. Outputs — Kya Kya Aaya?

| Output | File / Place | Kya hai |
|--------|----------------|--------|
| **Trained model** | `model.joblib` | XGBoost classifier (best by F1) |
| **Preprocessing** | `scaler.joblib`, `feature_columns.joblib` | Scaling aur feature list — prediction ke liye |
| **Daily data** | `daily_aggregated.csv` | Preprocessed daily data + features |
| **Evaluation** | `evaluation.csv` | RF vs XGBoost accuracy & F1 |
| **Predictions** | `predictions.csv` (run `generate_predictions.py` se) | City, Date, Predicted_AQI_Category, next N days |
| **App** | `app.py` (Streamlit) | City select, forecast next days, Unhealthy alerts |

---

## 5. Prediction — Kaise Forecast Kiya?

- **Lead time:** Next 1–7 days (default 3 days).
- **Method:** Multi-step iterative: Day+1 last real day se; Day+2 ke liye predicted Day+1 ko lag mein use kiya; same Day+3 ke liye.
- **Output format:** CSV mein **City**, **Date**, **Predicted AQI category** (Good / Moderate / Unhealthy for Sensitive / Unhealthy / Very Unhealthy).

---

## 6. Streamlit App — Kya Kya Hai?

- **City selection** — dropdown se city choose karo.
- **Forecast next N days** — 1 se 7 days select karo.
- **AQI forecast** — har date ke liye predicted category dikhta hai.
- **Alerts** — Unhealthy / Very Unhealthy par warning.
- **Bonus:** “All cities — next day” table (city-wise risk jaisa).

App trained model load karti hai (`model.joblib`), isliye pehle `train.py` run karna zaroori tha.

---

## 7. Code & Files — Kya Use Kiya (Summary)

| File | Kaam |
|------|------|
| `preprocessing.py` | Load CSVs, daily aggregate, lag/rolling features, missing handle. |
| `train.py` | Preprocess → train RF + XGBoost → best (F1) save, evaluation.csv. |
| `predict.py` | Model load, next N days forecast, numeric handling (city column fix). |
| `generate_predictions.py` | CLI se predictions.csv banane ke liye. |
| `app.py` | Streamlit: city, forecast, alerts. |
| `training_notebook.ipynb` | Same pipeline notebook se run karne ke liye. |
| `README.md` | Dataset, features, model, evaluation, limitations. |

---

## 8. Short Summary (Ek Nazar Mein)

- **Dataset:** 5 cities, Jul–Dec 2024, hourly → daily.  
- **Features:** Lag 1/2/3, rolling 3 & 7, pollutants, weather.  
- **Models:** Random Forest (75% acc, 0.726 F1) aur **XGBoost (76.67% acc, 0.749 F1) — best, saved.**  
- **Output:** `model.joblib`, `predictions.csv`, Streamlit app with city + forecast + Unhealthy alerts.  
- **Evaluation:** Accuracy / F1-score; prediction lead time = 1–7 days.

Agar aur detail chahiye (e.g. exact classification report, confusion matrix) to `train.py` run karke console output dekho, ya batao kaunsa section expand karna hai.
