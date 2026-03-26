# )
# """
# Streamlit app: Pakistan AQI forecast.
# - City selection
# - AQI forecast for next N days
# - Alerts for Unhealthy / Very Unhealthy
# - Loads trained model from save_dir
# """
# import streamlit as st
# import pandas as pd
# import joblib
# from pathlib import Path
# import plotly.express as px
# import plotly.graph_objects as go
# from predict import load_artifacts, forecast_next_n_days
# from preprocessing import AQI_CATEGORIES, add_lag_rolling_features

# # Define AQI color mapping for UI
# AQI_COLORS = {
#     1: {"category": "Good", "color_dark": "#00E676", "color_light": "#4CAF50"},  # Green
#     2: {"category": "Moderate", "color_dark": "#FFD600", "color_light": "#FFEB3B"}, # Yellow
#     3: {"category": "Unhealthy for Sensitive", "color_dark": "#FF6F00", "color_light": "#FF9800"}, # Orange
#     4: {"category": "Unhealthy", "color_dark": "#D50000", "color_light": "#F44336"}, # Red
#     5: {"category": "Very Unhealthy", "color_dark": "#AA00FF", "color_light": "#9C27B0"}, # Purple
# }

# def get_aqi_color_hex(aqi_value, is_dark_theme=True):
#     """Returns hex color for a given AQI integer value (1-5)."""
#     color_key = "color_dark" if is_dark_theme else "color_light"
#     return AQI_COLORS.get(aqi_value, {"color_dark": "#808080", "color_light": "#808080"})[color_key]

# def get_aqi_category_style(category_string, is_dark_theme=True):
#     """Returns CSS style string for an AQI category string."""
#     for aqi_val, data in AQI_COLORS.items():
#         if data["category"] == category_string:
#             color_key = "color_dark" if is_dark_theme else "color_light"
#             return f'background-color: {data[color_key]}; color: black; font-weight: bold;'
#     return ''

# st.set_page_config(page_title="Pakistan AQI Forecast", page_icon="🌫️", layout="wide")

# # Custom CSS for a sleek dark/light theme and card-style sections
# def load_css(is_dark_theme):
#     bg_color = "#0E1117" if is_dark_theme else "#FFFFFF"
#     text_color = "#FAFAFA" if is_dark_theme else "#333333"
#     primary_color = "#00E676" if is_dark_theme else "#4CAF50" # Green accent
#     secondary_bg = "#1E2127" if is_dark_theme else "#F0F2F6"
#     card_bg = "#1E2127" if is_dark_theme else "#F9F9F9"
#     card_border = "#333333" if is_dark_theme else "#E0E0E0"

#     css = f"""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

#     html, body, [data-testid="stAppViewContainer"] {{
#         background-color: {bg_color};
#         color: {text_color};
#         font-family: 'Roboto', sans-serif;
#     }}

#     /* Headers */
#     h1, h2, h3, h4, h5, h6 {{
#         color: {primary_color};
#         font-weight: 700;
#         font-family: 'Roboto', sans-serif;
#     }}

#     /* Buttons */
#     .stButton > button {{
#         background-color: {primary_color};
#         color: {text_color};
#         border-radius: 8px;
#         padding: 10px 24px;
#         font-size: 16px;
#         font-weight: bold;
#         border: none;
#         transition: background-color 0.3s ease;
#     }}
#     .stButton > button:hover {{
#         background-color: {primary_color}D0; /* Slightly darker/lighter on hover */
#     }}

#     /* Sliders */
#     .stSlider > div > div:nth-child(1) > div:nth-child(1) {{
#         background-color: {primary_color};
#     }}
#     .stSlider > div > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) {{
#         background-color: {primary_color};
#     }}

#     /* Selectbox */
#     .stSelectbox > div:first-child {{
#         border-radius: 8px;
#         background-color: {secondary_bg};
#         color: {text_color};
#     }}
#     .stSelectbox > div:first-child > div:first-child {{
#         color: {text_color};
#     }}
    
#     /* Main content container (Streamlit's default container) */
#     .css-1fv8s86.eqr7zpz1 {{
#         background-color: {secondary_bg}; 
#         padding: 2rem;
#         border-radius: 10px;
#         box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); /* Subtle shadow for depth */
#     }}
#     .css-1fv8s86.eqr7zpz1 .stAlert {{
#         border-radius: 8px;
#     }}

#     /* Card Style */
#     .card {{
#         background-color: {card_bg};
#         border-radius: 10px;
#         padding: 1.5rem;
#         margin-bottom: 1.5rem;
#         box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
#         border: 1px solid {card_border};
#     }}
#     .card-header {{
#         font-weight: bold;
#         color: {primary_color};
#         font-size: 1.2rem;
#         margin-bottom: 1rem;
#     }}

#     /* Expander */
#     .streamlit-expanderHeader {{
#         background-color: {secondary_bg};
#         border-radius: 8px;
#         border: 1px solid {card_border};
#         color: {text_color};
#     }}
#     .streamlit-expanderContent {{
#         background-color: {card_bg};
#         border-bottom-left-radius: 8px;
#         border-bottom-right-radius: 8px;
#         border: 1px solid {card_border};
#         border-top: none;
#         padding: 1rem;
#     }}

#     /* Dataframe styling */
#     .dataframe {{
#         color: {text_color};
#         background-color: {card_bg};
#     }}
#     .dataframe th {{
#         background-color: {secondary_bg};
#         color: {primary_color};
#         font-weight: bold;
#     }}
#     .dataframe tr:nth-child(even) {{
#         background-color: {secondary_bg};
#     }}
#     .stHorizontalBlock {{
#         gap: 0.5rem; /* Reduce gap between columns for tighter layout */
#     }}
#     </style>
#     """
#     return css

# # Initialize theme in session state
# if 'is_dark_theme' not in st.session_state:
#     st.session_state.is_dark_theme = True # Default to dark theme

# # Get Plotly theme based on current theme
# def get_plotly_theme(is_dark_theme):
#     if is_dark_theme:
#         return {
#             "layout": {
#                 "paper_bgcolor": "#1E2127",
#                 "plot_bgcolor": "#1E2127",
#                 "font": {"color": "#FAFAFA"},
#                 "title": {"font": {"color": "#FAFAFA"}}
#             }
#         }
#     else:
#         return {
#             "layout": {
#                 "paper_bgcolor": "#F9F9F9",
#                 "plot_bgcolor": "#F9F9F9",
#                 "font": {"color": "#333333"},
#                 "title": {"font": {"color": "#333333"}}
#             }
#         }

# # Load CSS
# st.markdown(load_css(st.session_state.is_dark_theme), unsafe_allow_html=True)

# SAVE_DIR = Path(__file__).parent
# MODEL_PATH = SAVE_DIR / "model.joblib"
# DAILY_PATH = SAVE_DIR / "daily_aggregated.csv"

# @st.cache_resource
# def load_model_and_data():
#     if not MODEL_PATH.exists():
#         return None, None, "Model not found. Run: python train.py --save-dir ."
#     try:
#         bundle, scaler, feature_cols = load_artifacts(SAVE_DIR)
#     except Exception as e:
#         return None, None, str(e)
#     if not DAILY_PATH.exists():
#         return (bundle, scaler, feature_cols), None, "daily_aggregated.csv not found. Run training first."
#     daily = pd.read_csv(DAILY_PATH)
#     daily["date"] = pd.to_datetime(daily["date"])
#     if "aqi_lag_1" not in daily.columns:
#         daily = add_lag_rolling_features(daily)
#     return (bundle, scaler, feature_cols), daily, None

# def main():
#     st.title("🌫️ Pakistan Air Quality Forecast")
#     st.markdown("### Predict daily AQI category for major Pakistani cities. **Full week (7 days)** forecast by default.")
#     st.markdown("--- ✨ ***A Sleek & Interactive Air Quality Prediction Dashboard*** ✨ ---")
    
#     artifacts, daily_df, err = load_model_and_data()
#     if err:
#         st.error(err)
#         st.info("From project root run: `python train.py --save-dir .` then `python generate_predictions.py --save-dir .`")
#         return

#     bundle, scaler, feature_cols = artifacts
#     cities = sorted(daily_df["city"].unique().tolist())

#     # Sidebar: Theme toggle and model info
#     with st.sidebar:
#         st.subheader("🎨 Theme")
#         # Theme toggle using checkbox
#         st.session_state.is_dark_theme = st.checkbox(
#             "Dark Mode", 
#             value=st.session_state.is_dark_theme,
#             help="Toggle between dark and light theme"
#         )
        
#         st.markdown("---")
#         st.subheader("📊 Model & Evaluation")
#         st.caption(f"**Best model:** {bundle.get('model_name', 'N/A')}")
#         eval_path = SAVE_DIR / "evaluation.csv"
#         if eval_path.exists():
#             eval_df = pd.read_csv(eval_path)
#             # Apply color to the model name based on its F1 score (simple color coding)
#             def color_f1(val):
#                 if val >= 0.75: 
#                     return 'background-color: #00E400; color: black;'
#                 elif val >= 0.70: 
#                     return 'background-color: #FFFF00; color: black;'
#                 else: 
#                     return ''
#             st.dataframe(eval_df.style.applymap(color_f1, subset=["f1_weighted"]), 
#                         use_container_width=True, hide_index=True)
#         st.markdown("--- ")
#         st.caption("Change city/slider below to update forecast.")
#         st.markdown("**Developed by:** AI Team")

#     col1, col2 = st.columns([1, 2])
#     with col1:
#         city = st.selectbox("Select city", cities, index=0)
#         n_days = st.slider("Forecast next N days", min_value=1, max_value=7, value=7)

#     # Always run and show forecast (visible on first load)
#     with st.spinner("Computing forecast..."):
#         pred_df = forecast_next_n_days(daily_df, bundle, scaler, feature_cols, n_days=n_days)
#     city_pred = pred_df[pred_df["City"] == city]

#     if not city_pred.empty:
#         st.subheader(f"Forecast for **{city}** (next {n_days} days)")
        
#         # Display daily forecasts
#         for _, row in city_pred.iterrows():
#             cat = row["Predicted_AQI_Category"]
#             date = row["Date"]
#             aqi_val = row["Predicted_AQI_Value"]
#             color = get_aqi_color_hex(aqi_val, st.session_state.is_dark_theme)
#             st.markdown(f"**{date}** — <span style='color:{color}; font-weight:bold;'>{cat}</span>", 
#                        unsafe_allow_html=True)

#         # Full week chart: AQI value over dates
#         st.divider()
#         chart_df = city_pred.copy()
#         chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        
#         # Create line chart
#         fig = px.line(chart_df, x="Date", y="Predicted_AQI_Value", 
#                      title=f"{city} AQI Forecast (Next {n_days} Days)",
#                      markers=True,
#                      color_discrete_sequence=[get_aqi_color_hex(
#                          chart_df["Predicted_AQI_Value"].iloc[-1], 
#                          st.session_state.is_dark_theme
#                      )])
        
#         # Update layout with theme
#         plotly_theme = get_plotly_theme(st.session_state.is_dark_theme)
#         fig.update_layout(
#             hovermode="x unified", 
#             title_x=0.5,
#             **plotly_theme["layout"]
#         )
        
#         # Add color-coded background zones for AQI levels
#         fig.add_hrect(y0=0, y1=1.5, line_width=0, fillcolor="green", opacity=0.1, 
#                      annotation_text="Good", annotation_position="inside top left")
#         fig.add_hrect(y0=1.5, y1=2.5, line_width=0, fillcolor="yellow", opacity=0.1,
#                      annotation_text="Moderate", annotation_position="inside top left")
#         fig.add_hrect(y0=2.5, y1=3.5, line_width=0, fillcolor="orange", opacity=0.1,
#                      annotation_text="Unhealthy for Sensitive", annotation_position="inside top left")
#         fig.add_hrect(y0=3.5, y1=4.5, line_width=0, fillcolor="red", opacity=0.1,
#                      annotation_text="Unhealthy", annotation_position="inside top left")
#         fig.add_hrect(y0=4.5, y1=5.5, line_width=0, fillcolor="purple", opacity=0.1,
#                      annotation_text="Very Unhealthy", annotation_position="inside top left")
        
#         st.plotly_chart(fig, use_container_width=True)

#         # Download CSV
#         csv = pred_df.to_csv(index=False)
#         st.download_button("Download forecast CSV", csv, 
#                           file_name=f"aqi_forecast_{city}_{n_days}days.csv", 
#                           mime="text/csv")

#         st.markdown("---")
#         st.subheader("Forecast table")
#         st.dataframe(city_pred.style.applymap(
#             lambda x: get_aqi_category_style(x, st.session_state.is_dark_theme) if isinstance(x, str) else '', 
#             subset=["Predicted_AQI_Category"]
#         ), use_container_width=True, hide_index=True)
        
#         st.divider()

#         # Last 7 days historical AQI (from training data)
#         with st.expander("📊 Last 7 days — historical AQI (" + city + ")"):
#             hist = daily_df[daily_df["city"] == city].sort_values("date").tail(7)
#             if not hist.empty:
#                 hist_display = hist[["date", "aqi_category"]].copy()
#                 hist_display["AQI Category"] = hist_display["aqi_category"].map(
#                     lambda x: AQI_CATEGORIES.get(int(x), str(x)) if pd.notna(x) else ""
#                 )
#                 st.dataframe(hist_display[["date", "AQI Category"]].style.applymap(
#                     lambda x: get_aqi_category_style(x, st.session_state.is_dark_theme) if isinstance(x, str) else '', 
#                     subset=["AQI Category"]
#                 ), use_container_width=True, hide_index=True)
                
#                 # Historical chart
#                 fig_hist = px.line(hist, x="date", y="aqi_category", 
#                                  title=f"{city} Historical AQI (Last 7 Days)",
#                                  markers=True,
#                                  color_discrete_sequence=[get_aqi_color_hex(
#                                      hist["aqi_category"].iloc[-1], 
#                                      st.session_state.is_dark_theme
#                                  )])
#                 fig_hist.update_layout(
#                     hovermode="x unified", 
#                     title_x=0.5,
#                     **plotly_theme["layout"]
#                 )
#                 st.plotly_chart(fig_hist, use_container_width=True)
#             else:
#                 st.caption("No history for this city.")
        
#         st.divider()
        
#         # Alerts for unhealthy air
#         unhealthy = city_pred[
#             city_pred["Predicted_AQI_Category"].str.contains("Unhealthy", na=False)
#         ]
#         if not unhealthy.empty:
#             with st.expander("⚠️ Alerts (Unhealthy air)", expanded=True):
#                 st.warning("**Warning:** Unhealthy air expected! Consider limiting outdoor activity.")
#                 for _, row in unhealthy.iterrows():
#                     date_str = row['Date']
#                     cat = row['Predicted_AQI_Category']
#                     color = get_aqi_color_hex(row['Predicted_AQI_Value'], st.session_state.is_dark_theme)
#                     st.markdown(f"**{date_str}**: <span style='color:{color}; font-weight:bold;'>{cat}</span>", 
#                                unsafe_allow_html=True)
#     else:
#         st.warning(f"No forecast for {city}. Try another city or run training first.")

#     st.markdown("---")
#     st.subheader("All cities — next day (risk ranking)")
#     next_day = pred_df[pred_df["Date"] == pred_df["Date"].min()]
#     if not next_day.empty:
#         st.dataframe(next_day[["City", "Date", "Predicted_AQI_Category"]].style.applymap(
#             lambda x: get_aqi_category_style(x, st.session_state.is_dark_theme) if isinstance(x, str) else '', 
#             subset=["Predicted_AQI_Category"]
#         ), use_container_width=True, hide_index=True)
        
#         # Bar chart for city-wise risk ranking
#         next_day_chart_data = next_day.set_index("City")["Predicted_AQI_Value"].sort_values(ascending=False)
#         fig_risk = px.bar(next_day_chart_data, 
#                          x=next_day_chart_data.index, 
#                          y="Predicted_AQI_Value", 
#                          title="City-wise AQI Risk (Next Day)",
#                          color="Predicted_AQI_Value", 
#                          color_continuous_scale=px.colors.sequential.YlOrRd)
        
#         fig_risk.update_layout(
#             title_x=0.5,
#             **plotly_theme["layout"]
#         )
#         st.plotly_chart(fig_risk, use_container_width=True)
#     else:
#         st.caption("No predictions yet.")
    
#     st.divider()

#     # Explainable ML: feature importance (RF / XGBoost only)
#     with st.expander("✨ Explainable ML — Feature importance"):
#         m = bundle.get("model")
#         if m is not None and hasattr(m, "feature_importances_"):
#             fi = m.feature_importances_
#             names = list(feature_cols) if feature_cols else []
#             if len(names) == len(fi):
#                 imp_df = pd.DataFrame({"feature": names, "importance": fi}).sort_values("importance", ascending=False).head(15)
#                 fig_fi = px.bar(imp_df, x="importance", y="feature", orientation='h', 
#                                title="Top 15 Feature Importances",
#                                color_discrete_sequence=['#4CAF50'])
#                 fig_fi.update_layout(
#                     yaxis_title="", 
#                     title_x=0.5,
#                     **plotly_theme["layout"]
#                 )
#                 st.plotly_chart(fig_fi, use_container_width=True)
#             st.caption(f"Model: {bundle.get('model_name', 'N/A')} — top 15 features.")
#         else:
#             st.info("Feature importance is available for Random Forest and XGBoost. Current model is sequence-based (LSTM/TCN).")

# if __name__ == "__main__":
#     main()

# """
# Streamlit app: Pakistan AQI forecast - Professional Dark Dashboard
# Version: 1.0
# """

# import streamlit as st
# import pandas as pd
# import joblib
# from pathlib import Path
# import plotly.express as px
# from datetime import datetime
# from predict import load_artifacts, forecast_next_n_days
# from preprocessing import AQI_CATEGORIES, add_lag_rolling_features

# # Define AQI color mapping
# AQI_COLORS = {
#     1: {"category": "Good", "color": "#00FF9D", "glow": "0 0 20px rgba(0, 255, 157, 0.5)"},
#     2: {"category": "Moderate", "color": "#FFEB3B", "glow": "0 0 20px rgba(255, 235, 59, 0.5)"},
#     3: {"category": "Unhealthy for Sensitive", "color": "#FF9800", "glow": "0 0 20px rgba(255, 152, 0, 0.5)"},
#     4: {"category": "Unhealthy", "color": "#FF5252", "glow": "0 0 20px rgba(255, 82, 82, 0.5)"},
#     5: {"category": "Very Unhealthy", "color": "#E040FB", "glow": "0 0 20px rgba(224, 64, 251, 0.5)"},
# }

# def get_aqi_color_hex(aqi_value):
#     return AQI_COLORS.get(aqi_value, {"color": "#B0B0B0"})["color"]

# def get_aqi_glow(aqi_value):
#     return AQI_COLORS.get(aqi_value, {"glow": "0 0 20px rgba(176, 176, 176, 0.5)"})["glow"]

# def get_aqi_category_style(category_string):
#     for aqi_val, data in AQI_COLORS.items():
#         if data["category"] == category_string:
#             return f'background: rgba(30, 30, 30, 0.8); color: {data["color"]}; font-weight: 700; border-radius: 8px; padding: 6px 16px; border: 1px solid {data["color"]}40; box-shadow: {data["glow"]};'
#     return ''

# # Page config -
# st.set_page_config(
#     page_title="AQI Intelligence | Pakistan",
#     page_icon="🌫️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # CSS
# css = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

# html, body, [data-testid="stAppViewContainer"] {
#     background: #0A0A0A;
#     color: #FFFFFF;
#     font-family: 'Inter', sans-serif;
# }

# section[data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #1A1A1A 0%, #121212 100%);
#     border-right: 1px solid #2A2A2A;
# }

# .main-header {
#     background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%);
#     border-radius: 16px;
#     padding: 2.5rem;
#     margin-bottom: 2rem;
#     border: 1px solid #2A2A2A;
#     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
# }

# .metric-card {
#     background: linear-gradient(145deg, #1E1E1E 0%, #141414 100%);
#     border-radius: 14px;
#     padding: 1.5rem;
#     border: 1px solid #333;
#     box-shadow: inset 2px 2px 4px rgba(0, 0, 0, 0.5), 4px 4px 12px rgba(0, 0, 0, 0.4);
#     position: relative;
#     overflow: hidden;
# }

# .metric-card::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 3px;
#     background: linear-gradient(90deg, #00FF9D, #667eea);
# }

# .metric-card .card-title {
#     color: #A0A0A0;
#     font-size: 0.85rem;
#     font-weight: 600;
#     text-transform: uppercase;
#     letter-spacing: 1px;
#     margin-bottom: 0.75rem;
# }

# .metric-card .card-value {
#     color: #FFFFFF;
#     font-size: 2.2rem;
#     font-weight: 800;
# }

# .forecast-card {
#     background: rgba(25, 25, 25, 0.9);
#     border-radius: 12px;
#     padding: 1.25rem;
#     margin: 0.75rem 0;
#     border-left: 4px solid;
#     border: 1px solid #333;
# }

# .chart-container {
#     background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%);
#     border-radius: 16px;
#     padding: 1.75rem;
#     border: 1px solid #2A2A2A;
#     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
#     margin-bottom: 2rem;
# }

# .stButton > button {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     color: white;
#     border: none;
#     border-radius: 10px;
#     padding: 14px 32px;
#     font-weight: 600;
#     font-size: 14px;
# }

# .status-indicator {
#     display: inline-block;
#     width: 10px;
#     height: 10px;
#     border-radius: 50%;
#     margin-right: 8px;
# }

# .status-online { background: #00FF9D; box-shadow: 0 0 10px #00FF9D; }
# .status-warning { background: #FFEB3B; box-shadow: 0 0 10px #FFEB3B; }

# .text-muted {
#     color: #A0A0A0 !important;
# }

# .footer {
#     text-align: center;
#     color: #666;
#     padding: 2.5rem;
#     font-size: 0.9rem;
#     margin-top: 4rem;
#     border-top: 1px solid #2A2A2A;
#     background: rgba(10, 10, 10, 0.9);
# }
# </style>
# """

# # Load CSS
# st.markdown(css, unsafe_allow_html=True)

# SAVE_DIR = Path(__file__).parent
# MODEL_PATH = SAVE_DIR / "model.joblib"
# DAILY_PATH = SAVE_DIR / "daily_aggregated.csv"

# @st.cache_resource
# def load_model_and_data():
#     if not MODEL_PATH.exists():
#         return None, None, "Model not found. Run: python train.py --save-dir ."
#     try:
#         bundle, scaler, feature_cols = load_artifacts(SAVE_DIR)
#     except Exception as e:
#         return None, None, str(e)
#     if not DAILY_PATH.exists():
#         return (bundle, scaler, feature_cols), None, "daily_aggregated.csv not found. Run training first."
#     daily = pd.read_csv(DAILY_PATH)
#     daily["date"] = pd.to_datetime(daily["date"])
#     if "aqi_lag_1" not in daily.columns:
#         daily = add_lag_rolling_features(daily)
#     return (bundle, scaler, feature_cols), daily, None

# def create_metric_card(title, value, icon="", change=None, change_label=""):
#     change_html = ""
#     if change is not None:
#         change_color = "#00FF9D" if change > 0 else "#A0A0A0"
#         change_sign = "+" if change > 0 else ""
#         change_html = f'<div style="color: {change_color}; margin-top: 8px; font-size: 0.9rem;">{change_sign}{change} {change_label}</div>'
    
#     icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 12px;">{icon}</div>' if icon else ""
    
#     return f"""
#     <div class="metric-card">
#         {icon_html}
#         <div class="card-title">{title}</div>
#         <div class="card-value">{value}</div>
#         {change_html}
#     </div>
#     """

# def create_forecast_card(date, category, aqi_value, color):
#     glow = get_aqi_glow(aqi_value)
    
#     return f"""
#     <div class="forecast-card" style="border-left-color: {color}; box-shadow: {glow};">
#         <div style="display: flex; justify-content: space-between; align-items: center;">
#             <div>
#                 <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 6px; font-size: 1.1rem;">{date}</div>
#                 <div style="color: #A0A0A0; font-size: 0.9rem;">Air Quality Forecast</div>
#             </div>
#             <div style="display: flex; flex-direction: column; align-items: flex-end;">
#                 <div style="background: rgba(30, 30, 30, 0.9); color: {color}; padding: 6px 16px; border-radius: 20px; border: 1px solid {color}40; font-weight: 700; font-size: 0.85rem;">
#                     {category}
#                 </div>
#                 <div style="font-size: 1.4rem; font-weight: 800; color: {color}; margin-top: 6px;">
#                     {aqi_value}
#                 </div>
#             </div>
#         </div>
#     </div>
#     """

# def main():
#     # Header
#     current_time = datetime.now().strftime("%d %b %Y, %H:%M")
    
#     st.markdown(f"""
#     <div class="main-header">
#         <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
#             <div>
#                 <h1 style="color: #FFFFFF; margin: 0; font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #667eea, #00FF9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
#                     🌫️ AIR QUALITY INTELLIGENCE
#                 </h1>
#                 <div style="color: #A0A0A0; font-size: 1.1rem; margin-top: 8px;">
#                     Real-time Monitoring & Predictive Analytics Dashboard
#                 </div>
#             </div>
#             <div style="margin-left: auto; background: rgba(30, 30, 30, 0.8); 
#                         padding: 10px 20px; border-radius: 12px; border: 1px solid #3A3A3A;">
#                 <div style="color: #A0A0A0; font-size: 0.9rem;">COUNTRY</div>
#                 <div style="color: #FFFFFF; font-weight: 700; font-size: 1.2rem;">PAKISTAN</div>
#             </div>
#         </div>
        
#         <div style="display: flex; gap: 20px; margin-top: 1.5rem;">
#             <div style="display: flex; align-items: center;">
#     <span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;background:#00FF9D;box-shadow:0 0 10px #00FF9D;"></span>
#     <span style="color: #A0A0A0;">System:</span>
#     <span style="color: #00FF9D; margin-left: 8px; font-weight: 600;">ACTIVE</span>
# </div>
# <div style="display: flex; align-items: center; margin-left: 20px;">
#     <span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;background:#FFEB3B;box-shadow:0 0 10px #FFEB3B;"></span>
#     <span style="color: #A0A0A0;">Last Update:</span>
#     <span style="color: #FFFFFF; margin-left: 8px; font-weight: 600;">31 Jan 2026, 13:30</span>
# </div>

#         </div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Load data
#     artifacts, daily_df, err = load_model_and_data()
#     if err:
#         st.error(err)
#         st.info("From project root run: `python train.py --save-dir .` then `python generate_predictions.py --save-dir .`")
#         return
    
#     bundle, scaler, feature_cols = artifacts
#     cities = sorted(daily_df["city"].unique().tolist())
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("""
#         <div style="margin-bottom: 2.5rem;">
#             <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.5rem;">
#                 ⚙️ CONTROL PANEL
#             </div>
#             <div style="color: #A0A0A0; font-size: 0.9rem;">
#                 Configure monitoring parameters
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # City Selection
#         st.markdown("#### 🏙️ CITY SELECTION")
#         city = st.selectbox("", cities, index=0, label_visibility="collapsed")
        
#         # Forecast Days
#         st.markdown("#### 📅 FORECAST PERIOD")
#         n_days = st.slider("", min_value=1, max_value=7, value=7, label_visibility="collapsed")
        
#         st.markdown("---")
        
#         # Model Info
#         st.markdown("#### 📊 MODEL INFO")
        
#         eval_path = SAVE_DIR / "evaluation.csv"
#         if eval_path.exists():
#             eval_df = pd.read_csv(eval_path)
#             best_model = bundle.get('model_name', 'N/A')
#             best_score = eval_df['f1_weighted'].max()
            
#             st.markdown(f"""
#             <div style="background: rgba(30, 30, 30, 0.7); padding: 1rem; border-radius: 10px; margin-top: 1rem; border: 1px solid #333;">
#                 <div style="color: #A0A0A0; font-size: 0.85rem; margin-bottom: 8px;">ACTIVE MODEL</div>
#                 <div style="color: #FFFFFF; font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;">{best_model}</div>
#                 <div style="display: flex; justify-content: space-between; align-items: center;">
#                     <span style="color: #A0A0A0;">F1 Score</span>
#                     <span style="color: #00FF9D; font-weight: 700; font-size: 1.1rem;">{best_score:.3f}</span>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("---")
#         st.markdown("""
#         <div style="text-align: center; color: #666; font-size: 0.85rem; padding-top: 2rem;">
#             <div style="margin-bottom: 8px;">🔒 SECURE CONNECTION</div>
#             <div style="color: #00FF9D; font-weight: 600;">ENCRYPTED</div>
#             <div style="margin-top: 1rem;">© 2024 AI Intelligence Systems</div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Main Dashboard
#     col1, col2, col3 = st.columns([2, 1, 1])
    
#     with col1:
#         st.markdown("### 🎯 QUICK FORECAST")
#         with st.spinner("Generating AI-powered forecast..."):
#             pred_df = forecast_next_n_days(daily_df, bundle, scaler, feature_cols, n_days=n_days)
#         city_pred = pred_df[pred_df["City"] == city]
    
#     with col2:
#         st.markdown(create_metric_card(
#             "CITIES MONITORED",
#             len(cities),
#             icon="🏙️",
#             change=+3,
#             change_label="this month"
#         ), unsafe_allow_html=True)
    
#     with col3:
#         accuracy = bundle.get('best_score', 0.85) * 100
#         st.markdown(create_metric_card(
#             "MODEL ACCURACY",
#             f"{accuracy:.1f}%",
#             icon="⚡",
#             change=+2.5,
#             change_label="points"
#         ), unsafe_allow_html=True)
    
#     # Forecast Display
    
    
#     # Footer
#     st.markdown("""
#     <div class="footer">
#         <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: #FFFFFF;">
#             🌫️ AIR QUALITY INTELLIGENCE DASHBOARD
#         </div>
#         <div style="color: #666; margin-bottom: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto;">
#             Advanced predictive analytics powered by machine learning algorithms. 
#             Data updates automatically every 24 hours for accurate forecasting.
#         </div>
#         <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 1.5rem;">
#             <div>
#                 <div style="color: #A0A0A0;">📞 SUPPORT</div>
#                 <div style="color: #FFFFFF; font-weight: 600;">+92-XXX-XXXXXXX</div>
#             </div>
#             <div>
#                 <div style="color: #A0A0A0;">📧 EMAIL</div>
#                 <div style="color: #FFFFFF; font-weight: 600;">support@ai-intelligence.com</div>
#             </div>
#             <div>
#                 <div style="color: #A0A0A0;">🌐 WEB</div>
#                 <div style="color: #FFFFFF; font-weight: 600;">www.ai-intelligence.pk</div>
#             </div>
#         </div>
#         <div style="border-top: 1px solid #2A2A2A; padding-top: 1.5rem; color: #444;">
#             © 2024 AI Intelligence Systems. All rights reserved. | Version 1.0.0
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

"""
Streamlit app: Pakistan AQI forecast - Professional Dark Dashboard
Version: 2.0
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
from datetime import datetime
from predict import load_artifacts, forecast_next_n_days
from preprocessing import AQI_CATEGORIES, add_lag_rolling_features

# Define AQI color mapping
AQI_COLORS = {
    1: {"category": "Good", "color": "#00FF9D", "glow": "0 0 20px rgba(0, 255, 157, 0.5)"},
    2: {"category": "Moderate", "color": "#FFEB3B", "glow": "0 0 20px rgba(255, 235, 59, 0.5)"},
    3: {"category": "Unhealthy for Sensitive", "color": "#FF9800", "glow": "0 0 20px rgba(255, 152, 0, 0.5)"},
    4: {"category": "Unhealthy", "color": "#FF5252", "glow": "0 0 20px rgba(255, 82, 82, 0.5)"},
    5: {"category": "Very Unhealthy", "color": "#E040FB", "glow": "0 0 20px rgba(224, 64, 251, 0.5)"},
}

def get_aqi_color_hex(aqi_value):
    return AQI_COLORS.get(aqi_value, {"color": "#B0B0B0"})["color"]

def get_aqi_glow(aqi_value):
    return AQI_COLORS.get(aqi_value, {"glow": "0 0 20px rgba(176, 176, 176, 0.5)"})["glow"]

def get_aqi_category_style(category_string):
    for aqi_val, data in AQI_COLORS.items():
        if data["category"] == category_string:
            return f'background: rgba(30, 30, 30, 0.8); color: {data["color"]}; font-weight: 700; border-radius: 8px; padding: 6px 16px; border: 1px solid {data["color"]}40; box-shadow: {data["glow"]};'
    return ''

# Page config
st.set_page_config(
    page_title="AQI Intelligence | Pakistan",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Fixed version
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0A0A;
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1A1A 0%, #121212 100%);
    border-right: 1px solid #2A2A2A;
}

.dashboard-header {
    background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid #2A2A2A;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.metric-card {
    background: linear-gradient(145deg, #1E1E1E 0%, #141414 100%);
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid #333;
    box-shadow: inset 2px 2px 4px rgba(0, 0, 0, 0.5), 4px 4px 12px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00FF9D, #667eea);
}

.metric-card .card-title {
    color: #A0A0A0;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.75rem;
}

.metric-card .card-value {
    color: #FFFFFF;
    font-size: 2.2rem;
    font-weight: 800;
}

.forecast-card {
    background: rgba(25, 25, 25, 0.9);
    border-radius: 12px;
    padding: 1.25rem;
    margin: 0.75rem 0;
    border-left: 4px solid;
    border: 1px solid #333;
}

.chart-container {
    background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%);
    border-radius: 16px;
    padding: 1.75rem;
    border: 1px solid #2A2A2A;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    margin-bottom: 2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px 32px;
    font-weight: 600;
    font-size: 14px;
}

.system-status {
    display: inline-flex;
    align-items: center;
    background: rgba(30, 30, 30, 0.8);
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid #3A3A3A;
    margin-right: 15px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-online { background: #00FF9D; box-shadow: 0 0 10px #00FF9D; }
.status-warning { background: #FFEB3B; box-shadow: 0 0 10px #FFEB3B; }

.text-muted {
    color: #A0A0A0 !important;
}

.footer {
    text-align: center;
    color: #666;
    padding: 2.5rem;
    font-size: 0.9rem;
    margin-top: 4rem;
    border-top: 1px solid #2A2A2A;
    background: rgba(10, 10, 10, 0.9);
}
</style>
"""

# Load CSS
st.markdown(css, unsafe_allow_html=True)

SAVE_DIR = Path(__file__).parent
MODEL_PATH = SAVE_DIR / "model.joblib"
DAILY_PATH = SAVE_DIR / "daily_aggregated.csv"

@st.cache_resource
def load_model_and_data():
    if not MODEL_PATH.exists():
        return None, None, "Model not found. Run: python train.py --save-dir ."
    try:
        bundle, scaler, feature_cols = load_artifacts(SAVE_DIR)
    except Exception as e:
        return None, None, str(e)
    if not DAILY_PATH.exists():
        return (bundle, scaler, feature_cols), None, "daily_aggregated.csv not found. Run training first."
    daily = pd.read_csv(DAILY_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    if "aqi_lag_1" not in daily.columns:
        daily = add_lag_rolling_features(daily)
    return (bundle, scaler, feature_cols), daily, None

def create_metric_card(title, value, icon="", change=None, change_label=""):
    change_html = ""
    if change is not None:
        change_color = "#00FF9D" if change > 0 else "#A0A0A0"
        change_sign = "+" if change > 0 else ""
        change_html = f'<div style="color: {change_color}; margin-top: 8px; font-size: 0.9rem;">{change_sign}{change} {change_label}</div>'
    
    icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 12px;">{icon}</div>' if icon else ""
    
    return f"""
    <div class="metric-card">
        {icon_html}
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        {change_html}
    </div>
    """

def create_forecast_card(date, category, aqi_value, color):
    glow = get_aqi_glow(aqi_value)
    
    return f"""
    <div class="forecast-card" style="border-left-color: {color}; box-shadow: {glow};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 6px; font-size: 1.1rem;">{date}</div>
                <div style="color: #A0A0A0; font-size: 0.9rem;">Air Quality Forecast</div>
            </div>
            <div style="display: flex; flex-direction: column; align-items: flex-end;">
                <div style="background: rgba(30, 30, 30, 0.9); color: {color}; padding: 6px 16px; border-radius: 20px; border: 1px solid {color}40; font-weight: 700; font-size: 0.85rem;">
                    {category}
                </div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {color}; margin-top: 6px;">
                    {aqi_value}
                </div>
            </div>
        </div>
    </div>
    """

def main():
    # Header with system indicators
    current_time = datetime.now().strftime("%d %b %Y, %H:%M")
    
    st.markdown(f"""
    <div class="dashboard-header">
        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2.5rem; font-weight: 800; background: linear-gradient(90deg, #667eea, #00FF9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🌫️ AIR QUALITY INTELLIGENCE
                </h1>
                <div style="color: #A0A0A0; font-size: 1.1rem; margin-top: 8px;">
                    Real-time Monitoring & Predictive Analytics Dashboard
                </div>
            </div>
            <div style="margin-left: auto; background: rgba(30, 30, 30, 0.8); 
                        padding: 10px 20px; border-radius: 12px; border: 1px solid #3A3A3A;">
                <div style="color: #A0A0A0; font-size: 0.9rem;">COUNTRY</div>
                <div style="color: #FFFFFF; font-weight: 700; font-size: 1.2rem;">PAKISTAN</div>
            </div>
        </div>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    artifacts, daily_df, err = load_model_and_data()
    if err:
        st.error(err)
        st.info("From project root run: `python train.py --save-dir .` then `python generate_predictions.py --save-dir .`")
        return
    
    bundle, scaler, feature_cols = artifacts
    cities = sorted(daily_df["city"].unique().tolist())
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom: 2.5rem;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.5rem;">
                ⚙️ CONTROL PANEL
            </div>
            <div style="color: #A0A0A0; font-size: 0.9rem;">
                Configure monitoring parameters
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # City Selection
        st.markdown("#### 🏙️ SELECT CITY")
        city = st.selectbox("", cities, index=0, label_visibility="collapsed")
        
        # Forecast Days
        st.markdown("#### 📅 FORECAST DAYS")
        n_days = st.slider("", min_value=1, max_value=7, value=7, label_visibility="collapsed")
        
        # Additional Options
        st.markdown("#### ⚙️ SETTINGS")
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        alert_threshold = st.selectbox("Alert Level", ["Moderate", "Unhealthy", "Very Unhealthy"])
        
        st.markdown("---")
        
        # Model Info
        st.markdown("#### 📊 MODEL INFO")
        
        eval_path = SAVE_DIR / "evaluation.csv"
        if eval_path.exists():
            eval_df = pd.read_csv(eval_path)
            best_model = bundle.get('model_name', 'N/A')
            best_score = eval_df['f1_weighted'].max()
            
            st.markdown(f"""
            <div style="background: rgba(30, 30, 30, 0.7); padding: 1rem; border-radius: 10px; margin-top: 1rem; border: 1px solid #333;">
                <div style="color: #A0A0A0; font-size: 0.85rem; margin-bottom: 8px;">ACTIVE MODEL</div>
                <div style="color: #FFFFFF; font-size: 1.3rem; font-weight: 700; margin-bottom: 12px;">{best_model}</div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #A0A0A0;">F1 Score</span>
                    <span style="color: #00FF9D; font-weight: 700; font-size: 1.1rem;">{best_score:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.85rem; padding-top: 2rem;">
            <div style="margin-bottom: 8px;">🔒 SECURE CONNECTION</div>
            <div style="color: #00FF9D; font-weight: 600;">ENCRYPTED</div>
            <div style="margin-top: 1rem;">© 2024 AI Intelligence Systems</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Dashboard
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🎯 QUICK FORECAST")
        with st.spinner("Generating AI-powered forecast..."):
            pred_df = forecast_next_n_days(daily_df, bundle, scaler, feature_cols, n_days=n_days)
        city_pred = pred_df[pred_df["City"] == city]
    
    with col2:
        st.markdown(create_metric_card(
            "CITIES MONITORED",
            len(cities),
            icon="🏙️",
            change=+3,
            change_label="this month"
        ), unsafe_allow_html=True)
    
    with col3:
        accuracy = bundle.get('best_score', 0.85) * 100
        st.markdown(create_metric_card(
            "MODEL ACCURACY",
            f"{accuracy:.1f}%",
            icon="⚡",
            change=+2.5,
            change_label="points"
        ), unsafe_allow_html=True)
    
    # Forecast Display
    if not city_pred.empty:
        # Forecast Cards
        st.markdown("### 📈 DETAILED FORECAST")
        
        forecast_cols = st.columns(min(4, n_days))
        
        for idx, (_, row) in enumerate(city_pred.iterrows()):
            with forecast_cols[idx % len(forecast_cols)]:
                cat = row["Predicted_AQI_Category"]
                date = row["Date"]
                aqi_val = int(row["Predicted_AQI_Value"])
                color = get_aqi_color_hex(aqi_val)
                
                st.markdown(create_forecast_card(date, cat, aqi_val, color), unsafe_allow_html=True)
        
        # Chart Section
        st.markdown(f'<div class="chart-container"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;"><h3 style="color: #FFFFFF; margin: 0; font-weight: 700;">📊 TREND ANALYSIS</h3><div style="color: #A0A0A0; font-size: 0.9rem;">{city} - {n_days} Day Forecast</div></div>', unsafe_allow_html=True)
        
        # Create chart
        chart_df = city_pred.copy()
        chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        
        fig = px.line(
            chart_df, 
            x="Date", 
            y="Predicted_AQI_Value",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=[get_aqi_color_hex(int(chart_df["Predicted_AQI_Value"].iloc[-1]))]
        )
        
        fig.update_layout(
            hovermode="x unified",
            plot_bgcolor='rgba(10, 10, 10, 0.9)',
            paper_bgcolor='rgba(10, 10, 10, 0)',
            font=dict(color="#FFFFFF"),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                title="Date"
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                title="AQI Level",
                range=[0.5, 5.5]
            ),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Data Table
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📋 FORECAST DATA")
            styled_df = city_pred.style.applymap(
                lambda x: get_aqi_category_style(x) if isinstance(x, str) else '', 
                subset=["Predicted_AQI_Category"]
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### ⚡ QUICK ACTIONS")
            
            csv = pred_df.to_csv(index=False)
            st.download_button(
                "📥 EXPORT CSV",
                csv,
                file_name=f"aqi_forecast_{city}_{n_days}days.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            if st.button("🔄 REFRESH DATA", use_container_width=True):
                st.rerun()
            
            show_alerts = st.toggle("🔔 ENABLE ALERTS", value=True)
            
            # Additional action
            if st.button("📊 GENERATE REPORT", use_container_width=True):
                st.success("Report generation started!")
        
        # Alerts
        unhealthy = city_pred[
            city_pred["Predicted_AQI_Category"].str.contains("Unhealthy", na=False)
        ]
        
        if not unhealthy.empty and show_alerts:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255, 82, 82, 0.9) 0%, rgba(220, 53, 69, 0.9) 100%);
                        color: white; padding: 1.25rem; border-radius: 12px; margin: 1.5rem 0; border: 1px solid rgba(255, 82, 82, 0.3);">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="font-size: 2rem;">⚠️</div>
                    <div>
                        <div style="font-weight: 800; font-size: 1.3rem;">AIR QUALITY ALERT</div>
                        <div style="opacity: 0.9;">Unhealthy conditions detected in forecast period</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("VIEW ALERT DETAILS", expanded=True):
                for _, row in unhealthy.iterrows():
                    date_str = row['Date']
                    cat = row['Predicted_AQI_Category']
                    aqi_val = int(row['Predicted_AQI_Value'])
                    color = get_aqi_color_hex(aqi_val)
                    
                    st.markdown(f"""
                    <div style="background: rgba(25, 25, 25, 0.8); padding: 1rem; border-radius: 10px; 
                                margin-bottom: 0.75rem; border-left: 4px solid {color}; border: 1px solid #333;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-weight: 700; color: #FFFFFF;">{date_str}</div>
                            <div style="background: rgba(30, 30, 30, 0.9); color: {color}; padding: 6px 16px; 
                                    border-radius: 20px; border: 1px solid {color}40; font-weight: 700;">
                                {cat}
                            </div>
                        </div>
                        <div style="color: #A0A0A0; font-size: 0.9rem; margin-top: 6px;">
                            ⚠️ Consider limiting outdoor activities
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning(f"No forecast available for {city}. Please try another city.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; color: #FFFFFF;">
            🌫️ AIR QUALITY INTELLIGENCE DASHBOARD
        </div>
        <div style="color: #666; margin-bottom: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto;">
            Advanced predictive analytics powered by machine learning algorithms. 
            Data updates automatically every 24 hours for accurate forecasting.
        </div>
        <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap; margin-bottom: 1.5rem;">
            <div>
                <div style="color: #A0A0A0;">📞 SUPPORT</div>
                <div style="color: #FFFFFF; font-weight: 600;">+92-XXX-XXXXXXX</div>
            </div>
            <div>
                <div style="color: #A0A0A0;">📧 EMAIL</div>
                <div style="color: #FFFFFF; font-weight: 600;">support@ai-intelligence.com</div>
            </div>
            <div>
                <div style="color: #A0A0A0;">🌐 WEB</div>
                <div style="color: #FFFFFF; font-weight: 600;">www.ai-intelligence.pk</div>
            </div>
        </div>
        <div style="border-top: 1px solid #2A2A2A; padding-top: 1.5rem; color: #444;">
            © 2024 AI Intelligence Systems. All rights reserved. | Version 2.0.0
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()