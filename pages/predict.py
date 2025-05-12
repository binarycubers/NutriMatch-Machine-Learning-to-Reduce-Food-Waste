import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
import keras.losses

# --- Page Config ---
st.set_page_config(
    page_title="Prediction - HealthFusion",
    layout="wide",
    page_icon="🧪"
)

# --- Global CSS Styling (Dark + Vibrant) ---
st.markdown("""
    <style>
    /* Backgrounds */
    .stApp, .main {
        background-color: #0F0F1A;
        color: #FFFFFF;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1B1B2F;
        color: white;
        padding-top: 2rem;
    }
    /* Buttons */
    .stButton > button {
        background-color: #4EF037;
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #6CFF57;
    }
    /* Headings + accents */
    h1, h2, h3, h4 {
        color: #00FFE1;
    }
    a {
        color: #4EF037;
    }
    [data-testid="stSidebarNav"]::before {
    content: "HealthFusion";
    display: flex;
    flex-direction: column;
    align-items: center;
    font-weight: bold;
    font-size: 1.8rem;
    color: #00ffe1;
    margin-top: -40px;
    margin-bottom: 20px;
    padding-top: 60px;
    background-image: url('https://img.icons8.com/external-flat-icons-inmotus-design/67/external-Health-Technology-flat-icons-inmotus-design.png');
    background-repeat: no-repeat;
    background-size: 55px;
    background-position: top center;
    height: 100px;
}
    </style>
""", unsafe_allow_html=True)

# --- Page Title ---
st.title("📈 Predict Nutrient Waste (Model-Based Forecast)")

# --- Dropdowns for Model and Nutrient ---
model_choice = st.selectbox("Choose model", ["Random Forest", "XGBoost", "LSTM"])
nutrient_choice = st.selectbox("Select nutrient", ["carbohydrates", "protein", "fat", "fiber"])

model_dir = "models"
data_path = f"data/engineered/{nutrient_choice}_lagged.csv"

try:
    df = pd.read_csv(data_path)
    X_test = df.drop(["target", "date"], axis=1, errors="ignore").tail(8)

    if model_choice == "LSTM":
        lstm_model = load_model(f"{model_dir}/{nutrient_choice}_lstm.h5", compile=False)
        X_input = np.array(X_test)[..., np.newaxis]
        y_pred = lstm_model.predict(X_input)
    else:
        model_file = f"{model_dir}/{nutrient_choice}_{model_choice.lower().replace(' ', '_')}.pkl"
        st.write(f"🔍 Loading model: `{model_file}`")

        if not os.path.exists(model_file):
            st.error(f"❌ Model file not found: `{model_file}`")
            st.stop()

        model = joblib.load(model_file)
        y_pred = model.predict(X_test)

    # --- Load and Display Forecast Graph ---
    chart_path = f"graphs/{model_choice.lower().replace(' ', '_')}/{nutrient_choice}_{model_choice.lower().replace(' ', '_')}_forecast.png"
    if os.path.exists(chart_path):
        st.image(chart_path, caption=f"{model_choice} Forecast for {nutrient_choice.title()}", width=800)
    else:
        st.warning("Forecast graph not found. Please generate it using the chart generation script.")

    # --- Load and Display Model Accuracy ---
    score_file = f"data/forecast/{model_choice.lower().replace(' ', '_')}_model_scores.csv"
    if os.path.exists(score_file):
        scores_df = pd.read_csv(score_file)
        row = scores_df[scores_df["nutrient"] == nutrient_choice]
        if not row.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MSE", f"{row.iloc[0]['mse']:.2f}")
            with col2:
                st.metric("RMSE", f"{row.iloc[0]['rmse']:.2f}")
        else:
            st.warning(f"No score data found for nutrient '{nutrient_choice}' in score file.")
    else:
        st.warning(f"Score file not found: {score_file}")

    # --- Display forecasted values ---
    forecast_df = pd.DataFrame({
        "Week": list(range(1, 9)),
        f"{nutrient_choice.title()} Forecast": y_pred.flatten()
    })
    st.subheader(f"📊 Forecasted {nutrient_choice.title()} for Next 8 Weeks")
    st.dataframe(forecast_df, use_container_width=True)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{nutrient_choice}_{model_choice.lower().replace(' ', '_')}_forecast.csv")
    forecast_df.to_csv(result_file, index=False)
    st.success(f"📁 Forecast saved to: `{result_file}`")

except FileNotFoundError:
    st.error(f"❌ Missing data or model for {nutrient_choice}. Ensure preprocessing and training are complete.")
