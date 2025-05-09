import streamlit as st
import pandas as pd
import numpy as np
import joblib 
from keras.models import load_model
import os

# Apply consistent styling across pages
st.markdown("""
    <style>
    /* Backgrounds */
    .stApp, .main {
        background-color: #0f0f1a;
        color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1b1b2f;
        color: white;
        padding-top: 2rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4ef037;
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #6cff57;
    }

    /* Headings + accents */
    h1, h2, h3, h4 {
        color: #00ffe1;
    }

    a {
        color: #4ef037;
    }
            
    [data-testid="stSidebarNav"]::before {
    content: "Nutrition App";
    display: flex;
    flex-direction: column;
    align-items: center;
    font-weight: bold;
    font-size: 1.2rem;
    color: #4ef037;
    margin-top: -20px;
    margin-bottom: 20px;
    padding-top: 60px;
    background-image: url('https://img.icons8.com/emoji/96/broccoli-emoji.png');
    background-repeat: no-repeat;
    background-size: 60px;
    background-position: top center;
    height: 100px;
}
    </style>
""", unsafe_allow_html=True)

st.markdown("## :crystal_ball: Predict Nutrient Waste")

# Dropdowns for model and nutrient
model_choice = st.selectbox("Choose model", ["Random Forest", "XGBoost", "LSTM"])
nutrient_choice = st.selectbox("Select nutrient", ["carbohydrates", "protein", "fat", "fiber"])

model_dir = "models"
data_path = f"data/engineered/{nutrient_choice}_lagged.csv"

try:
    df = pd.read_csv(data_path)
    X_test = df.drop("target", axis=1).tail(8)

    if model_choice == "LSTM":
        lstm_model = load_model(f"{model_dir}/{nutrient_choice}_lstm_model.h5")
        X_input = np.array(X_test)[..., np.newaxis]  # Reshape for LSTM
        y_pred = lstm_model.predict(X_input)

    elif model_choice == "XGBoost":
        model_path = os.path.join(model_dir, f"{nutrient_choice}_xgboost.pkl")
        st.write(f":mag: Trying to load model from: `{model_path}`")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = joblib.load(model_path)
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                st.error(":warning: Loaded XGBoost object is not a valid model with 'predict' method.")
                st.stop()
        else:
            st.error(f":x: Model file not found: `{model_path}`")
            st.stop()

    elif model_choice == "Random Forest":
        model_path = os.path.join(model_dir, f"{nutrient_choice}_random_forest.pkl")
        st.write(f":mag: Trying to load model from: `{model_path}`")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = joblib.load(model_path)
            if hasattr(model, "predict"):
                y_pred = model.predict(X_test)
            else:
                st.error(":warning: Loaded Random Forest object is not a valid model with 'predict' method.")
                st.stop()
        else:
            st.error(f":x: Model file not found: `{model_path}`")
            st.stop()

    # Display forecasted values
    forecast_df = pd.DataFrame({
        "Week": list(range(1, 9)),
        f"{nutrient_choice.title()} Forecast": y_pred.flatten()
    })
    st.subheader(f"Forecasted {nutrient_choice.title()} (Next 8 Weeks)")
    st.dataframe(forecast_df, use_container_width=True)

    # Save to CSV in results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{nutrient_choice}_{model_choice.lower().replace(' ', '_')}_forecast.csv")
    forecast_df.to_csv(result_file, index=False)
    st.success(f"Saved forecast to: {result_file}")

except FileNotFoundError:
    st.error(f"Missing data or model for {nutrient_choice}. Ensure preprocessing and training are complete.")
