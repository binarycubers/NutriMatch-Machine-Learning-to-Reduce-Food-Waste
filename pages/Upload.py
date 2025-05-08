# pages/Predict.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from keras.models import load_model

# Apply consistent styling across pages
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #00f5d4;
    }
    body {
        background-color: #0d1117;
        color: #ffffff;
    }
    .stSelectbox > div:first-child {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("## 🔮 Predict Nutrient Waste")

# Dropdowns for model and nutrient
model_choice = st.selectbox("Choose model", ["Random Forest", "XGBoost", "LSTM"])
nutrient_choice = st.selectbox("Select nutrient", ["carbohydrates", "protein", "fat", "fiber"])

model_dir = "models"
data_path = f"data/engineered/{nutrient_choice}_lagged.csv"

try:
    df = pd.read_csv(data_path)
    X_test = df.drop("target", axis=1).tail(8)

    if model_choice == "LSTM":
        model = load_model(f"{model_dir}/{nutrient_choice}_lstm_model.h5")
        X_input = np.array(X_test)[..., np.newaxis]  # Reshape for LSTM
        y_pred = model.predict(X_input)
    else:
        model_type = "random_forest" if model_choice == "Random Forest" else "xgboost"
        with open(f"{model_dir}/{nutrient_choice}_{model_type}.pkl", "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)

    # Display forecasted values
    st.subheader(f"Forecasted {nutrient_choice.title()} (Next 8 Weeks)")
    forecast_df = pd.DataFrame({"Week": list(range(1, 9)), f"{nutrient_choice.title()} Forecast": y_pred.flatten()})
    st.dataframe(forecast_df, use_container_width=True)

except FileNotFoundError:
    st.error(f"Missing data or model for {nutrient_choice}. Ensure preprocessing and training are complete.")
