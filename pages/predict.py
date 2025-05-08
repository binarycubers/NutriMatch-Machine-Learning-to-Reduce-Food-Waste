import streamlit as st
import pandas as pd
import pickle
import numpy as np
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
        model_path = os.path.join(model_dir, f"{nutrient_choice}_{model_type}.pkl")
        st.write(f"🔍 Trying to load model from: `{model_path}`")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(X_test)
        else:
            st.error(f"❌ Model file not found: `{model_path}`")
            st.stop()

    # Display forecasted values
    st.subheader(f"Forecasted {nutrient_choice.title()} (Next 8 Weeks)")
    forecast_df = pd.DataFrame({"Week": list(range(1, 9)), f"{nutrient_choice.title()} Forecast": y_pred.flatten()})
    st.dataframe(forecast_df, use_container_width=True)

except FileNotFoundError:
    st.error(f"Missing data or model for {nutrient_choice}. Ensure preprocessing and training are complete.")
