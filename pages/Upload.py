import streamlit as st
import pandas as pd
import pickle
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

st.markdown("## :inbox_tray: Upload & Predict on Your CSV")

# Upload section
uploaded_file = st.file_uploader("Upload your weekly nutrient CSV file", type=["csv"])
model_choice = st.selectbox("Choose model", ["Random Forest", "XGBoost", "LSTM"])
nutrient_choice = st.selectbox("Select nutrient", ["carbohydrates", "protein", "fat", "fiber"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(":open_file_folder: Uploaded Data:")
        st.dataframe(df)

        # Validate structure
        required_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_4']
        if not all(col in df.columns for col in required_cols):
            st.error("Uploaded file is missing required lag columns: lag_1 to lag_4")
            st.stop()

        model_dir = "models"
        X_input = np.array(df[required_cols])

        # Load model
        if model_choice == "LSTM":
            model_path = os.path.join(model_dir, f"{nutrient_choice}_lstm_model.h5")
            model = load_model(model_path)
            X_input = X_input[..., np.newaxis]  # reshape for LSTM
        else:
            model_type = "random_forest" if model_choice == "Random Forest" else "xgboost"
            model_path = os.path.join(model_dir, f"{nutrient_choice}_{model_type}.pkl")
            with open(model_path, "rb") as f:
                model = joblib.load(model_path)

        # Predict
        if hasattr(model, "predict"):
            y_pred = model.predict(X_input)
        else:
            st.error(f"Loaded model doesn't support prediction. Model: {model_path}")
            st.stop()

        # Results
        forecast_df = pd.DataFrame({
            "Index": list(range(1, len(y_pred)+1)),
            f"Predicted {nutrient_choice.title()}": y_pred.flatten()
        })
        st.subheader("Prediction Results")
        st.dataframe(forecast_df, use_container_width=True)

        # Download
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{nutrient_choice}_{model_choice}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f":x: Error processing file: {e}")