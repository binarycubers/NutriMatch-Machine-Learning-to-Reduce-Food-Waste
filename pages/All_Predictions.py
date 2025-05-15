import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
import numpy as np

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
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
import numpy as np

st.markdown("## ✨ Predict Nutrient Waste by Model")

nutrients = ["carbohydrates", "fiber", "protein", "fat"]
models = ["random_forest", "xgboost", "lstm"]

model_choice = st.selectbox("Select Model", models)
nutrient_choice = st.selectbox("Select Nutrient", nutrients)

forecast_path = f"data/forecast/{nutrient_choice}_{model_choice}_forecast.csv"
score_path = f"data/forecast/{model_choice}_model_scores.csv"

if not os.path.exists(forecast_path):
    st.warning(f"Forecast data not found for {nutrient_choice} ({model_choice}). Please train the model first.")
    st.stop()

# Load forecast CSV
df = pd.read_csv(forecast_path)

# Load scores
if os.path.exists(score_path):
    score_df = pd.read_csv(score_path)
    scores = score_df[score_df["nutrient"] == nutrient_choice].squeeze()
else:
    scores = None

# Plot graph
fig, ax = plt.subplots(figsize=(10, 5))
df["date"] = pd.to_datetime(df["date"])
ax.plot(df["date"], df["actual"], label="Actual Waste", color="black", linewidth=2.5, marker='o')
if "predicted_train" in df:
    ax.plot(df["date"], df["predicted_train"], label="Predicted Waste (Train)", linestyle="--", color="orange", linewidth=2)
if "predicted_test" in df:
    ax.plot(df["date"], df["predicted_test"], label="Predicted Waste (Test)", linestyle="-.", color="cyan", linewidth=2)

ax.set_title(f"Prediction Waste for {nutrient_choice.title()} ({model_choice.replace('_', ' ').title()})", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel(nutrient_choice.title())
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

# Show metrics
st.markdown("### \U0001F4DD Model Evaluation Metrics")
if scores is not None:
    st.markdown(f"- **Training MSE**: `{scores.train_mse:.4f}`")
    st.markdown(f"- **Training RMSE**: `{scores.train_rmse:.4f}`")
    st.markdown(f"- **Testing MSE**: `{scores.test_mse:.4f}`")
    st.markdown(f"- **Testing RMSE**: `{scores.test_rmse:.4f}`")
else:
    st.info("No score file found for this model.")

# Show forecasted values table
st.markdown(f"### \U0001F4C8 Forecasted {nutrient_choice.title()} for Next 8 Weeks")
next_8 = df[["date", "predicted_test"]].dropna().reset_index(drop=True)
next_8.columns = ["Date", f"{nutrient_choice.title()} Forecast"]
st.dataframe(next_8, use_container_width=True)


# --- Load and Display Top Wasted Items Chart ---
st.subheader("🥫 Top Wasted Food Items (Bar Chart)")
top_items_path = "graphs/top_wasted_items.png"

if os.path.exists(top_items_path):
    st.image(top_items_path, caption="Top Wasted Items - Figure 19", use_column_width=True)
else:
    st.warning("Top wasted items chart not found. Please generate it using the appropriate script.")