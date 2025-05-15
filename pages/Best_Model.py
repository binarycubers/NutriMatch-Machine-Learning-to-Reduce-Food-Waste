import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Visualize - HealthFusion",
    layout="wide",
    page_icon="🧪"
)


# Apply consistent styling across pages
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

st.markdown("## 🥇 Best Model Forecast: LSTM")

st.write("""
LSTM (Long Short-Term Memory) has been identified as the best-performing model based on overall accuracy and performance across nutrient categories.
Below you can select a nutrient and view its detailed forecast including actual vs. predicted values for both training and testing periods.
""")

nutrients = ["carbohydrates", "fiber", "protein", "fat"]
nutrient_choice = st.selectbox("Select Nutrient", nutrients)

forecast_path = f"data/forecast/{nutrient_choice}_lstm_forecast.csv"

if not os.path.exists(forecast_path):
    st.warning(f"Forecast data not found for {nutrient_choice} using LSTM.")
    st.stop()

# Load forecast data
df = pd.read_csv(forecast_path)
df["date"] = pd.to_datetime(df["date"])
score_path = "data/forecast/lstm_model_scores.csv"
scores = None
if os.path.exists(score_path):
    score_df = pd.read_csv(score_path)
    scores = score_df[score_df["nutrient"] == nutrient_choice].squeeze()


# Plot chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["date"], df["actual"], label="Actual Waste", color="black", linewidth=2.5, marker='o')
if "predicted_train" in df:
    ax.plot(df["date"], df["predicted_train"], label="Predicted Waste (Train)", linestyle="--", color="orange", linewidth=2)
if "predicted_test" in df:
    ax.plot(df["date"], df["predicted_test"], label="Predicted Waste (Test)", linestyle="-.", color="cyan", linewidth=2)

ax.set_title(f"Forecast for {nutrient_choice.title()} using LSTM", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Waste Quantity")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
st.pyplot(fig, clear_figure=True, use_container_width=True)

# Show metrics
st.markdown("### 🧠 Model Evaluation Metrics (LSTM)")
if scores is not None:
    st.markdown(f"- **Training MSE**: `{scores.train_mse:.4f}`")
    st.markdown(f"- **Training RMSE**: `{scores.train_rmse:.4f}`")
    st.markdown(f"- **Testing MSE**: `{scores.test_mse:.4f}`")
    st.markdown(f"- **Testing RMSE**: `{scores.test_rmse:.4f}`")
else:
    st.info("Score data for LSTM not found.")


# Show next 8 week forecast table
st.markdown(f"### \U0001F4C8 Forecasted {nutrient_choice.title()} for Next 8 Weeks")
next_8 = df[["date", "predicted_test"]].dropna().reset_index(drop=True)
next_8.columns = ["Date", f"{nutrient_choice.title()} Forecast"]
st.dataframe(next_8, use_container_width=True)
