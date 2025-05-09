import streamlit as st
import pandas as pd
import plotly.express as px
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

st.markdown("## 📈 Visualize Forecast Results")

# Inputs
nutrient = st.selectbox("Select Nutrient to Visualize", ["carbohydrates", "protein", "fat", "fiber"])
model = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM"])

# Results path
results_dir = "results"
result_file = os.path.join(results_dir, f"{nutrient}_{model.lower().replace(' ', '_')}_forecast.csv")

if os.path.exists(result_file):
    df = pd.read_csv(result_file)
    st.plotly_chart(
        px.line(df, x="Week", y=df.columns[1], title=f"{model} Forecast for {nutrient.title()}"),
        use_container_width=True
    )
else:
    st.warning(f"Forecast file not found: {result_file}. Please run predictions first.")
