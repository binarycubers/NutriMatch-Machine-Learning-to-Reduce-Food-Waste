import streamlit as st
from PIL import Image
import os

# --- General Config ---
st.set_page_config(
    page_title="HealthFusion Dashboard",
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

# --- Home Page Content ---
st.title("🏠 Welcome to HealthFusion Dashboard")

st.markdown("""
HealthFusion is an AI-powered platform designed to monitor, forecast, and analyze food nutrient waste over time.
This dashboard offers:
- 📊 Nutrient waste forecasting using Random Forest, XGBoost, and LSTM models
- 📈 Visual insights into top wasted food items
- 🥫 Stock tracking and item movement analysis
""")

# Display Pie Chart from generated image
chart_path = "graphs/stock_analysis_pie.png"
if os.path.exists(chart_path):
    st.subheader("🍽️ Top 10 Stocked Items (Proportional View)")
    st.image(Image.open(chart_path), caption="Stock Analysis - Pie Chart", use_column_width=True)
else:
    st.warning("Stock analysis chart not found. Please generate it first.")
