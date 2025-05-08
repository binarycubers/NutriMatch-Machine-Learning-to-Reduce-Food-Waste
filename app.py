# 🌐 Streamlit Multi-Page App Refactor - Clean Starter Structure

# This file (app.py) is the entry point
import streamlit as st
from PIL import Image

# --- General Config ---
st.set_page_config(
    page_title="Nutrition Forecast Dashboard",
    layout="wide",
    page_icon="🥦"
)

# with st.sidebar:
#     st.markdown('<div class="sidebar-logo"><img src="https://img.icons8.com/emoji/96/broccoli-emoji.png" alt="logo"></div>', unsafe_allow_html=True)
#     st.markdown('<div class="sidebar-title">Nutrition App</div>', unsafe_allow_html=True)


# --- Global CSS Styling (Dark + Vibrant) ---
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


# --- Welcome Page Body (Optional welcome message) ---
st.title("🌿 Nutrition Forecast Dashboard")
st.markdown("""
Welcome to the multipage AI-based nutrition forecasting tool.

Use the sidebar to navigate:
- **Home**: Project overview
- **Predict**: Using ML models to forecast nutrient wastes for each model; for instance Random Forest, XGBoost, and LSTM.
- **Visualize**: Explore trends and plots
- **Upload**: Submit your own food log for prediction
""")
