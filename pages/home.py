import streamlit as st
from PIL import Image


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
st.title("🌱 Nutrition Forecast Dashboard")

st.markdown("""
Welcome to the Nutrition Forecast Dashboard!  
Use this platform to:
- 📊 View and analyze historical nutrient waste data.
- 🔮 Predict future waste using ML models like Random Forest, XGBoost, and LSTM.
- 📤 Upload your own food waste data.
- 📈 Visualize trends across weeks.

---
""")

try:
    image = Image.open("info-images/dashboard_cover.png")
    st.image(image, use_column_width=True)
except FileNotFoundError:
    st.warning("⚠️ Cover image not found. Make sure `info-images/dashboard_cover.png` exists.")
