import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
    .st-emotion-cache-1v0mbdj{
        width: 90%;}
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

# --- Stock Analysis PieChart section---
st.markdown("## 🍕 Stock Analysis - Top Wasted Items")
st.write("""
This chart helps identify which items are most frequently wasted by total quantity.
Use the slider to control how many top items are shown in the pie chart.
""")

# --- Parameters ---
data_path = "data/processed/daily_nutrients.csv"
raw_item_path = "data/raw/Merged_ItemList.csv"
out_path = "graphs/stock_analysis_pie.png"
top_n_default = 10

# --- Load Merged Data ---
if not os.path.exists(raw_item_path):
    st.error("Merged_ItemList.csv not found in raw data directory.")
    st.stop()

item_df = pd.read_csv(raw_item_path)
item_df.columns = [c.strip().lower() for c in item_df.columns]

# --- Aggregate Quantity by Item Description ---
agg = item_df.groupby("item description")["qty"].sum().reset_index()
agg = agg.sort_values("qty", ascending=False).reset_index(drop=True)

# --- Prompt for top N input ---
top_n_input = st.slider("Select number of top items to display:", min_value=5, max_value=20, value=top_n_default)
agg_top = agg.head(top_n_input)

# --- Plot Pie Chart ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(agg_top["qty"], labels=agg_top["item description"], autopct="%1.1f%%", startangle=140)
ax.set_title(f"Top {top_n_input} Wasted Items by Quantity", fontsize=14, fontweight='bold')
container = st.container()
with container:
    st.pyplot(fig, use_container_width=False)
    st.markdown("""
        <style>
        .element-container:has(canvas) {
            # max-width: 500px !important;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)
