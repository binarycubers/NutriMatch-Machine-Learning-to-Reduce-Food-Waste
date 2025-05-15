import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- Streamlit Page ---
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
st.pyplot(fig, use_container_width=False)
