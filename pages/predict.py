import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
st.title(":mag: Nutrient Forecast Viewer")
# Sidebar filters
nutrients = ["carbohydrates", "fiber", "protein", "fat"]
models = ["random_forest", "xgboost", "lstm"]
selected_nutrient = st.selectbox("Select Nutrient", nutrients)
selected_model = st.radio("Select Model", models, horizontal=True)
# Build file paths
csv_path = f"data/forecast/{selected_nutrient}_{selected_model}_forecast.csv"
png_path = f"data/forecast/{selected_nutrient}_{selected_model}_forecast.png"
# Display forecast table
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.subheader(":bar_chart: 8-Week Forecast Table")
    st.dataframe(df)
else:
    st.error("Forecast CSV not found.")
# Display graph
if os.path.exists(png_path):
    st.subheader(":chart_with_upwards_trend: Forecast Graph")
    st.image(png_path, use_column_width=True)
else:
    st.warning("Graph image not available.")