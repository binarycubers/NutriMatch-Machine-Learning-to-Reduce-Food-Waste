import streamlit as st
st.title(":house: Project Overview")
st.markdown("""
This project aims to reduce food waste by predicting nutrient loss using machine learning models.
You can explore 8-week forecasts based on:
- :repeat: Weekly aggregated food waste data
- :robot_face: Models: Random Forest, XGBoost (LSTM coming soon!)
- :chart_with_upwards_trend: Visualized predictions for each nutrient (carbs, fats, protein, fiber)
Use the **Predict** page to interact with model outputs.
""")
st.info("All models were trained using historical weekly nutrient data with lag-based feature engineering.")