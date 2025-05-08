# import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# st.title(":bar_chart: Compare Nutrient Forecasts")
# nutrients = ["carbohydrates", "fiber", "protein", "fat"]
# selected_nutrient = st.selectbox("Nutrient", nutrients)
# try:
#     df_rf = pd.read_csv(f"data/forecast/{selected_nutrient}_random_forest_forecast.csv")
#     df_xgb = pd.read_csv(f"data/forecast/{selected_nutrient}_xgboost_forecast.csv")
#     st.subheader(f":chart_with_upwards_trend: {selected_nutrient.capitalize()} Forecast Comparison")
#     plt.figure(figsize=(10, 4))
#     plt.plot(df_rf['Week'], df_rf['Prediction'], label='Random Forest', marker='o')
#     plt.plot(df_xgb['Week'], df_xgb['Prediction'], label='XGBoost', marker='x')
#     plt.xlabel("Week")
#     plt.ylabel("Predicted Amount")
#     plt.title(f"{selected_nutrient.capitalize()} Prediction - Model Comparison")
#     plt.legend()
#     st.pyplot(plt)
# except Exception as e:
#     st.warning("One or both model outputs not found for this nutrient.")