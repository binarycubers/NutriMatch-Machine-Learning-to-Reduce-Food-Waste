import streamlit as st
from PIL import Image
st.set_page_config(page_title="Nutrition Forecast Dashboard", layout="centered")
st.title(":green_salad: Nutrient Forecast Dashboard")
st.write("""
Welcome to the prediction dashboard for nutrient waste!
Use the sidebar to navigate and explore forecasts for each nutrient using different ML models.
""")
image = Image.open("info-images/dashboard_cover.png") if "info-images/dashboard_cover.png" else None
if image:
    st.image(image, use_column_width=True)
st.success("Navigate using the sidebar to explore the forecasts!")
