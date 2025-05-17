import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="Best Model - HealthFusion",
    layout="wide",
    page_icon="🧪"
)

# --- Load model scores to determine best model dynamically ---
@st.cache_data
def load_scores():
    model_paths = {
        "LSTM": "data/forecast/lstm_model_scores.csv",
        "XGBoost": "data/forecast/xgboost_model_scores.csv",
        "Random Forest": "data/forecast/random_forest_model_scores.csv"
    }
    scores = {}
    for model, path in model_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            scores[model] = df["test_rmse"].mean()
    best_model = min(scores, key=scores.get)
    return best_model, model_paths[best_model]

best_model, best_model_score_path = load_scores()
score_df = pd.read_csv(best_model_score_path)

# Apply consistent styling across pages
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

# --- Page config ---
st.markdown(f"## 🥇 Best Model Forecast: {best_model}")
st.markdown(f"""
{best_model} has been identified as the best-performing model based on **lowest Testing RMSE** across nutrient categories.  
Below you can select a nutrient and view its detailed forecast including:
- Actual vs Predicted values (Training & Testing)
- Training/Testing MSE & RMSE
- Weekly forecast values
""")

# --- Nutrient Selector ---
nutrients = ["carbohydrates", "protein", "fat", "fiber"]
selected_nutrient = st.selectbox("Select Nutrient", nutrients)

# --- Load & Display Graph ---
graph_path = f"graphs/{best_model.lower()}/{selected_nutrient}_{best_model.lower()}_forecast.png"
if os.path.exists(graph_path):
    st.image(graph_path, caption=f"Forecast for {selected_nutrient.title()} using {best_model}", width=850)
else:
    st.warning(f"Graph not found: {graph_path}")

# --- Display Score Metrics ---
st.markdown(f"### 🧠 Model Evaluation Metrics ({best_model})")
score_row = score_df[score_df['nutrient'] == selected_nutrient]
if not score_row.empty:
    row = score_row.iloc[0]
    st.markdown(f"- **Training MSE**: `{row['train_mse']:.4f}`")
    st.markdown(f"- **Training RMSE**: `{row['train_rmse']:.4f}`")
    st.markdown(f"- **Testing MSE**: `{row['test_mse']:.4f}`")
    st.markdown(f"- **Testing RMSE**: `{row['test_rmse']:.4f}`")
else:
    st.error(f"No score found for {selected_nutrient} in {best_model} score file.")

# --- Load & Display Forecast Table ---
# --- Full Model Comparison for Selected Nutrient ---
st.markdown(f"### 🔍 {selected_nutrient.title()} Comparison Across All Models")

comparison_data = []

# Load scores for all models
for model_name, path in {
    "LSTM": "data/forecast/lstm_model_scores.csv",
    "XGBoost": "data/forecast/xgboost_model_scores.csv",
    "Random Forest": "data/forecast/random_forest_model_scores.csv"
}.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df[df['nutrient'] == selected_nutrient]
        if not row.empty:
            r = row.iloc[0]
            comparison_data.append({
                "Model": model_name,
                "Train MSE": f"{r['train_mse']:.2f}",
                "Train RMSE": f"{r['train_rmse']:.2f}",
                "Test MSE": f"{r['test_mse']:.2f}",
                "Test RMSE": f"{r['test_rmse']:.2f}",
            })

# Convert to DataFrame and show
if comparison_data:
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)
else:
    st.warning("Comparison data not available for this nutrient.")
