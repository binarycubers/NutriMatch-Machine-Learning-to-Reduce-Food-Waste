HealthFusion - Nutrient Waste Forecasting Dashboard

🌱 Project Overview

HealthFusion is an AI-powered food waste prediction dashboard that helps monitor, analyze, and forecast nutrient waste using machine learning models such as Random Forest, XGBoost, and LSTM. It also includes visual analytics for stock trends and the most wasted food items.

⚙️ 1. Environment Setup

📦 Requirements
To get started, install the required dependencies:

pip install -r requirements.txt

🧠 Key Libraries Used

streamlit – for the interactive dashboard
pandas – for data manipulation
matplotlib – for plotting
scikit-learn – for training ML models (Random Forest, XGBoost)
xgboost – for XGBoost model
tensorflow/keras – for LSTM models
joblib – for saving/loading model objects



📁 2. Project Directory Structure

NutriMatch-Machine-Learning-to-Reduce-Food-Waste/
│
├── data/
│   ├── raw/                # Initial input CSVs (Merged_ItemList.csv, etc.)
│   ├── interim/            # Intermediate cleaned/preprocessed data
│   ├── processed/          # Daily and weekly nutrient data
│   ├── engineered/         # Lagged feature files for model training
│   └── forecast/           # Model forecast outputs and score CSVs
│
├── models/                 # Saved models (.pkl for RF/XGBoost, .h5 for LSTM)
├── graphs/                 # Generated visual forecast and stock analysis charts
│   ├── lstm/
│   ├── random_forest/
│   ├── xgboost/
│   └── stock_analysis_pie.png
│
├── pages/                  # Streamlit multipage app scripts
│   ├── Home.py
│   ├── Upload.py
│   ├── All_Predictions.py
│   └── Best_Model.py
│
├── src/                    # Core scripts for model training, preprocessing, charts
│   ├── generate_forecast_charts.py
│   ├── generate_model_scores.py
│   ├── generate_lag_features.py
│   ├── generate_stock_analysis_piechart.py
│   ├── preprocess_nutrient_data.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   └── train_lstm.py
│
├── results/                # (Optional) Any CSV or result outputs
├── README.md               # This file
├── app.py / main.py        # Streamlit entry point
└── requirements.txt


🔁 3. Reusing for New Dataset
If you want to analyze a new dataset, do the following cleanup first:

🗑 Delete These:
CSV Data:
    data/engineered/*.csv
    data/forecast/*.csv
    data/interim/*.csv
    data/processed/*.csv

Model Files:
    models/*.pkl
    models/*.h5

Graphs:
    graphs/**/*.png (or specifically forecast/*.png and stock_analysis_*.png)

This ensures your new data is processed and modeled cleanly without conflicts from previous runs.


📥 4. Required Format for Raw Data
The main raw input must be:

📄 Merged_ItemList.csv
Column Name	Description
item description	Name of the food item
qty	Quantity wasted per item
date	Date of the entry (dd/mm/yyyy)

Make sure column names are spelled exactly (lowercase preferred) and qty must be numerical.

⚙️ 5. Step-by-Step Usage Instructions
After placing the new raw files in data/raw/, follow this order:

🔹 Step 1: Preprocess the Raw Data

python src/preprocess_nutrient_data.py
Generates: daily_nutrient_waste.csv and weekly_nutrient_waste.csv


🔹 Step 2: Generate Lagged Features

python src/generate_lag_features.py
Generates: *_lagged.csv files in data/engineered/

🔹 Step 3: Train Models

python src/train_random_forest.py
python src/train_xgboost.py
python src/train_lstm.py
Saves models in models/

Also generates forecast CSVs in data/forecast/

🔹 Step 4: Evaluate & Score Models

python src/generate_model_scores.py
Saves: *_model_scores.csv files in data/forecast/

🔹 Step 5: Generate Forecast Graphs

python src/generate_forecast_charts.py
Saves: line charts in graphs/{model}/{nutrient}_{model}_forecast.png

🔹 Step 6: Generate Stock Analysis Chart

python src/generate_stock_analysis_piechart.py
Saves: stock_analysis_pie.png to graphs/

🔹 Step 7: Run the Dashboard

streamlit run Home.py
From the sidebar:

Home – Project overview and stock analysis chart

All Predictions – View charts, RMSEs, and forecasts for all models

Best Model – Displays LSTM results and evaluation

Upload – (Coming soon) for custom file input



🧠 Notes
Best model currently is LSTM, based on lowest RMSE across nutrients.
All charts are styled to reflect academic report formatting.
Training & test splits are handled inside model scripts (last 8 weeks as test).
Model performance is logged and displayed dynamically.