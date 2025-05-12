import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Paths
engineered_dir = "data/engineered"
models_dir = "models"
forecast_dir = "data/forecast"
metrics_path = os.path.join(forecast_dir, "xgb_model_scores.csv")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(forecast_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]

# Function to load data
def load_data(nutrient):
    return pd.read_csv(os.path.join(engineered_dir, f"{nutrient}_lagged.csv"))

# Function to split data
def split_data(df):
    X = df.drop(columns=["target", "date"], errors='ignore')
    y = df["target"]
    return X[:-8], X[-8:], y[:-8], y[-8:]

# Save evaluation results
results = []

# Loop over nutrients
for nutrient in nutrients:
    df = load_data(nutrient)
    dates = df["date"].values[-8:] if "date" in df.columns else np.arange(1, 9)

    X_train, X_test, y_train, y_test = split_data(df)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mse = mean_squared_error(y_test, preds)

    print(f"{nutrient} XGBoost RMSE: {rmse:.2f}")

    # Save model
    model_path = os.path.join(models_dir, f"{nutrient}_xgboost.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved model: {model_path}")

    # Save forecast CSV
    forecast_df = pd.DataFrame({
        "date": dates,
        "actual": y_test.values,
        "predicted": preds
    })
    forecast_csv_path = os.path.join(forecast_dir, f"{nutrient}_xgboost_forecast.csv")
    forecast_df.to_csv(forecast_csv_path, index=False)
    print(f"📊 Saved forecast CSV: {forecast_csv_path}")

    # Append metrics
    results.append({
        "nutrient": nutrient,
        "model": "XGBoost",
        "mse": round(mse, 4),
        "rmse": round(rmse, 4)
    })

# Save all metrics
results_df = pd.DataFrame(results)
results_df.to_csv(metrics_path, index=False)
print(f"📈 Saved model evaluation scores: {metrics_path}")
