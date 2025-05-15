import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

engineered_dir = "data/engineered"
models_dir = "models"
forecast_dir = "data/forecast"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(forecast_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]
model_name = "random_forest"
all_scores = []

for nutrient in nutrients:
    df = pd.read_csv(os.path.join(engineered_dir, f"{nutrient}_lagged.csv"))

    X = df.drop(["target", "date"], axis=1, errors="ignore")
    y = df["target"]

    # Split into train/test (last 8 for test)
    X_train, X_test = X[:-8], X[-8:]
    y_train, y_test = y[:-8], y[-8:]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, train_preds)
    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = mean_squared_error(y_test, test_preds, squared=False)

    all_scores.append({
        "nutrient": nutrient,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "test_mse": test_mse,
        "test_rmse": test_rmse
    })

    # Save model
    model_path = os.path.join(models_dir, f"{nutrient}_random_forest.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved model: {model_path}")

    # Save combined prediction CSV
    combined_df = pd.DataFrame({
        "date": df["date"],
        "actual": y,
        "predicted_train": list(train_preds) + [None]*8,
        "predicted_test": [None]*(len(y)-8) + list(test_preds)
    })
    forecast_path = os.path.join(forecast_dir, f"{nutrient}_random_forest_forecast.csv")
    combined_df.to_csv(forecast_path, index=False)
    print(f"📊 Saved forecast: {forecast_path}")

# Save score file
score_df = pd.DataFrame(all_scores)
score_df.to_csv(os.path.join(forecast_dir, "random_forest_model_scores.csv"), index=False)
print("📈 Saved random forest score file.")
