import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

engineered_dir = "data/engineered"
model_dir = "models"
forecast_dir = "data/forecast"
os.makedirs(forecast_dir, exist_ok=True)

nutrients = ["carbohydrates", "protein", "fat", "fiber"]
models = {
    "random_forest": lambda n: joblib.load(f"{model_dir}/{n}_random_forest.pkl"),
    "xgboost": lambda n: joblib.load(f"{model_dir}/{n}_xgboost.pkl"),
    "lstm": lambda n: load_model(f"{model_dir}/{n}_lstm.h5", compile=False)
}

for model_name, load_func in models.items():
    records = []
    for nutrient in nutrients:
        df = pd.read_csv(f"{engineered_dir}/{nutrient}_lagged.csv")
        X = df.drop(["target", "date"], axis=1)
        y = df["target"]
        X_train, X_test = X[:-8], X[-8:]
        y_train, y_test = y[:-8], y[-8:]

        model = load_func(nutrient)

        if model_name == "lstm":
            X_train_input = np.array(X_train)[..., np.newaxis]
            X_test_input = np.array(X_test)[..., np.newaxis]
            y_train_pred = model.predict(X_train_input).flatten()
            y_test_pred = model.predict(X_test_input).flatten()
        else:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)

        records.append({
            "nutrient": nutrient,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "test_mse": test_mse,
            "test_rmse": test_rmse
        })

    df_scores = pd.DataFrame(records)
    df_scores.to_csv(f"{forecast_dir}/{model_name}_model_scores.csv", index=False)
    print(f"✅ Saved: {forecast_dir}/{model_name}_model_scores.csv")
