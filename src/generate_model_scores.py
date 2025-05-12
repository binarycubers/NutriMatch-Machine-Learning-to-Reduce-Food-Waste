import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

nutrients = ["carbohydrates", "protein", "fat", "fiber"]
models = ["random_forest", "xgboost", "lstm"]

engineered_dir = "data/engineered"
model_dir = "models"
scores_dir = "data/forecast"
os.makedirs(scores_dir, exist_ok=True)

for model_name in models:
    records = []
    for nutrient in nutrients:
        df_path = os.path.join(engineered_dir, f"{nutrient}_lagged.csv")
        if not os.path.exists(df_path):
            continue

        df = pd.read_csv(df_path)
        X = df.drop(["target", "date"], axis=1, errors="ignore")
        y = df["target"]
        X_test, y_test = X[-8:], y[-8:]

        if model_name == "lstm":
            model_path = os.path.join(model_dir, f"{nutrient}_lstm.h5")
            if not os.path.exists(model_path):
                continue
            model = load_model(model_path, compile=False)
            preds = model.predict(np.array(X_test)[..., np.newaxis])
        else:
            model_path = os.path.join(model_dir, f"{nutrient}_{model_name}.pkl")
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        records.append({"nutrient": nutrient, "mse": mse, "rmse": rmse})

    # Save per model
    if records:
        df_scores = pd.DataFrame(records)
        df_scores.to_csv(os.path.join(scores_dir, f"{model_name}_model_scores.csv"), index=False)
        print(f"Saved: {model_name}_model_scores.csv")
