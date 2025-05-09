import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

engineered_dir = "data/engineered"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]

def load_data(nutrient):
    return pd.read_csv(os.path.join(engineered_dir, f"{nutrient}_lagged.csv"))

def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X[:-8], X[-8:], y[:-8], y[-8:]

for nutrient in nutrients:
    df = load_data(nutrient)
    X_train, X_test, y_train, y_test = split_data(df)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"{nutrient} Random Forest RMSE: {rmse:.2f}")

    model_path = os.path.join(models_dir, f"{nutrient}_random_forest.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved: {model_path}")