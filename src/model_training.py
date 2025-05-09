import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

engineered_dir = "data/engineered"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]

def load_lagged_data(nutrient):
    path = os.path.join(engineered_dir, f"{nutrient}_lagged.csv")
    return pd.read_csv(path)

def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X[:-8], X[-8:], y[:-8], y[-8:]

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name, nutrient):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"{model_name} | {nutrient} → RMSE: {rmse:.2f}")

    # Save the actual model, not predictions
    filename = f"{nutrient}_{model_name.lower().replace(' ', '_')}.pkl"
    model_path = os.path.join(models_dir, filename)
    joblib.dump(model, model_path)
    print(f"✅ Saved: {model_path}")

def main():
    for nutrient in nutrients:
        print(f"\n🔹 Training models for: {nutrient}")
        df = load_lagged_data(nutrient)
        X_train, X_test, y_train, y_test = split_data(df)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        train_and_evaluate(X_train, X_test, y_train, y_test, rf, "Random Forest", nutrient)

        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        train_and_evaluate(X_train, X_test, y_train, y_test, xgb, "XGBoost", nutrient)

if __name__ == "__main__":
    main()
