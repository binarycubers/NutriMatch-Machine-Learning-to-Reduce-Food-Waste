import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_lagged_data(nutrient, directory="data/engineered"):
    path = os.path.join(directory, f"{nutrient}_lagged.csv")
    return pd.read_csv(path)

def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return X[:-8], X[-8:], y[:-8], y[-8:]

def train_and_save_model(X_train, X_test, y_train, y_test, nutrient, output_dir="models"):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"XGBoost | {nutrient} → RMSE: {rmse:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{nutrient}_xgboost.pkl")
    joblib.dump(model, model_path)
    print(f":white_check_mark: Saved model: {model_path}\n")

def main():
    nutrients = ["carbohydrates", "fiber", "protein", "fat"]
    for nutrient in nutrients:
        print(f"\n:small_blue_diamond: Training XGBoost for: {nutrient}")
        df = load_lagged_data(nutrient)
        X_train, X_test, y_train, y_test = split_data(df)
        train_and_save_model(X_train, X_test, y_train, y_test, nutrient)

if __name__ == "__main__":
    main()
