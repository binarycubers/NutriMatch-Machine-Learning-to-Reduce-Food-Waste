import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
# Directories
engineered_dir = "data/engineered"
models_dir = "models"
forecast_dir = "data/forecast"
os.makedirs(forecast_dir, exist_ok=True)
# Nutrients and their matching files
nutrients = ["carbohydrates", "fiber", "protein", "fat"]
models = ["random_forest", "xgboost"]  # You can change this to just the best one if needed
def forecast_next_8_weeks(df_last, model):
    """
    Given a lagged dataframe (last row), forecast 8 weeks ahead
    """
    predictions = []
    current_input = df_last.values.flatten().tolist()
    for _ in range(8):
        pred = model.predict([current_input])[0]
        predictions.append(pred)
        # Shift lag values and insert new prediction
        current_input = current_input[0:-1]  # remove oldest lag
        current_input.insert(0, pred)        # add new prediction at front
    return predictions
def plot_predictions(nutrient, model_name, predictions):
    weeks = list(range(1, 9))
    plt.figure(figsize=(8, 5))
    plt.plot(weeks, predictions, marker='o', linestyle='-')
    plt.title(f"8-Week Forecast for {nutrient.capitalize()} ({model_name.replace('_', ' ').title()})")
    plt.xlabel("Week")
    plt.ylabel("Predicted Value")
    plt.grid(True)
    file_path = os.path.join(forecast_dir, f"{nutrient}_{model_name}_forecast.png")
    plt.savefig(file_path)
    plt.close()
def main():
    for nutrient in nutrients:
        print(f"\n:small_blue_diamond: Forecasting {nutrient} for next 8 weeks")
        # Load lagged data
        lagged_file = os.path.join(engineered_dir, f"{nutrient}_lagged.csv")
        df = pd.read_csv(lagged_file)
        last_row = df.iloc[-1:].drop("target", axis=1)
        for model_name in models:
            model_file = os.path.join(models_dir, f"{nutrient}_{model_name}.pkl")
            model = joblib.load(model_file)
            predictions = forecast_next_8_weeks(last_row, model)
            # Save predictions
            pred_df = pd.DataFrame({"Week": list(range(1, 9)), "Prediction": predictions})
            csv_path = os.path.join(forecast_dir, f"{nutrient}_{model_name}_forecast.csv")
            pred_df.to_csv(csv_path, index=False)
            print(f":white_check_mark: Saved forecast: {csv_path}")
            # Save plot
            plot_predictions(nutrient, model_name, predictions)
            print(f":chart_with_upwards_trend: Saved graph: {nutrient}_{model_name}_forecast.png")
if __name__ == "__main__":
    main()