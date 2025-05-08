import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Paths
engineered_dir = "data/engineered"
forecast_dir = "data/forecast"
models_dir = "models"
os.makedirs(forecast_dir, exist_ok=True)
# Nutrients
nutrients = ["carbohydrates", "fiber", "protein", "fat"]
# Forecast function
def forecast_lstm(nutrient):
    print(f"\n:crystal_ball: Forecasting with LSTM for {nutrient}...")
    # Load data
    df = pd.read_csv(os.path.join(engineered_dir, f"{nutrient}_lagged.csv"))
    data = df.drop("target", axis=1).values[-1:]  # last row
    # Reshape to (1, 4, 1)
    current_input = data.reshape((1, data.shape[1], 1))
    model = load_model(os.path.join(models_dir, f"{nutrient}_lstm_model.h5"))
    predictions = []
    for _ in range(8):
        pred = model.predict(current_input)[0][0]
        predictions.append(pred)
        # Update input: shift and append
        new_input = current_input.flatten().tolist()[1:]  # drop oldest
        new_input.append(pred)  # add new pred
        current_input = np.array(new_input).reshape((1, 4, 1))
    # Save CSV
    forecast_df = pd.DataFrame({"Week": range(1, 9), "Prediction": predictions})
    csv_path = os.path.join(forecast_dir, f"{nutrient}_lstm_forecast.csv")
    forecast_df.to_csv(csv_path, index=False)
    print(f":white_check_mark: Saved forecast CSV: {csv_path}")
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(forecast_df["Week"], forecast_df["Prediction"], marker='o', linestyle='-')
    plt.title(f"8-Week Forecast for {nutrient.capitalize()} (LSTM)")
    plt.xlabel("Week")
    plt.ylabel("Predicted Value")
    plt.grid(True)
    plot_path = os.path.join(forecast_dir, f"{nutrient}_lstm_forecast.png")
    plt.savefig(plot_path)
    plt.close()
    print(f":chart_with_upwards_trend: Saved forecast graph: {plot_path}")
# Main driver
if __name__ == "__main__":
    for nutrient in nutrients:
        forecast_lstm(nutrient)