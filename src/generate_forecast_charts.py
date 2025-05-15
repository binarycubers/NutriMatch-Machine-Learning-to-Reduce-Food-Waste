import pandas as pd
import matplotlib.pyplot as plt
import os

forecast_dir = "data/forecast"
output_dir = "graphs"

models = ["random_forest", "xgboost", "lstm"]
nutrients = ["carbohydrates", "fiber", "protein", "fat"]

for model in models:
    model_output_dir = os.path.join(output_dir, model)
    os.makedirs(model_output_dir, exist_ok=True)

    for nutrient in nutrients:
        forecast_file = os.path.join(forecast_dir, f"{nutrient}_{model}_forecast.csv")
        if not os.path.exists(forecast_file):
            print(f"❌ Missing: {forecast_file}")
            continue

        df = pd.read_csv(forecast_file)
        plt.figure(figsize=(12, 6))

        if "date" in df.columns:
            x = pd.to_datetime(df["date"])
        else:
            x = list(range(len(df)))

        plt.plot(x, df["actual"], label="Actual Waste", color="black", linewidth=2.5, marker='o')
        if "predicted_train" in df:
            plt.plot(x, df["predicted_train"], label="Predicted Waste (Train)", linestyle="--", color="orange", linewidth=2)
        if "predicted_test" in df:
            plt.plot(x, df["predicted_test"], label="Predicted Waste (Test)", linestyle="-.", color="cyan", linewidth=2)

        plt.title(f"Prediction Waste for {nutrient.title()} ({model.replace('_', ' ').title()})", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel(nutrient.title(), fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(model_output_dir, f"{nutrient}_{model}_forecast.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")
