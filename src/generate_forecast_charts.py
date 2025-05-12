import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
forecast_dir = "data/forecast"
graphs_dir = "graphs"
os.makedirs(graphs_dir, exist_ok=True)

nutrients = ["carbohydrates", "fiber", "protein", "fat"]
models = ["random_forest", "xgboost", "lstm"]

for nutrient in nutrients:
    for model in models:
        forecast_file = os.path.join(forecast_dir, f"{nutrient}_{model}_forecast.csv")

        if not os.path.exists(forecast_file):
            print(f"⚠️ Skipping missing file: {forecast_file}")
            continue

        df = pd.read_csv(forecast_file)

        # Ensure dates are sorted and in datetime format
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        else:
            df["date"] = [f"Week {i+1}" for i in range(len(df))]

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(df["date"], df["actual"], label="Actual", marker='o')
        plt.plot(df["date"], df["predicted"], label="Predicted", marker='x')
        plt.title(f"{nutrient.capitalize()} Forecast - {model.replace('_', ' ').title()}")
        plt.xlabel("Date")
        plt.ylabel(f"{nutrient.capitalize()} Value")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(graphs_dir, f"{nutrient}_{model}_forecast.png")
        plt.savefig(output_file)
        plt.close()

        print(f"✅ Saved graph: {output_file}")
