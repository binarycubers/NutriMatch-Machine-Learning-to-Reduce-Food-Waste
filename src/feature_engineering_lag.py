import pandas as pd
import os
# Load and prepare the data
df = pd.read_csv("data/processed/weekly_food_waste_20250507_000105.csv")
df = df.rename(columns={
    "Carbohydrates": "carbohydrates",
    "Fiber": "fiber",
    "Protein": "protein",
    "Fat": "fat"
})
df = df.sort_values("Week").reset_index(drop=True)
# Create output folder if not exists
os.makedirs("data/engineered", exist_ok=True)
# Function to create lag features
def create_lag_features(df, col, window=4):
    data = pd.DataFrame()
    for i in range(window):
        data[f'lag_{i+1}'] = df[col].shift(i + 1)
    data['target'] = df[col]
    return data.dropna().reset_index(drop=True)
# Function to split train/test (optional if needed)
def split_data(data):
    X = data.drop("target", axis=1)
    y = data["target"]
    return X[:-8], X[-8:], y[:-8], y[-8:]
# Process all nutrients
nutrients = ["carbohydrates", "fiber", "protein", "fat"]
for nutrient in nutrients:
    print(f"🔹 Creating lag features for: {nutrient}")
    lagged = create_lag_features(df, nutrient)
    lagged.to_csv(f"data/engineered/{nutrient}_lagged.csv", index=False)
    print(f"✅ Saved to data/engineered/{nutrient}_lagged.csv")