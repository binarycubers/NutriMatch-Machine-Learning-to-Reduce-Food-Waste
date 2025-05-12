import pandas as pd
import os

# Paths
input_path = "data/processed/weekly_nutrient_waste.csv"
output_dir = "data/engineered"
os.makedirs(output_dir, exist_ok=True)

# Number of lag features to generate
NUM_LAGS = 4

# Load weekly data
df = pd.read_csv(input_path, parse_dates=['date'])
df = df.sort_values('date')

# Create lagged features per nutrient
nutrients = ["carbohydrates", "fiber", "protein", "fat"]

for nutrient in nutrients:
    nutrient_df = pd.DataFrame()
    nutrient_df['date'] = df['date']
    nutrient_df[nutrient] = df[nutrient]

    # Generate lag features
    for lag in range(1, NUM_LAGS + 1):
        nutrient_df[f"lag_{lag}"] = nutrient_df[nutrient].shift(lag)

    # Set the prediction target (the current week's value)
    nutrient_df['target'] = nutrient_df[nutrient]

    # Drop rows with NaN lag values (due to shifting)
    nutrient_df = nutrient_df.dropna().reset_index(drop=True)

    # Save to CSV
    output_file = os.path.join(output_dir, f"{nutrient}_lagged.csv")
    nutrient_df.to_csv(output_file, index=False)
    print(f"✅ Saved lagged data for {nutrient} to {output_file}")
