import pandas as pd
import os

# Paths
raw_path = "data/raw/Item_FullList.csv"
interim_path = "data/interim/daily_nutrient_waste.csv"
processed_path = "data/processed/weekly_nutrient_waste.csv"
os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load the raw data from Item_FullList.csv
df = pd.read_csv(raw_path)

# Standardize column names (lowercase, stripped)
df.columns = [col.strip().lower() for col in df.columns]

# Convert date column
if 'date' not in df.columns:
    raise ValueError("Missing 'date' column in source file")
df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date'])

# Explicitly select only correct nutrient columns (assumed totals already multiplied by quantity)
selected_columns = ['date', 'carbohydrates', 'protein', 'fat', 'fiber']
df_filtered = df[selected_columns].copy()

# Group by date and sum nutrient values
daily_df = df_filtered.groupby('date')[['carbohydrates', 'fiber', 'protein', 'fat']].sum().reset_index()
daily_df.to_csv(interim_path, index=False)

# Resample to weekly totals (ending on Sunday)
daily_df.set_index('date', inplace=True)
weekly_df = daily_df.resample('W-SUN').sum().reset_index()
weekly_df.to_csv(processed_path, index=False)

print("✅ Cleaned nutrient data preprocessing completed.")
print(f"- Daily totals saved to: {interim_path}")
print(f"- Weekly totals saved to: {processed_path}")