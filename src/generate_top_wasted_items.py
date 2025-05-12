import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
input_path = "data/raw/Item_FullList.csv"
output_path = "graphs/top_wasted_items.png"
os.makedirs("graphs", exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]

# Check necessary columns
required_cols = ["item description", "quantity"]
nutrient_cols = ["carbohydrates", "fiber", "protein", "fat"]
missing = [col for col in required_cols + nutrient_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Compute total waste per item by multiplying quantity with each nutrient, then summing
for col in nutrient_cols:
    df[f"{col}_total"] = df[col] * df["quantity"]

df["total_waste"] = df[[f"{col}_total" for col in nutrient_cols]].sum(axis=1)

# Group by item and sum total waste
top_items = df.groupby("item description")["total_waste"].sum().sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
top_items.plot(kind="bar", color="tomato")
plt.title("Top 10 Wasted Items by Predicted Nutrient Waste", fontsize=14)
plt.xlabel("Item Description", fontsize=12)
plt.ylabel("Total Predicted Waste", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save
plt.savefig(output_path)
plt.close()
print(f"✅ Saved chart: {output_path}")
