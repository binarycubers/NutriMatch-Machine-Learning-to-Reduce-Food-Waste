import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
input_path = "data/raw/Item_Quantity.csv"
output_path = "graphs/stock_analysis.png"
os.makedirs("graphs", exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]

# Check required columns
if "item description" not in df.columns or "quantity" not in df.columns:
    raise ValueError("Missing 'item description' or 'quantity' column in Item_Quantity.csv")

# Group by item and sum quantities
top_items = df.groupby("item description")["quantity"].sum().sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
top_items.plot(kind="bar", color="skyblue")
plt.title("Top 10 Stocked Items (Based on Quantity)", fontsize=14)
plt.xlabel("Item Description", fontsize=12)
plt.ylabel("Total Quantity", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save
plt.savefig(output_path)
plt.close()
print(f"✅ Saved chart: {output_path}")
