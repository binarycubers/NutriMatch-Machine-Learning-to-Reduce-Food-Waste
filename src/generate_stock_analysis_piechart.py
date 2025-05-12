import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
input_path = "data/raw/Item_Quantity.csv"
output_path = "graphs/stock_analysis_pie.png"
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

# Prepare labels and values
labels = top_items.index.tolist()
values = top_items.values

# Pie chart
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10.colors  # consistent color scheme
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Top 10 Wasted Items by Quantity (Pie Chart)", fontsize=14)
plt.axis('equal')
plt.tight_layout()

# Add annotation (highest and lowest)
highest_item = labels[0] + f" ({values[0]} units)"
lowest_item = labels[-1] + f" ({values[-1]} units)"
caption = f"Highest: {highest_item}\nLowest: {lowest_item}"
plt.gcf().text(0.02, 0.02, caption, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

# Save
plt.savefig(output_path)
plt.close()
print(f"✅ Saved pie chart: {output_path}")