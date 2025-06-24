# instacart_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Current working directory:", os.getcwd())


# Configure display
pd.set_option('display.max_columns', 20)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
DATA_PATH = "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files" 

orders = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/orders.csv"))
products = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/products.csv"))
order_products_prior = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/order_products__prior.csv"))
order_products_train = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/order_products__train.csv"))
aisles = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/aisles.csv"))
departments = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/departments.csv"))

# Preview
print("Orders:")
print(orders.head())
print("\nProducts:")
print(products.head())
print("\nPrior Order Products:")
print(order_products_prior.head())

# -------------------------------------------------------------------
# 1️⃣ Top 20 Most Frequently Ordered Products
# -------------------------------------------------------------------

print("\nTop 20 Most Frequently Ordered Products:")
top_products = order_products_prior['product_id'].value_counts().head(20).reset_index()

print("Columns in top_products before rename:", top_products.columns)
print(top_products.head())

# Try renaming if the columns are named differently
if 'index' in top_products.columns and 'product_id' in top_products.columns:
    top_products = top_products.rename(columns={'index': 'product_id', 'product_id': 'order_count'})
elif 'product_id' in top_products.columns and 'product_id' in top_products.columns:
    # This case means columns are duplicated
    top_products.columns = ['product_id', 'order_count']
else:
    print("Columns do not match expected pattern. Skipping rename.")

print("Columns in top_products after rename:", top_products.columns)
print(top_products.head())

# Now check if 'product_id' column exists before merging
if 'product_id' in top_products.columns:
    top_products = top_products.merge(products, on='product_id')
    print(top_products[['product_name', 'order_count']])
else:
    print("No 'product_id' column to merge on. Merge skipped.")

print(top_products.columns)

# Plot
plt.figure()
sns.barplot(data=top_products, y='product_name', x='order_count', palette='viridis')
plt.title("Top 20 Most Frequently Ordered Products")
plt.xlabel("Number of Orders")
plt.ylabel("Product Name")
plt.tight_layout()
plt.savefig("visuals/top_products.png")
plt.show()

# -------------------------------------------------------------------
# 2️⃣ Reorder Ratios
# -------------------------------------------------------------------

print("\nReorder Ratios:")
reorder_ratio = order_products_prior['reordered'].mean()
print(f"Overall reorder rate: {reorder_ratio:.2%}")

reorder_by_product = (order_products_prior.groupby('product_id')['reordered']
                      .mean()
                      .reset_index()
                      .merge(products, on='product_id'))

top_reordered = reorder_by_product.sort_values('reordered', ascending=False).head(20)

# Plot
plt.figure()
sns.barplot(data=top_reordered, y='product_name', x='reordered', palette='mako')
plt.title("Top 20 Most Reordered Products")
plt.xlabel("Reorder Rate")
plt.ylabel("Product Name")
plt.tight_layout()
plt.savefig("visuals/top_reordered_products.png")
plt.show()

# -------------------------------------------------------------------
# 3️⃣ Orders by Hour and Day
# -------------------------------------------------------------------

# Hour of Day
plt.figure()
sns.countplot(data=orders, x='order_hour_of_day', palette='rocket')
plt.title("Orders by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("visuals/orders_by_hour.png")
plt.show()

# Day of Week
plt.figure()
sns.countplot(data=orders, x='order_dow', palette='flare')
plt.title("Orders by Day of Week")
plt.xlabel("Day of Week (0=Sunday)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("visuals/orders_by_day.png")
plt.show()