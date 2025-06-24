# detailed_analysis.py

import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', 20)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load data
DATA_PATH = "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files" 

orders = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/orders.csv"))
products = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/products.csv"))
order_products_prior = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/order_products__prior.csv"))
order_products_train = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/order_products__train.csv"))
aisles = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/aisles.csv"))
departments = pd.read_csv(os.path.join(DATA_PATH, "/Users/liamshepherd/Desktop/GitHub/instacart-customer-analytics/instacart-customer-analytics/data/Project_Files/departments.csv"))

# Merge product info into prior orders
prior = order_products_prior.merge(products, on='product_id', how='left') \
                            .merge(aisles, on='aisle_id', how='left') \
                            .merge(departments, on='department_id', how='left') \
                            .merge(orders[['order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day']], on='order_id', how='left')

print("✅ Data loaded and merged.")

# -------------------------------------------------------------------
#  CUSTOMER SEGMENTATION METRICS
# -------------------------------------------------------------------

print("\n🔍 Starting customer segmentation analysis...")

# Total orders per user
user_orders = orders.groupby('user_id')['order_number'].max().reset_index()
user_orders.columns = ['user_id', 'total_orders']

# Average basket size
prior_basket_sizes = prior.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
avg_basket_size = prior_basket_sizes.groupby('user_id')['basket_size'].mean().reset_index()
avg_basket_size.columns = ['user_id', 'avg_basket_size']

# Reorder ratio per user
user_reorders = prior.groupby('user_id')['reordered'].mean().reset_index()
user_reorders.columns = ['user_id', 'reorder_ratio']

# Average days between orders
days_between_orders = orders.groupby('user_id')['days_since_prior_order'].mean().reset_index()
days_between_orders.columns = ['user_id', 'avg_days_between_orders']

# Combine all user features
user_features = user_orders \
    .merge(avg_basket_size, on='user_id') \
    .merge(user_reorders, on='user_id') \
    .merge(days_between_orders, on='user_id')

print("✅ User feature matrix created. Sample:")
print(user_features.head())

# -------------------------------------------------------------------
# 📊 VISUALISE CUSTOMER FEATURES
# -------------------------------------------------------------------

import matplotlib.ticker as mtick

print("\n📊 Visualising customer feature distributions...")

# Create /visuals if needed
if not os.path.exists("visuals"):
    os.makedirs("visuals")

# Histogram: Total Orders
plt.figure()
sns.histplot(user_features['total_orders'], bins=30, kde=True, color='skyblue')
plt.title("Total Orders per Customer")
plt.xlabel("Total Orders")
plt.ylabel("Count of Customers")
plt.savefig("visuals/customer_total_orders.png")
plt.close()

# Histogram: Average Basket Size
plt.figure()
sns.histplot(user_features['avg_basket_size'], bins=30, kde=True, color='orange')
plt.title("Average Basket Size")
plt.xlabel("Average Basket Size")
plt.ylabel("Count of Customers")
plt.savefig("visuals/customer_avg_basket_size.png")
plt.close()

# Histogram: Reorder Ratio
plt.figure()
sns.histplot(user_features['reorder_ratio'], bins=30, kde=True, color='seagreen')
plt.title("Reorder Ratio Distribution")
plt.xlabel("Reorder Ratio")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylabel("Count of Customers")
plt.savefig("visuals/customer_reorder_ratio.png")
plt.close()

# Histogram: Average Days Between Orders
plt.figure()
sns.histplot(user_features['avg_days_between_orders'], bins=30, kde=True, color='purple')
plt.title("Avg Days Between Orders")
plt.xlabel("Average Days")
plt.ylabel("Count of Customers")
plt.savefig("visuals/customer_days_between_orders.png")
plt.close()

print("✅ Visuals saved to /visuals.")

# -------------------------------------------------------------------
# 🤝 PRODUCT AFFINITY ANALYSIS (Market Basket)
# -------------------------------------------------------------------

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

print("\n🤝 Running product affinity analysis...")

# STEP 1: Prepare transactions list (each order as a list of product names)
# Limit to smaller sample for performance
sample_orders = prior.groupby('order_id')['product_name'].apply(list)

# Optional: sample only N transactions to speed up
sample_transactions = sample_orders.sample(n=10000, random_state=42)

# STEP 2: Encode transactions into 0/1 matrix
te = TransactionEncoder()
te_ary = te.fit(sample_transactions).transform(sample_transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# STEP 3: Run Apriori to find frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# STEP 4: Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Sort rules by lift (highest first)
rules_sorted = rules.sort_values(by="lift", ascending=False)

# Save top rules to CSV
rules_sorted.to_csv("visuals/product_affinity_rules.csv", index=False)

# Print top 5 product pairs
print("Top 5 Product Affinity Rules:")
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

print("✅ Product affinity rules saved to 'visuals/product_affinity_rules.csv'.")

# -------------------------------------------------------------------
# 🤝 VISUALISE PRODUCT AFFINITY RULES
# -------------------------------------------------------------------

print("\n📊 Visualising product affinity rules...")

plt.figure(figsize=(10, 6))

# Scatter plot of support vs confidence, colored by lift
scatter = plt.scatter(
    rules_sorted['support'],
    rules_sorted['confidence'],
    c=rules_sorted['lift'],
    cmap='viridis',
    s=50,
    alpha=0.7
)

plt.colorbar(scatter, label='Lift')
plt.title('Product Affinity Rules: Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True)
plt.savefig("visuals/product_affinity_scatter.png")
plt.close()

print("✅ Product affinity scatterplot saved to /visuals/product_affinity_scatter.png")

# -------------------------------------------------------------------
# 📝 PRODUCT AFFINITY SUMMARY
# -------------------------------------------------------------------

summary = """
The product affinity analysis uncovered meaningful relationships between commonly
purchased items in the Instacart dataset. Notably, organic fruits tend to be
frequently bought together, such as Organic Strawberries and Organic Raspberries,
which show a high lift of approximately 2.9, indicating these products co-occur
nearly three times more often than by random chance.

Other strong associations include Organic Raspberries with Bag of Organic Bananas,
and Organic Fuji Apples with Bananas, all having lifts above 2.5, reflecting
a strong affinity between these complementary products.

Support and confidence scores suggest these product pairs are not only strongly
associated but also relatively common among transactions, making them valuable
insights for targeted marketing, product placement, and recommendation engines.
"""

print(summary)
