import pandas as pd
import matplotlib.pyplot as plt
"""
Brazilian E-Commerce Analysis

Analysis performed:
1. Monthly revenue trends
2. Top product categories
3. Revenue by state
4. Revenue per customer by state
5. Delivery time distribution
6. Average delivery time by state

Dataset:
Olist Brazilian E-Commerce Dataset
"""

customers = pd.read_csv("olist_customers_dataset.csv")
geolocation = pd.read_csv("olist_geolocation_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")
payments = pd.read_csv("olist_order_payments_dataset.csv")
order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
orders = pd.read_csv("olist_orders_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")


orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M")

df = orders.merge(items, on="order_id", how="left")
df = df.merge(payments, on="order_id", how="left")
df = df.merge(customers, on="customer_id", how="left")
df = df.merge(products, on="product_id", how="left")
print(df.columns)

orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)

df = orders.merge(items, on="order_id", how="left")
df = df.merge(payments, on="order_id", how="left")
df = df.merge(customers, on="customer_id", how="left")
df = df.merge(products, on="product_id", how="left")

monthly_revenue = (
    df.groupby("month")["payment_value"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))
plt.plot(
    monthly_revenue["month"].astype(str),
    monthly_revenue["payment_value"],
    marker="o"
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chart_1_monthly_revenue.png", dpi=300)
plt.show()

top_categories = (
    df.groupby("product_category_name")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12, 6))
plt.barh(top_categories["product_category_name"], top_categories["payment_value"])
plt.title("Top 10 Product Categories by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product Category")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("chart_2_top_categories.png", dpi=300)
plt.show()

state_revenue = (
    df.groupby("customer_state")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(12, 6))
plt.bar(state_revenue["customer_state"], state_revenue["payment_value"])
plt.title("Revenue by Brazilian State")
plt.xlabel("State")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("chart_3_revenue_by_state.png", dpi=300)
plt.show()

# =========================
# Revenue per customer by state
# =========================

state_metrics = (
    df.groupby("customer_state")
    .agg(
        revenue=("payment_value", "sum"),
        customers=("customer_unique_id", "nunique")
    )
)

state_metrics["revenue_per_customer"] = (
    state_metrics["revenue"] /
    state_metrics["customers"]
)
state_metrics = state_metrics[
    state_metrics["customers"] >= 500
]
state_metrics = (
    state_metrics
    .sort_values("revenue_per_customer", ascending=False)
    .reset_index()
)

# Plot top 15 states

top_states = state_metrics.head(15)

plt.figure(figsize=(12, 6))
plt.barh(
    top_states["customer_state"],
    top_states["revenue_per_customer"]
)

plt.title("Average Revenue per Customer by State")
plt.xlabel("Revenue per Customer")
plt.ylabel("State")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(
    "chart_revenue_per_customer_by_state.png",
    dpi=300
)
plt.show()


orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"],
    errors="coerce"
)
orders["delivery_days"] = (
    orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
).dt.days

delivered_orders = orders.dropna(subset=["delivery_days"])

plt.figure(figsize=(12, 6))
plt.hist(delivered_orders["delivery_days"], bins=40)
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time in Days")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("chart_4_delivery_distribution.png", dpi=300)
plt.show()

# =====================================
# Average delivery time by state
# =====================================

orders["delivery_days"] = (
    orders["order_delivered_customer_date"] -
    orders["order_purchase_timestamp"]
).dt.days

delivery_state = (
    orders.merge(
        customers[["customer_id", "customer_state"]],
        on="customer_id",
        how="left"
    )
)

delivery_state = delivery_state.dropna(subset=["delivery_days"])

avg_delivery = (
    delivery_state.groupby("customer_state")
    .agg(
        avg_delivery_days=("delivery_days", "mean"),
        orders=("order_id", "count")
    )
)

# Remove tiny sample sizes
avg_delivery = avg_delivery[
    avg_delivery["orders"] >= 100
]

avg_delivery = (
    avg_delivery
    .sort_values("avg_delivery_days", ascending=False)
    .reset_index()
)

plt.figure(figsize=(12, 7))

plt.barh(
    avg_delivery["customer_state"],
    avg_delivery["avg_delivery_days"]
)

plt.title("Average Delivery Time by State")
plt.xlabel("Average Delivery Time (Days)")
plt.ylabel("State")

plt.tight_layout()
plt.savefig(
    "average_delivery_by_state.png",
    dpi=300
)
plt.show()
