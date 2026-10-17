"""Interactive Streamlit dashboard for retail analytics and repeat-purchase prediction."""
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "retail_customers.csv"
FEATURES = [
    "age", "region", "orders_last_6m", "avg_order_value",
    "days_since_last_order", "discount_usage_rate", "returns_last_6m",
    "site_visits_last_30d", "support_tickets_last_6m"
]
TARGET = "repeat_purchase_next_30d"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    numeric = [c for c in FEATURES if c != "region"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["region"])
    ])
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=42
        ))
    ])
    pipe.fit(X, y)
    return pipe

st.set_page_config(page_title="Retail Analytics & ML", page_icon="📊", layout="wide")
st.title("📊 Retail Sales Analytics & Customer Prediction")
st.caption("Portfolio demo using synthetic data only — not real customer records.")
df = load_data()

with st.sidebar:
    st.header("Filters")
    regions = st.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    min_orders, max_orders = int(df.orders_last_6m.min()), int(df.orders_last_6m.max())
    order_range = st.slider("Orders in last 6 months", min_orders, max_orders, (min_orders, max_orders))

filtered = df[
    df.region.isin(regions)
    & df.orders_last_6m.between(order_range[0], order_range[1])
]
if filtered.empty:
    st.warning("No customers match these filters. Adjust the sidebar.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(filtered):,}")
c2.metric("Average order value", f"₹{filtered.avg_order_value.mean():,.0f}")
c3.metric("Repeat-purchase rate", f"{filtered[TARGET].mean():.1%}")
c4.metric("Orders per customer", f"{filtered.orders_last_6m.mean():.1f}")

left, right = st.columns(2)
with left:
    st.subheader("Customers by region")
    st.bar_chart(filtered["region"].value_counts().sort_index())
with right:
    st.subheader("Repeat-purchase rate by region")
    region_rate = filtered.groupby("region")[TARGET].mean().sort_values(ascending=False)
    st.bar_chart(region_rate)

st.subheader("Customer activity")
activity = filtered.copy()
activity["activity_segment"] = pd.cut(
    activity.orders_last_6m, bins=[-1, 1, 4, 20],
    labels=["Low (0-1)", "Medium (2-4)", "High (5+)"]
)
segment = activity.groupby("activity_segment", observed=False).agg(
    customers=("customer_id", "count"),
    average_order_value=("avg_order_value", "mean"),
    repeat_purchase_rate=(TARGET, "mean")
).round(3)
st.dataframe(segment, use_container_width=True)

st.subheader("Try a repeat-purchase prediction")
st.write("Change the example customer features below. The model is trained on the included synthetic dataset.")
model = train_model(df)
p1, p2, p3 = st.columns(3)
with p1:
    age = st.slider("Age", 18, 70, 32)
    region = st.selectbox("Region", sorted(df.region.unique()))
    orders = st.slider("Orders in last 6 months", 0, 15, 4)
with p2:
    aov = st.number_input("Average order value (₹)", min_value=150.0, max_value=4500.0, value=1500.0, step=50.0)
    days = st.slider("Days since last order", 0, 180, 25)
    discount = st.slider("Discount usage rate", 0.0, 1.0, 0.25, 0.05)
with p3:
    returns = st.slider("Returns in last 6 months", 0, 5, 0)
    visits = st.slider("Site visits in last 30 days", 0, 35, 6)
    tickets = st.slider("Support tickets in last 6 months", 0, 6, 0)

if st.button("Predict repeat-purchase likelihood", type="primary"):
    customer = pd.DataFrame([{
        "age": age, "region": region, "orders_last_6m": orders,
        "avg_order_value": aov, "days_since_last_order": days,
        "discount_usage_rate": discount, "returns_last_6m": returns,
        "site_visits_last_30d": visits, "support_tickets_last_6m": tickets
    }], columns=FEATURES)
    probability = model.predict_proba(customer)[0, 1]
    st.metric("Model-estimated likelihood", f"{probability:.1%}")
    st.info("This is a demonstration model trained on synthetic data. Do not use it to make real customer decisions.")

with st.expander("View filtered sample data"):
    st.dataframe(filtered.head(100), use_container_width=True)
st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_retail_customers.csv",
    mime="text/csv"
)
