"""Exploratory analysis and baseline metrics for the synthetic retail dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "retail_customers.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    print("\n--- Dataset shape ---")
    print(df.shape)
    print("\n--- Missing values ---")
    print(df.isna().sum())
    print("\n--- Summary statistics ---")
    print(df.describe(include="all").T)
    print("\n--- Repeat purchase rate by region ---")
    print(df.groupby("region")["repeat_purchase_next_30d"].mean().sort_values(ascending=False))
    print("\n--- Customer activity segments ---")
    df["activity_segment"] = pd.cut(
        df["orders_last_6m"], bins=[-1, 1, 4, 20],
        labels=["Low (0-1)", "Medium (2-4)", "High (5+)"]
    )
    summary = df.groupby("activity_segment", observed=False).agg(
        customers=("customer_id", "count"),
        avg_order_value=("avg_order_value", "mean"),
        repeat_purchase_rate=("repeat_purchase_next_30d", "mean")
    ).round(3)
    summary.to_csv(REPORTS / "activity_segment_summary.csv")
    print(summary)
    print(f"\nSaved segment summary to {REPORTS / 'activity_segment_summary.csv'}")

if __name__ == "__main__":
    main()
