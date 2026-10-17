"""Train and evaluate a simple repeat-purchase classification model."""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "retail_customers.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

FEATURES = [
    "age", "region", "orders_last_6m", "avg_order_value",
    "days_since_last_order", "discount_usage_rate", "returns_last_6m",
    "site_visits_last_30d", "support_tickets_last_6m"
]
TARGET = "repeat_purchase_next_30d"

def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    numeric = [c for c in FEATURES if c != "region"]
    categorical = ["region"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_rows": int(len(y_test)),
        "note": "Synthetic demonstration data; metrics are not evidence of real-world business performance."
    }
    (REPORTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "classification_report.txt").write_text(classification_report(y_test, pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))
    print(f"\nSaved metrics to {REPORTS / 'model_metrics.json'}")

if __name__ == "__main__":
    main()
