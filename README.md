# Retail Sales Analytics & Customer Prediction

An end-to-end portfolio project combining **Python, Pandas, SQL, exploratory data analysis, machine learning, and an interactive Streamlit dashboard**.

> **Data notice:** `data/retail_customers.csv` is synthetic demonstration data generated for this portfolio. It does not represent real customers or real business outcomes. Model metrics on this dataset must not be presented as real-world performance.

## Project goals
- Analyze customer activity and retail purchasing patterns.
- Compare repeat-purchase rates across regions and customer activity segments.
- Train a baseline Random Forest classifier to estimate whether a customer will make a repeat purchase in the next 30 days.
- Present insights and an interactive prediction demo in a Streamlit dashboard.
- Demonstrate basic SQL aggregation and data-quality queries.

## Tech stack
- Python, Pandas, NumPy
- SQL (example queries in `sql/analysis_queries.sql`)
- scikit-learn: preprocessing pipeline, train/test split, Random Forest, classification metrics
- Streamlit for the dashboard

## Repository structure
```text
retail-analytics-ml/
├── app.py
├── data/
│   └── retail_customers.csv
├── reports/
├── sql/
│   └── analysis_queries.sql
├── src/
│   ├── analysis.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Run locally
Use Python 3.10+.

### 1. Create and activate a virtual environment
Windows:
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages
```bash
python -m pip install -r requirements.txt
```

### 3. Run exploratory analysis
```bash
python src/analysis.py
```

### 4. Train and evaluate the baseline model
```bash
python src/train_model.py
```
Metrics are written to `reports/model_metrics.json` and `reports/classification_report.txt`.

### 5. Launch the dashboard
```bash
streamlit run app.py
```
Your terminal will show a local URL (normally `http://localhost:8501`). Open it in a browser.

## Dataset fields
- `customer_id`: synthetic customer identifier
- `age`, `region`: basic customer attributes
- `orders_last_6m`, `avg_order_value`: purchasing activity
- `days_since_last_order`: recency
- `discount_usage_rate`, `returns_last_6m`: purchase behavior
- `site_visits_last_30d`, `support_tickets_last_6m`: engagement/support indicators
- `repeat_purchase_next_30d`: synthetic binary target (0 = no, 1 = yes)

## How to discuss this project in an interview
1. **Business question:** Which customer activity patterns are associated with repeat purchasing?
2. **Analytics:** I explored customer activity and compared repeat-purchase rates across regions and activity segments.
3. **Model:** I built a preprocessing + Random Forest pipeline and evaluated it on a held-out test set using precision, recall, F1 and ROC-AUC.
4. **Limitations:** The data and labels are synthetic. Results only demonstrate the workflow; the model is not validated for real retail decisions.
5. **Next improvements:** Use a public real-world dataset, add unit tests, compare logistic regression with Random Forest, tune the decision threshold, and deploy the dashboard.

## Before publishing
- Run every command and confirm the app works on your computer.
- Add screenshots of your own running dashboard.
- Update this README with actual findings after running the analysis.
- Do not claim the model is production-ready or that it improves real revenue.
