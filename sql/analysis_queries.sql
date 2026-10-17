-- SQL examples for the retail customer dataset.
-- Import data/retail_customers.csv into a table named retail_customers first.

-- 1. Repeat-purchase rate by region
SELECT region,
       COUNT(*) AS customers,
       ROUND(AVG(repeat_purchase_next_30d) * 100, 2) AS repeat_purchase_rate_pct
FROM retail_customers
GROUP BY region
ORDER BY repeat_purchase_rate_pct DESC;

-- 2. Average order value by customer activity segment
SELECT CASE
         WHEN orders_last_6m <= 1 THEN 'Low (0-1)'
         WHEN orders_last_6m <= 4 THEN 'Medium (2-4)'
         ELSE 'High (5+)'
       END AS activity_segment,
       COUNT(*) AS customers,
       ROUND(AVG(avg_order_value), 2) AS average_order_value
FROM retail_customers
GROUP BY activity_segment
ORDER BY average_order_value DESC;

-- 3. Customers who may need re-engagement (example rule-based segment)
SELECT customer_id, region, orders_last_6m, days_since_last_order,
       avg_order_value
FROM retail_customers
WHERE days_since_last_order >= 60
  AND orders_last_6m <= 2
ORDER BY days_since_last_order DESC;

-- 4. Data quality check
SELECT COUNT(*) AS total_rows,
       SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS missing_customer_ids,
       SUM(CASE WHEN avg_order_value < 0 THEN 1 ELSE 0 END) AS invalid_order_values
FROM retail_customers;
