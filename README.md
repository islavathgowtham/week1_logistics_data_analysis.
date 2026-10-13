# Week 1: Strategic Planning and Data Exploration in Logistics

Python starter project for e-commerce logistics analytics: KPI calculation, exploratory data analysis, a baseline demand forecast, and product clustering.

## Scenario
An e-commerce retailer wants to understand delivery reliability, delivery time, shipping cost, and product demand to improve logistics planning.

## KPIs
- On-time delivery rate
- Median delivery time
- Shipping cost per order (if cost data is available)
- Forecast MAE/RMSE
- Product segments based on sales volume and order frequency

## Files
- `logistics_analysis.py` — analysis script
- `data/sample_orders.csv` — synthetic demo data
- `requirements.txt` — dependencies
- `outputs/` — generated charts and summaries

## Run
```bash
pip install -r requirements.txt
python logistics_analysis.py
```
To use another CSV:
```bash
python logistics_analysis.py --input path/to/orders.csv
```

## Expected columns
`order_id`, `order_date`, `promised_date`, `delivered_date`, `product_id`, `quantity`; optional: `delivery_zone`, `shipping_cost`. Update the script if your dataset uses different column names.

## Workflow
1. Load and profile data.
2. Clean duplicates, dates, and quantities.
3. Calculate logistics KPIs.
4. Explore daily orders and demand.
5. Train a simple time-ordered forecast when sufficient history exists.
6. Segment products using K-means.
7. Interpret results and recommend actions.

## Important notes
The included CSV is synthetic demonstration data, not company data. The forecast is educational and not production-ready. Route optimisation is proposed in the planning report but requires coordinates, vehicle capacities, delivery windows, and a distance/time matrix. Never commit confidential company data or customer personal information.

## Further resources
- Olist e-commerce dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
- Google OR-Tools routing: https://developers.google.com/optimization/routing
- scikit-learn: https://scikit-learn.org/stable/
