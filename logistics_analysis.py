import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean(path):
    df = pd.read_csv(path)
    required = {"order_id", "order_date", "product_id", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in ["order_date", "promised_date", "delivered_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.drop_duplicates().copy()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    return df[df["quantity"].notna() & (df["quantity"] >= 0)].copy()

def calculate_kpis(df):
    kpis = {
        "rows_after_cleaning": int(len(df)),
        "unique_orders": int(df["order_id"].nunique()),
        "total_units": float(df["quantity"].sum())
    }
    required_dates = {"order_date", "promised_date", "delivered_date"}
    if required_dates.issubset(df.columns):
        delivered = df[df["delivered_date"].notna()].copy()
        if not delivered.empty:
            delivered["on_time"] = delivered["delivered_date"] <= delivered["promised_date"]
            order_status = delivered.groupby("order_id")["on_time"].all()
            kpis["on_time_delivery_rate_pct"] = round(float(order_status.mean() * 100), 2)
            delivered["delivery_days"] = (delivered["delivered_date"] - delivered["order_date"]).dt.total_seconds() / 86400
            kpis["median_delivery_days"] = round(float(delivered.groupby("order_id")["delivery_days"].max().median()), 2)
    if "shipping_cost" in df.columns:
        cost = pd.to_numeric(df["shipping_cost"], errors="coerce").fillna(0)
        order_cost = df.assign(_cost=cost).groupby("order_id")["_cost"].sum()
        kpis["shipping_cost_total"] = round(float(order_cost.sum()), 2)
        kpis["shipping_cost_per_order"] = round(float(order_cost.sum() / max(df["order_id"].nunique(), 1)), 2)
    return kpis

def make_charts(df, out):
    dated = df.dropna(subset=["order_date"]).copy()
    if dated.empty:
        return
    daily_orders = dated.groupby(dated["order_date"].dt.floor("D"))["order_id"].nunique()
    daily_orders.plot(figsize=(10, 4), title="Daily Unique Order Volume")
    plt.xlabel("Date"); plt.ylabel("Unique orders"); plt.tight_layout()
    plt.savefig(out / "daily_order_volume.png", dpi=150); plt.close()
    daily_demand = dated.groupby(dated["order_date"].dt.floor("D"))["quantity"].sum()
    daily_demand.plot(figsize=(10, 4), title="Daily Demand (Units)")
    plt.xlabel("Date"); plt.ylabel("Units"); plt.tight_layout()
    plt.savefig(out / "daily_demand.png", dpi=150); plt.close()

def forecast(df, out):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    dated = df.dropna(subset=["order_date"]).copy()
    if dated.empty:
        return None
    daily = dated.groupby(dated["order_date"].dt.floor("D"))["quantity"].sum().rename("demand")
    daily = daily.asfreq("D", fill_value=0).to_frame()
    for lag in [1, 7, 14]:
        daily[f"lag_{lag}"] = daily["demand"].shift(lag)
    daily["day_of_week"] = daily.index.dayofweek
    daily = daily.dropna()
    if len(daily) < 35:
        print("Skipping forecast: not enough usable daily history.")
        return None
    split = int(len(daily) * .8)
    train, test = daily.iloc[:split], daily.iloc[split:]
    features = ["lag_1", "lag_7", "lag_14", "day_of_week"]
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(train[features], train["demand"])
    pred = model.predict(test[features])
    result = {
        "test_days": int(len(test)),
        "mae": round(float(mean_absolute_error(test["demand"], pred)), 3),
        "rmse": round(float(mean_squared_error(test["demand"], pred) ** .5), 3),
        "note": "Educational baseline only; compare with simpler baselines before business use."
    }
    (out / "forecast_metrics.json").write_text(json.dumps(result, indent=2))
    return result

def cluster_products(df, out):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    features = df.groupby("product_id").agg(
        units_sold=("quantity", "sum"),
        order_frequency=("order_id", "nunique")
    ).fillna(0)
    if len(features) < 3:
        print("Skipping clustering: need at least 3 products.")
        return
    X = StandardScaler().fit_transform(features[["units_sold", "order_frequency"]])
    features["segment"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    features.to_csv(out / "product_segments.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_orders.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(Path(args.input))
    print("Rows after cleaning:", len(df))
    print("Missing values:\\n", df.isna().sum().to_string())
    kpis = calculate_kpis(df)
    (out / "kpi_summary.json").write_text(json.dumps(kpis, indent=2))
    print("KPI summary:\\n", json.dumps(kpis, indent=2))
    make_charts(df, out)
    try:
        result = forecast(df, out)
        if result: print("Forecast metrics:", result)
        cluster_products(df, out)
    except ImportError:
        print("Install requirements.txt to enable forecasting and clustering.")
    print("Outputs saved to", out.resolve())

if __name__ == "__main__":
    main()
