import os
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def make_synthetic_business_df(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365, size=rows)]
    products = rng.choice(["Alpha", "Beta", "Gamma", "Delta"], size=rows, replace=True)
    regions = rng.choice(["NA", "EU", "APAC", "LATAM"], size=rows, replace=True)
    channel = rng.choice(["Online", "Retail", "Wholesale"], size=rows, replace=True)
    units = rng.integers(1, 200, size=rows)
    price = rng.uniform(5, 150, size=rows)
    discount = rng.uniform(0, 0.35, size=rows)
    marketing = rng.uniform(0, 5000, size=rows)
    cust_age = rng.normal(35, 10, size=rows).clip(18, 80)
    repeat = rng.choice([0, 1], size=rows, p=[0.6, 0.4])
    returned = rng.choice([0, 1], size=rows, p=[0.95, 0.05])

    df = pd.DataFrame({
        "date": dates,
        "product": products,
        "region": regions,
        "channel": channel,
        "units_sold": units,
        "unit_price": price,
        "discount_rate": discount,
        "marketing_spend": marketing,
        "customer_age": cust_age,
        "repeat_customer": repeat,
        "returned": returned,
    })
    df["revenue"] = df["units_sold"] * df["unit_price"] * (1 - df["discount_rate"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic business dataset")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("data", "synthetic_business_data.csv"),
        help="Output CSV path (default: data/synthetic_business_data.csv)"
    )
    args = parser.parse_args()

    # Ensure the output directory exists (relative to project root)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = make_synthetic_business_df(args.rows)
    df.to_csv(args.out, index=False)
    print(f"Saved synthetic dataset to: {args.out}")


if __name__ == "__main__":
    main()
