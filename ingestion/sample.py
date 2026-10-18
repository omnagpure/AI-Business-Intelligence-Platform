"""Small synthetic dataset for first-time users. Always labelled as synthetic in the UI."""
from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_LABEL = "Synthetic sample data (randomly generated, not real)"


def make_sample_sales(seed: int = 42, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    cat = rng.choice(["Electronics", "Home", "Toys", "Clothing", "home "], n, p=[.3, .25, .15, .25, .05])
    qty = rng.integers(1, 8, n)
    price = rng.choice([9.99, 24.5, 49.0, 120.0], n)
    df = pd.DataFrame({
        "order_id": [f"ORD{i:05d}" for i in range(n)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "customer_id": rng.integers(1000, 1150, n),
        "product_category": cat,
        "quantity": qty,
        "unit_price": price,
        "revenue": (qty * price).round(2),
        "region": rng.choice(["North", "South", "East", "West"], n),
    })
    df.loc[rng.choice(n, 30, replace=False), "region"] = None
    return pd.concat([df, df.iloc[:5]], ignore_index=True)
