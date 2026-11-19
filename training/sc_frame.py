"""Builds a small synthetic table whose columns carry the supply-chain headers and role-typical values.

Used to test the semantic layer end to end (header + value profile), which is closer to real use than names alone.
The values are random and exist only to give each column a plausible shape (ids unique, dates parse, flags 0/1 ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLE_SHAPE = {"identifier": "id", "datetime": "date", "supplier": "code", "warehouse": "code", "product": "code", "customer": "code",
              "quantity": "int", "cost": "money", "revenue": "money", "balance": "money", "duration": "days", "category": "cat",
              "percentage": "pct", "flag": "bit", "location": "place", "price": "money", "profit": "money"}


def make_frame(pairs: list[tuple[str, str]], n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {}
    for name, role in pairs:
        shape = ROLE_SHAPE.get(role, "int")
        if shape == "id":
            cols[name] = [f"{name[:2].upper()}{i:05d}" for i in range(n)]
        elif shape == "date":
            cols[name] = (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")).strftime("%Y-%m-%d")
        elif shape == "code":
            cols[name] = rng.choice([f"{name[:1].upper()}{k}" for k in range(1, 9)], n)
        elif shape == "int":
            cols[name] = rng.integers(0, 500, n)
        elif shape == "money":
            cols[name] = rng.uniform(5, 900, n).round(2)
        elif shape == "days":
            cols[name] = rng.integers(1, 40, n)
        elif shape == "cat":
            cols[name] = rng.choice(["A", "B", "C", "D"], n)
        elif shape == "pct":
            cols[name] = rng.uniform(60, 100, n).round(1)
        elif shape == "bit":
            cols[name] = rng.choice([0, 1], n)
        else:
            cols[name] = rng.choice(["North", "South", "East", "West", "Central"], n)
    return pd.DataFrame(cols)
