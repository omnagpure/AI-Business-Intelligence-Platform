"""Sparkline and period change for a KPI tile, only where they can be computed honestly.

A trend is shown only for a KPI that is a plain SUM or AVG of one column with no extra row filters (ratios, rates and
filtered KPIs have no trend we could compute the same way), only from complete periods, and only when the data has at
least three of them. The change compares the latest complete period with the one before it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from analytics.descriptive import aggregate_time, datetime_series

_SIMPLE = re.compile(r"^\s*(SUM|AVG|COUNT DISTINCT)\(\s*([^()]+?)\s*\)\s*$", re.IGNORECASE)
_ROWS = re.compile(r"^\s*COUNT\(rows( after filters)?\)\s*$", re.IGNORECASE)
# KPIs where a rise is bad news, so the change is coloured the other way round.
_LOWER_IS_BETTER = ("cost", "lead", "stockout", "defect", "late", "delay", "return", "backorder", "damage", "cycle_time", "days")
MIN_PERIODS = 3
MAX_POINTS = 12


@dataclass(frozen=True)
class KpiTrend:
    points: list[float]            # complete periods, oldest first
    change_pct: float | None       # latest complete period vs the one before, in percent
    period_label: str              # "month", "week" or "day"


def lower_is_better(key: str) -> bool:
    k = key.lower()
    return any(word in k for word in _LOWER_IS_BETTER)


def _distinct(df: pd.DataFrame, date_col: str, col: str, freq: str) -> pd.DataFrame:
    """Distinct values of `col` per period, on the same periods (and completeness) as aggregate_time."""
    base = aggregate_time(df, date_col, None, freq, "count")
    d = datetime_series(df[date_col])
    keep = d.notna()
    counts = df.loc[keep, col].groupby(d[keep].dt.to_period(freq)).nunique()
    base["value"] = counts.reindex(pd.PeriodIndex(base["period"], freq=freq)).fillna(0).to_numpy(dtype="float64")
    return base


def kpi_trend(df: pd.DataFrame, formula: str, filters, date_col: str | None, freq: str = "M") -> KpiTrend | None:
    if not date_col or date_col not in df.columns or filters:
        return None
    try:
        if _ROWS.match(formula or ""):
            t = aggregate_time(df, date_col, None, freq, "count")
        else:
            m = _SIMPLE.match(formula or "")
            if m is None:
                return None
            fn, col = m.group(1).upper(), m.group(2).strip().strip('"`[]')
            if col not in df.columns or col == date_col:
                return None
            t = _distinct(df, date_col, col, freq) if fn == "COUNT DISTINCT" else aggregate_time(df, date_col, col, freq, "sum" if fn == "SUM" else "mean")
    except Exception:
        return None
    t = t[t["complete"] & (t["rows"] > 0)].dropna(subset=["value"])
    if len(t) < MIN_PERIODS:
        return None
    vals = [float(v) for v in t["value"].tail(MAX_POINTS)]
    if not all(math.isfinite(v) for v in vals):
        return None
    prev, last = vals[-2], vals[-1]
    change = (last - prev) / abs(prev) * 100 if prev not in (0, 0.0) else None
    return KpiTrend(vals, change, {"M": "month", "W": "week", "D": "day"}.get(freq, "period"))
