"""Detect a lasting change in the shape of the data near its end (fewer rows per period, fewer lines per order).

A sharp, lasting drop or jump in how many rows each period has, or how many rows each order has, usually means the
source changed (a new export, a narrower extract, one line per order instead of several), not the business. Totals
and growth figures that cross such a change compare unlike data, so the dashboard says so. Pure pandas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from analytics.descriptive import aggregate_time, datetime_series

MIN_BEFORE = 6          # complete periods needed before a change point
MIN_AFTER = 2           # the change must last at least this many complete periods
LOOKBACK = 12           # periods before the change used as the baseline
RATIO = 1.6             # a change of at least x1.6 (or /1.6) counts
_FREQ_WORD = {"M": "month", "W": "week", "D": "day"}


@dataclass(frozen=True)
class DataShift:
    start: pd.Timestamp            # first period after the change
    measure: str                   # "rows per order" | "rows per month" ...
    before: float
    after: float

    @property
    def ratio(self) -> float:
        return self.after / self.before if self.before else math.inf


def _change_point(values: pd.Series) -> tuple[int, float, float] | None:
    """(index of the first changed period, baseline median, median after) for the strongest lasting change near the end."""
    v = values.reset_index(drop=True)
    best = None
    for i in range(max(MIN_BEFORE, len(v) - 8), len(v) - MIN_AFTER + 1):
        before = v.iloc[max(0, i - LOOKBACK):i]
        after = v.iloc[i:]
        b, a = float(before.median()), float(after.median())
        if b <= 0 or a <= 0:
            continue
        r = a / b
        if not (r >= RATIO or r <= 1 / RATIO):
            continue
        # every later period must sit on the far side of the midpoint, so a single outlier does not count
        mid = (a + b) / 2
        if (r < 1 and (after > mid).any()) or (r > 1 and (after < mid).any()):
            continue
        if best is None or abs(math.log(r)) > abs(math.log(best[2] / best[1])):
            best = (i, b, a)
    return best


def detect_shifts(df: pd.DataFrame, date_col: str | None, order_col: str | None = None, freq: str = "M") -> list[DataShift]:
    """Lasting changes in rows per period and rows per order near the end of the data (complete periods only)."""
    if not date_col or date_col not in df.columns or len(df) == 0:
        return []
    try:
        t = aggregate_time(df, date_col, None, freq, "count")
    except Exception:
        return []
    t = t[t["complete"]].reset_index(drop=True)
    if len(t) < MIN_BEFORE + MIN_AFTER:
        return []
    out: list[DataShift] = []
    found = _change_point(t["rows"].astype(float))
    if found:
        i, b, a = found
        out.append(DataShift(t["period"].iloc[i], f"rows per {_FREQ_WORD.get(freq, 'period')}", b, a))
    if order_col and order_col in df.columns:
        per = datetime_series(df[date_col]).dt.to_period(freq)
        f = pd.DataFrame({"p": per, "o": df[order_col]}).dropna()
        g = f.groupby("p").agg(rows=("o", "size"), orders=("o", "nunique"))
        g.index = g.index.to_timestamp()
        g = g.reindex(t["period"]).dropna()
        g = g[g["orders"] > 0]
        if len(g) >= MIN_BEFORE + MIN_AFTER:
            found = _change_point(g["rows"] / g["orders"])
            if found:
                i, b, a = found
                out.append(DataShift(g.index[i], "rows per order", b, a))
    return out


def describe_shifts(shifts: list[DataShift]) -> str:
    """One sentence for the dashboard, or "" when nothing changed."""
    if not shifts:
        return ""
    first = min(s.start for s in shifts)
    parts = [f"{s.measure} went from {s.before:,.1f} to {s.after:,.1f}" if s.before < 100 else
             f"{s.measure} went from {s.before:,.0f} to {s.after:,.0f}" for s in shifts]
    return (f"The data changes shape from {first:%b %Y}: " + "; ".join(parts) + ". Totals and period-on-period changes "
            "after this point compare unlike data, so check whether the source export changed before reading them as "
            "business results.")
