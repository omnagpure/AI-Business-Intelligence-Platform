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

# Per period length: how far back from the end a change is looked for (about 8 months), how long it must last, how many
# periods before it are needed, and how many form the baseline.
SEARCH = {"M": 8, "W": 36, "D": 245}
MIN_AFTER = {"M": 2, "W": 4, "D": 21}
MIN_BEFORE = {"M": 6, "W": 12, "D": 42}
LOOKBACK = {"M": 12, "W": 26, "D": 90}
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

    def change_text(self) -> str:
        """'rows per order went from 3.0 to 1.0' (one decimal below 100, none above)."""
        f = (lambda v: f"{v:,.1f}") if self.before < 100 else (lambda v: f"{v:,.0f}")
        return f"{self.measure} went from {f(self.before)} to {f(self.after)}"


def _change_point(values: pd.Series, freq: str = "M") -> tuple[int, float, float] | None:
    """(index of the first changed period, baseline median, median after) for the strongest lasting change near the end."""
    v = values.reset_index(drop=True)
    best = None
    for i in range(max(MIN_BEFORE[freq], len(v) - SEARCH[freq]), len(v) - MIN_AFTER[freq] + 1):
        before = v.iloc[max(0, i - LOOKBACK[freq]):i]
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
    if freq not in SEARCH:
        return []
    t = t[t["complete"]].reset_index(drop=True)
    if len(t) < MIN_BEFORE[freq] + MIN_AFTER[freq]:
        return []
    out: list[DataShift] = []
    found = _change_point(t["rows"].astype(float), freq)
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
        if len(g) >= MIN_BEFORE[freq] + MIN_AFTER[freq]:
            found = _change_point(g["rows"] / g["orders"], freq)
            if found:
                i, b, a = found
                out.append(DataShift(g.index[i], "rows per order", b, a))
    return out


def describe_shifts(shifts: list[DataShift]) -> str:
    """One sentence for the dashboard, or "" when nothing changed."""
    if not shifts:
        return ""
    first = min(s.start for s in shifts)
    parts = [s.change_text() for s in shifts]
    return (f"The data changes shape from {first:%b %Y}: " + "; ".join(parts) + ". Totals and period-on-period changes "
            "after this point compare unlike data, so check whether the source export changed before reading them as "
            "business results.")
