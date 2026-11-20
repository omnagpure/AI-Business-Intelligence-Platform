"""Measures that are not a column but can be tracked per period: orders, on-time rate, shipping time, cancellations.

Each target is one value per row plus how rows combine into a period (sum or mean), so any time-series tool
(forecasting, unusual periods) can use it through ``aggregate_time`` like an ordinary column:
- Orders: 1 on the first row of each order, so the sum per period counts distinct orders (not order lines).
- On-time rate: 100 for an on-time row, 0 for a late one, empty when unknown or cancelled; the mean is the rate.
- Shipping or lead time, delay vs expected: days per row; the mean is the average.
- Cancelled rate: 100 or 0 per row with a status; the mean is the rate.
The definitions are the KPI engine's own (analytics.supply_chain_kpis), so a month's value matches the KPI on that month.
Pure pandas, no Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.kpis import DataFilter, _Ctx, _order_id
from config.analytics_config import DEFAULT_ANALYTICS

PREFIX = "KPI: "                   # marks a derived target in a list of column names


@dataclass(frozen=True)
class DerivedTarget:
    label: str                     # shown to the user, e.g. "KPI: On-time delivery rate (%)"
    values: pd.Series              # one value per row of the frame it was built from (NaN = not counted)
    agg: str                       # "sum" or "mean"
    unit: str                      # "percent" | "days" | "count"
    bounds: tuple[float, float] | None
    lower_is_better: bool
    note: str


def derived_targets(df: pd.DataFrame, model, config=DEFAULT_ANALYTICS) -> dict[str, DerivedTarget]:
    """Targets that apply to this data, keyed by label. Supply-chain measures only for supply-chain data."""
    from analytics.supply_chain_kpis import SHIPPING_WORDS, _on_time, _status, lead_days, lead_label
    ctx = _Ctx(df, model, config, DataFilter())
    out: dict[str, DerivedTarget] = {}

    def add(name, values, agg, unit, bounds, lower, note):
        if values is not None and values.notna().sum() > 0:
            label = PREFIX + name
            out[label] = DerivedTarget(label, values.reindex(df.index).astype("float64"), agg, unit, bounds, lower, note)

    oid = _order_id(ctx)
    if oid is not None:
        first = (~df[oid].duplicated()) & df[oid].notna()
        add("Orders", first.astype("float64"), "sum", "count", (0.0, float("inf")), False,
            f"Distinct values of '{oid}', counted in the period of each order's first row.")
    if not getattr(model, "is_supply_chain", False):
        return out
    try:
        on, delay, cols, how, _, _ = _on_time(ctx)
    except Exception:
        on, delay = None, None
    if on is not None:
        add("On-time delivery rate (%)", on.astype("Float64").mul(100), "mean", "percent", (0.0, 100.0), False,
            f"Share of rows delivered on time ({how}); cancelled or unknown rows are left out.")
        if delay is not None:
            add("Average delay vs expected (days)", delay, "mean", "days", None, True,
                "Actual minus expected days; negative means early.")
    try:
        days, dcols, _ = lead_days(ctx)
    except Exception:
        days, dcols = None, []
    if days is not None and len(days):
        add(f"Average {lead_label(dcols)} (days)", days, "mean", "days", (0.0, float("inf")), True,
            f"Days per row from {', '.join(dcols)}.")
    cancelled, _, scol = _status(ctx, SHIPPING_WORDS)
    if scol is not None:
        known = df[scol].notna()
        add("Cancelled rate (%)", cancelled.astype("float64").mul(100).where(known), "mean", "percent", (0.0, 100.0), True,
            f"Rows whose '{scol}' says cancelled.")
    return out


def frame_with(df: pd.DataFrame, target: DerivedTarget) -> pd.DataFrame:
    """A copy of `df` with the target as an extra column named after its label (the original frame is untouched)."""
    return df.assign(**{target.label: target.values.to_numpy()})
