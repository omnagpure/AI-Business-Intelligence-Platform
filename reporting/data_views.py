"""Extra views of the analysed data for the report: how the headline KPIs moved month by month, the latest period
against the one before, the dashboard's breakdown views, and the series drawn on the cover.

Everything reuses the app's own engines (compute_kpis, kpi_history, suggest_views, aggregate_time), so a number in
the report means exactly what it means on the Dashboard. Each part fails on its own and is then simply absent.
No Streamlit, no I/O.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from analytics.descriptive import aggregate_time, datetime_series
from core.logging_config import get_logger

log = get_logger("reporting.views")

NO_COMPARE = ("row_count", "revenue_growth", "revenue_by_category")      # a share or growth of a period is not a level
MAX_HISTORY = 24


@dataclass(frozen=True)
class Comparison:
    window: str                                   # "12 months"
    recent_label: str                             # "Feb 2017 – Jan 2018"
    prior_label: str
    rows: tuple[tuple[str, str, str, float, float], ...]     # (key, name, unit, prior, recent)

    def get(self, key: str):
        return next((r for r in self.rows if r[0] == key), None)


@dataclass(frozen=True)
class Extras:
    date_col: str | None = None
    histories: dict = field(default_factory=dict)            # kpi key -> KpiHistory
    comparison: Comparison | None = None
    views: tuple = ()                                         # SuggestedView
    cover_series: tuple[float, ...] = ()


def _months(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def _label(start: pd.Timestamp, end_excl: pd.Timestamp) -> str:
    last = end_excl - pd.Timedelta(days=1)
    return f"{start:%b %Y} – {last:%b %Y}" if (start.year, start.month) != (last.year, last.month) else f"{start:%b %Y}"


def compare_periods(df, model, config, date_col: str, keys: tuple[str, ...]) -> Comparison | None:
    """The latest complete window (12 months with two years of data, else 6 or 3) against the window before it,
    each recomputed by the KPI engine. Only KPIs that compute in both windows are kept."""
    from analytics.kpis import OK, DataFilter, compute_kpis
    d = datetime_series(df[date_col]).dropna()
    if d.empty:
        return None
    first, last = d.min(), d.max()
    end = pd.Timestamp(last.year, last.month, 1)                       # drop the last month unless it is complete
    if (last + pd.Timedelta(days=1)).month != last.month:
        end = end + pd.DateOffset(months=1)
    span = _months(pd.Timestamp(first.year, first.month, 1), end)
    w = 12 if span >= 24 else 6 if span >= 12 else 3 if span >= 6 else 0
    if not w:
        return None
    r_start = end - pd.DateOffset(months=w)
    p_start = r_start - pd.DateOffset(months=w)
    day = pd.Timedelta(days=1)

    def run(a, b):
        flt = DataFilter(date_col=date_col, date_start=a.date().isoformat(), date_end=(b - day).date().isoformat())
        return compute_kpis(df, model, config, flt)
    recent, prior = run(r_start, end), run(p_start, r_start)
    rows = []
    for k in keys:
        a, b = prior.get(k), recent.get(k)
        if k in NO_COMPARE or a is None or b is None or a.status != OK or b.status != OK:
            continue
        if a.value is None or b.value is None or not (math.isfinite(a.value) and math.isfinite(b.value)):
            continue
        rows.append((k, b.name, b.unit, float(a.value), float(b.value)))
    if not rows:
        return None
    return Comparison(f"{w} months", _label(r_start, end), _label(p_start, r_start), tuple(rows))


def change(unit: str, prior: float, recent: float) -> tuple[float | None, str]:
    """(change, unit): percentage points for percent KPIs, relative percent otherwise (None when prior is 0)."""
    if unit == "percent":
        return recent - prior, "pts"
    return ((recent - prior) / abs(prior) * 100 if prior else None), "%"


def _cover_series(df, model, kpis, date_col: str) -> tuple[float, ...]:
    """Monthly total of the main measure (revenue, else units, else row count), complete months only."""
    col = None
    if kpis is not None:
        for key in ("total_revenue", "total_spend", "units"):
            r = kpis.get(key)
            if r is not None and r.status == "ok" and r.columns and r.columns[0] in df.columns:
                col = r.columns[0]
                break
    t = aggregate_time(df, date_col, col, "M", "sum" if col else "count")
    t = t[t["complete"] & (t["rows"] > 0)]
    return tuple(float(v) for v in t["value"].tail(48) if pd.notna(v))


def gather(df, model, config, kpis, keys: tuple[str, ...]) -> Extras:
    """Collect every extra view that can be computed; missing pieces are left empty."""
    if df is None or model is None or len(df) == 0:
        return Extras()
    from analytics.kpi_trend import kpi_history
    from analytics.kpis import primary_date_column
    from analytics.supply_chain_views import suggest_views
    date = None
    try:
        date = primary_date_column(df, model)
    except Exception:
        log.exception("No primary date column")
    hist, comp, views, cover = {}, None, (), ()
    if date:
        try:
            hist = kpi_history(df, model, config, date, "M", keys=keys, max_points=MAX_HISTORY)
        except Exception:
            log.exception("KPI history failed")
        try:
            all_keys = tuple(dict.fromkeys(keys + tuple(r.key for r in (kpis.ok() if kpis is not None else []))))
            comp = compare_periods(df, model, config, date, all_keys)
        except Exception:
            log.exception("Period comparison failed")
        try:
            cover = _cover_series(df, model, kpis, date)
        except Exception:
            log.exception("Cover series failed")
    try:
        views = suggest_views(df, model, config, max_views=4, skip=("trend",))
    except Exception:
        log.exception("Breakdown views failed")
    return Extras(date, hist, comp, tuple(views), cover)
