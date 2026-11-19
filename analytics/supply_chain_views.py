"""Suggested dashboard views. Pure pandas, no Streamlit or Plotly.

Each view is chosen from the roles present, carries the small table it plots, and a one-line caption computed from
that table (nothing is typed in). A view is returned only when its data exists; otherwise it is simply absent.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.descriptive import aggregate_time, top_n_with_other
from analytics.kpis import DataFilter, _Ctx, _category, _revenue, _stock, _units, primary_date_column
from analytics.supply_chain_kpis import (DEFECT_WORDS, SC, _cols, _on_time, _reorder, _status, lead_days,
                                          supplier_shares)
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.logging_config import get_logger
from semantic.semantic_model import SemanticModel

log = get_logger("analytics.views")
MIN_GROUP_ROWS = 10
MIN_GROUP_ROWS_SMALL = 5
MAX_SUPPLIER_GROUPS = 1000      # a supplier column may have many values; only the lowest 10 are drawn
GROUP_HINT = frozenset({"supplier", "vendor", "mode", "shipping", "shipment", "carrier", "method", "warehouse"})


@dataclass(frozen=True)
class SuggestedView:
    key: str
    title: str
    kind: str                      # histogram | bar | line | lines
    data: pd.DataFrame
    x: str                         # x column (label column for bars, "days" for the histogram)
    y: tuple[str, ...]             # value column(s)
    caption: str
    columns: tuple[str, ...]       # source columns, for transparency


def _group_col(ctx: _Ctx, min_rows_per_group: float = 3.0) -> str | None:
    """A column to compare groups on: supplier, then shipping mode / carrier / warehouse, then any category."""
    cands = []
    for role in ("supplier", "category", "warehouse", "location"):
        for c in ctx.rank(role, numeric=False):
            n = ctx.df[c].nunique(dropna=True)
            cap = max(ctx.config.max_group_cardinality, MAX_SUPPLIER_GROUPS) if role == "supplier" else ctx.config.max_group_cardinality
            if 2 <= n <= cap and "status" not in ctx.tokens(c) \
                    and len(ctx.df) / n >= min_rows_per_group:
                cands.append((-bool(ctx.tokens(c) & GROUP_HINT), role != "supplier", c))
    cands.sort()
    return cands[0][2] if cands else None


def _on_time_by_group(ctx: _Ctx) -> SuggestedView | None:
    on, _, cols, _, _, _ = _on_time(ctx)
    g = _group_col(ctx)
    if on is None or g is None:
        return None
    f = pd.DataFrame({"g": ctx.df[g].astype("string"), "on": on}).dropna()
    if f.empty:
        return None
    f["on"] = f["on"].astype(int)
    agg = f.groupby("g")["on"].agg(rate=lambda s: 100 * s.mean(), rows="size")
    for floor in (MIN_GROUP_ROWS, MIN_GROUP_ROWS_SMALL):
        kept = agg[agg["rows"] >= floor]
        if len(kept) >= 2:
            break
    else:
        return None
    low = kept.sort_values("rate")
    shown = low.head(10).reset_index().rename(columns={"g": "label", "rate": "value"})
    overall = 100 * f["on"].mean()
    worst, best = low.iloc[0], low.iloc[-1]
    cap = (f"{low.index[0]} has the lowest on-time rate ({worst['rate']:.1f}% of {int(worst['rows']):,} rows) against "
           f"{overall:.1f}% overall; the best is {low.index[-1]} ({best['rate']:.1f}%). Groups with at least {floor} rows.")
    return SuggestedView("on_time_by_group", f"Lowest on-time rate by {g}", "bar", shown[["label", "value", "rows"]],
                         "label", ("value",), cap, tuple(cols + [g]))


def _lead_time_distribution(ctx: _Ctx) -> SuggestedView | None:
    d, cols, _ = lead_days(ctx)
    if d is None or len(d) < 5:
        return None
    p50, p90 = float(d.median()), float(d.quantile(0.9))
    cap = f"Median lead time is {p50:.1f} days and 90% of rows arrive within {p90:.1f} days ({len(d):,} rows)."
    return SuggestedView("lead_time", "Lead time distribution (days)", "histogram", pd.DataFrame({"days": d.to_numpy()}),
                         "days", ("days",), cap, tuple(cols))


def _stock_vs_reorder(ctx: _Ctx) -> SuggestedView | None:
    stock, rp, date = _stock(ctx), _reorder(ctx), primary_date_column(ctx.df, ctx.model)
    if None in (stock, rp, date):
        return None
    f = pd.DataFrame({"date": ctx.dt(date).dt.normalize(), "stock": ctx.num(stock), "reorder_point": ctx.num(rp)}).dropna()
    per = f.groupby("date")[["stock", "reorder_point"]].sum()
    if len(per) < 2:
        return None
    below = per["stock"] < per["reorder_point"]
    last = per.iloc[-1]
    cap = (f"Summed over all items, stock was below the reorder points on {int(below.sum()):,} of {len(per):,} dates "
           f"({100 * below.mean():.1f}%); on the latest date it was {last['stock']:,.0f} against {last['reorder_point']:,.0f}.")
    plot = per if len(per) <= 150 else per.resample("W").mean().dropna()
    plot = plot.reset_index()
    return SuggestedView("stock_vs_reorder", "Stock vs reorder point over time", "lines", plot, "date",
                         ("stock", "reorder_point"), cap, (stock, rp, date))


def _defect_cols(ctx: _Ctx) -> tuple[str | None, str | None]:
    d = _cols(ctx, ("quantity",), lambda t: bool(t & DEFECT_WORDS), True)
    u, _ = _units(ctx)
    return (d[0] if d else None), u


def _defects(ctx: _Ctx) -> SuggestedView | None:
    d, u = _defect_cols(ctx)
    if d is None or u is None:
        return None
    f = pd.DataFrame({"d": ctx.num(d), "u": ctx.num(u)}).dropna()
    if f["u"].sum() <= 0:
        return None
    overall = 100 * f["d"].sum() / f["u"].sum()
    date = primary_date_column(ctx.df, ctx.model)
    if date is not None:
        dt = ctx.dt(date).reindex(f.index)
        ok = dt.notna()
        span = (dt[ok].max() - dt[ok].min()).days if ok.any() else 0
        freq = "M" if span >= 120 else "W" if span >= 28 else "D"
        if ok.sum() and f[ok]["u"].sum() > 0:
            p = f[ok].groupby(dt[ok].dt.to_period(freq).dt.start_time).sum()
            p = p[p["u"] > 0]
            if len(p) >= 2:
                p["rate"] = 100 * p["d"] / p["u"]
                top = p["rate"].idxmax()
                cap = (f"Defect rate ranged from {p['rate'].min():.1f}% to {p['rate'].max():.1f}% by "
                       f"{ {'M': 'month', 'W': 'week', 'D': 'day'}[freq] }, highest in {top:%Y-%m-%d}; {overall:.1f}% overall.")
                if (p["rate"] > 100).any():
                    cap += " Some periods exceed 100%, so check the data."
                return SuggestedView("defects_over_time", "Defect rate over time", "line",
                                     p.rename_axis("period").reset_index()[["period", "rate"]],
                                     "period", ("rate",), cap, (d, u, date))
    g = _group_col(ctx)
    if g is None:
        return None
    f["g"] = ctx.df[g].astype("string").reindex(f.index)
    a = f.dropna(subset=["g"]).groupby("g").agg(d=("d", "sum"), u=("u", "sum"), rows=("d", "size"))
    a = a[(a["u"] > 0) & (a["rows"] >= 3)]
    if len(a) < 2:
        return None
    a["rate"] = 100 * a["d"] / a["u"]
    a = a.sort_values("rate", ascending=False)
    cap = f"{a.index[0]} has the highest defect rate ({a['rate'].iloc[0]:.1f}%) against {overall:.1f}% overall."
    shown = a.head(10).reset_index().rename(columns={"g": "label", "rate": "value"})
    return SuggestedView("defects_by_group", f"Highest defect rate by {g}", "bar", shown[["label", "value", "rows"]],
                         "label", ("value",), cap, (d, u, g))


def _supplier_share(ctx: _Ctx) -> SuggestedView | None:
    g, label, cols, _ = supplier_shares(ctx)
    if g is None:
        return None
    share = 100 * g / g.sum()
    shown = top_n_with_other(g, ctx.config.top_n)
    t = pd.DataFrame({"label": shown.index.astype(str), "value": 100 * shown.to_numpy() / g.sum()})
    cap = (f"{g.index[0]} accounts for {share.iloc[0]:.1f}% of {label}; the top 3 of {len(g):,} suppliers account for "
           f"{share.iloc[:3].sum():.1f}%.")
    return SuggestedView("supplier_share", f"Share of {label} by supplier", "bar", t, "label", ("value",), cap, tuple(cols))


def _measure(ctx: _Ctx):
    rev, _ = _revenue(ctx)
    if rev is not None:
        return ctx.num(rev), rev
    u, _ = _units(ctx)
    if u is not None:
        return ctx.num(u), u
    return None, None


def _trend(ctx: _Ctx) -> SuggestedView | None:
    m, name = _measure(ctx)
    date = primary_date_column(ctx.df, ctx.model)
    if m is None or date is None:
        return None
    d = ctx.dt(date)
    span = (d.max() - d.min()).days if d.notna().any() else 0
    freq = "M" if span >= 120 else "W" if span >= 28 else "D"
    t = aggregate_time(pd.DataFrame({date: ctx.df[date], name: ctx.df[name]}), date, name, freq, "sum")
    if len(t) < 2:
        return None
    done = t[t["complete"]]
    peak = t.loc[t["value"].idxmax()]
    cap = f"{name} peaked at {peak['value']:,.0f} in the period starting {peak['period']:%Y-%m-%d}."
    if not done.empty:
        cap += f" The latest complete period was {done.iloc[-1]['value']:,.0f}."
    return SuggestedView("trend", f"{name} over time", "line", t[["period", "value"]], "period", ("value",), cap, (name, date))


def _category_share(ctx: _Ctx) -> SuggestedView | None:
    m, name = _measure(ctx)
    cat, _ = _category(ctx)
    if m is None or cat is None:
        return None
    g = pd.DataFrame({"c": ctx.df[cat].astype("string"), "v": m}).dropna().groupby("c")["v"].sum()
    if len(g) < 2 or g.sum() <= 0:
        return None
    g = g.sort_values(ascending=False)
    shown = top_n_with_other(g, ctx.config.top_n)
    t = pd.DataFrame({"label": shown.index.astype(str), "value": 100 * shown.to_numpy() / g.sum()})
    cap = f"{g.index[0]} is the largest {cat}, with {100 * g.iloc[0] / g.sum():.1f}% of {name} across {len(g):,} values."
    return SuggestedView("category_share", f"Share of {name} by {cat}", "bar", t, "label", ("value",), cap, (name, cat))


SUPPLY_VIEWS = (_on_time_by_group, _lead_time_distribution, _stock_vs_reorder, _defects, _supplier_share)
GENERAL_VIEWS = (_trend, _category_share)


def suggest_views(df: pd.DataFrame, model: SemanticModel, config: AnalyticsConfig = DEFAULT_ANALYTICS,
                  data_filter: DataFilter | None = None, max_views: int = 4) -> tuple[SuggestedView, ...]:
    """Up to `max_views` views chosen from the roles present. Supply-chain views first, then general ones."""
    flt = data_filter or DataFilter()
    d = flt.apply(df)
    if len(d) == 0:
        return ()
    ctx = _Ctx(d, model, config, flt)
    out: list[SuggestedView] = []
    for fn in SUPPLY_VIEWS + GENERAL_VIEWS:
        if len(out) >= max_views:
            break
        try:
            v = fn(ctx)
        except Exception:
            log.exception("Suggested view %s failed", fn.__name__)
            continue
        if v is not None:
            out.append(v)
    return tuple(out)
