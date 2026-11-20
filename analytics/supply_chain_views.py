"""Suggested dashboard views. Pure pandas, no Streamlit or Plotly.

Each view is chosen from the roles present, carries the small table it plots, and a one-line caption computed from
that table (nothing is typed in). A view is returned only when its data exists; otherwise it is simply absent.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analytics.descriptive import aggregate_time, top_n_with_other
from analytics.kpis import DataFilter, _Ctx, _category, _revenue, _stock, _units, primary_date_column
from analytics.supply_chain_kpis import (DEFECT_WORDS, SC, _cols, _on_time, _reorder, _status, lead_days, lead_label,
                                          supplier_shares)
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.logging_config import get_logger
from semantic.semantic_model import SemanticModel

log = get_logger("analytics.views")
MIN_GROUP_ROWS = 10
MIN_GROUP_ROWS_SMALL = 5
MAX_SUPPLIER_GROUPS = 1000      # a supplier column may have many values; only the lowest 10 are drawn
GROUP_HINT = frozenset({"supplier", "vendor", "mode", "shipping", "shipment", "carrier", "method", "warehouse"})
PLACE_HINT = frozenset({"market", "region", "country", "zone", "territory", "state"})
MAX_PLACE_GROUPS = 12
SMALL_SPREAD_PTS = 2.0      # best minus worst group below this: the grouping does not matter


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
    ref: float | None = None       # reference value drawn on the chart (for example the overall rate)
    unit: str = ""                 # "%" or "days" for value labels


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


def _on_time_rates(ctx: _Ctx, g: str):
    """(rates per group with at least `floor` rows sorted lowest first, overall rate, floor, columns) or None."""
    on, _, cols, _, _, _ = _on_time(ctx)
    if on is None:
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
    return kept.sort_values("rate"), 100 * f["on"].mean(), floor, cols


def _on_time_view(ctx: _Ctx, g: str, key: str) -> SuggestedView | None:
    res = _on_time_rates(ctx, g)
    if res is None:
        return None
    low, overall, floor, cols = res
    shown = low.head(10).reset_index().rename(columns={"g": "label", "rate": "value"})
    worst, best = low.iloc[0], low.iloc[-1]
    more = f" The {len(low) - 10} other groups are not drawn." if len(low) > 10 else ""
    if best["rate"] - worst["rate"] < SMALL_SPREAD_PTS:
        cap = (f"On-time rates hardly differ by {g}: from {worst['rate']:.1f}% ({low.index[0]}) to {best['rate']:.1f}% "
               f"({low.index[-1]}) against {overall:.1f}% overall, so {g} does not explain late deliveries. "
               f"Groups with at least {floor} rows.{more}")
    else:
        cap = (f"{low.index[0]} has the lowest on-time rate ({worst['rate']:.1f}% of {int(worst['rows']):,} rows) against "
               f"{overall:.1f}% overall; the best is {low.index[-1]} ({best['rate']:.1f}%). Groups with at least {floor} rows.{more}")
    title = f"On-time rate by {g}" if len(low) <= 10 else f"Lowest on-time rates by {g}"
    return SuggestedView(key, title, "bar", shown[["label", "value", "rows"]], "label", ("value",), cap,
                         tuple(cols + [g]), ref=overall, unit="%")


def _on_time_by_group(ctx: _Ctx) -> SuggestedView | None:
    g = _group_col(ctx)
    return None if g is None else _on_time_view(ctx, g, "on_time_by_group")


def _place_col(ctx: _Ctx, exclude: str | None) -> str | None:
    """A location column with a handful of values (market, region) to compare on-time rates across places."""
    cands = []
    for c in ctx.rank("location", numeric=False):
        n = ctx.df[c].nunique(dropna=True)
        if c != exclude and 3 <= n <= MAX_PLACE_GROUPS and len(ctx.df) / n >= MIN_GROUP_ROWS:
            cands.append((-bool(ctx.tokens(c) & PLACE_HINT), n, c))
    return sorted(cands)[0][2] if cands else None


def _on_time_by_place(ctx: _Ctx) -> SuggestedView | None:
    g = _place_col(ctx, _group_col(ctx))
    return None if g is None else _on_time_view(ctx, g, "on_time_by_place")


def _lead_time_distribution(ctx: _Ctx) -> SuggestedView | None:
    d, cols, _ = lead_days(ctx)
    if d is None or len(d) < 5:
        return None
    p50, p90 = float(d.median()), float(d.quantile(0.9))
    what = lead_label(cols)
    verb = "ship" if what == "shipping time" else "arrive"
    cap = f"Median {what} is {p50:.1f} days and 90% of rows {verb} within {p90:.1f} days ({len(d):,} rows)."
    return SuggestedView("lead_time", f"{what.capitalize()} distribution (days)", "histogram",
                         pd.DataFrame({"days": d.to_numpy()}), "days", ("days",), cap, tuple(cols), unit="days")


def _stock_vs_reorder(ctx: _Ctx) -> SuggestedView | None:
    """Share of stock records (item x location on a date) below their own reorder point, per date.
    Comparing summed stock with summed reorder points would hide shortages: one item far above its reorder point
    cancels out another below it."""
    stock, rp, date = _stock(ctx), _reorder(ctx), primary_date_column(ctx.df, ctx.model)
    if None in (stock, rp, date):
        return None
    f = pd.DataFrame({"date": ctx.dt(date).dt.normalize(), "s": ctx.num(stock), "r": ctx.num(rp)}).dropna()
    if f.empty:
        return None
    f["below"] = f["s"] < f["r"]
    per = f.groupby("date")["below"].agg(share=lambda b: 100 * b.mean(), below="sum", items="size")
    if len(per) < 2:
        return None
    span = (per.index.max() - per.index.min()).days
    if span >= 120:                                   # daily points are noisy over a long period: weekly averages
        plot = per["share"].resample("W").mean().dropna().rename_axis("date").reset_index()
        grain = "weekly average"
    else:
        plot = per["share"].rename_axis("date").reset_index()
        grain = "daily"
    last = per.iloc[-1]
    cap = (f"On the latest date ({per.index[-1]:%Y-%m-%d}), {int(last['below']):,} of {int(last['items']):,} items "
           f"({last['share']:.1f}%) were below their reorder point; {per['share'].mean():.1f}% on an average date, "
           f"highest {per['share'].max():.1f}% on {per['share'].idxmax():%Y-%m-%d}. Chart shows the {grain}.")
    return SuggestedView("stock_vs_reorder", "Items below reorder point", "line", plot, "date", ("share",), cap,
                         (stock, rp, date), ref=float(per["share"].mean()), unit="%")


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
    return SuggestedView("supplier_share", f"Share of {label} by supplier", "bar", t, "label", ("value",), cap, tuple(cols),
                         unit="%")


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
    return SuggestedView("category_share", f"Share of {name} by {cat}", "bar", t, "label", ("value",), cap, (name, cat),
                         unit="%")


SUPPLY_VIEWS = (_on_time_by_group, _on_time_by_place, _lead_time_distribution, _stock_vs_reorder, _defects, _supplier_share)
GENERAL_VIEWS = (_trend, _category_share)


def suggest_views(df: pd.DataFrame, model: SemanticModel, config: AnalyticsConfig = DEFAULT_ANALYTICS,
                  data_filter: DataFilter | None = None, max_views: int = 4,
                  skip: tuple[str, ...] = ()) -> tuple[SuggestedView, ...]:
    """Up to `max_views` views chosen from the roles present. Supply-chain views first, then general ones.
    `skip` names view keys the caller already shows another way (the dashboard has its own trend chart)."""
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
        if v is not None and v.key not in skip:
            out.append(v)
    return tuple(out)
