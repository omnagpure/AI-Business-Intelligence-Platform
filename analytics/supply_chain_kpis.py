"""Supply-chain KPIs: delivery, quantity, stock, quality and supplier concentration.

Same rules as analytics/kpis.py: a KPI exists only when the columns it needs exist, every KPI states its
formula, columns and limitations, zero denominators are "not calculable", and nothing is invented.
Delivery columns are found from role plus name tokens (expected/promised vs received/delivered).
Data problems (delivered before ordered, received above ordered, defects above units, flag columns that
are not real flags) are reported in limitations as "Data check: ..."; the data itself is never changed.
"""
from __future__ import annotations

import pandas as pd

from analytics.data_checks import parse_flag
from analytics.descriptive import top_n_with_other
from analytics.kpis import (KPIResult, _Ctx, _growth, _order_id, _revenue, _stock, _units, primary_date_column,
                            STOCK_TOKENS, REORDER_TOKENS, TOTAL_TOKENS, UNIT_COST_TOKENS)

SC = "Supply chain"
PROC = "Procurement"
EXPECTED_DATE = frozenset({"expected", "promised", "scheduled", "planned", "due", "eta", "committed", "commit"})
ACTUAL_DATE = frozenset({"received", "delivered", "arrival", "arrived", "actual"})
ORDERED_DATE = frozenset({"ordered", "order", "placed", "created"})
ACTUAL_DAYS = frozenset({"real", "actual"})
SCHEDULED_DAYS = frozenset({"scheduled", "planned", "promised", "expected"})
LATE_WORDS = frozenset({"late", "delayed", "overdue"})
QTY_WORDS = frozenset({"quantity", "qty", "units", "unit", "count"})
RECEIVED_QTY = frozenset({"received", "delivered", "fulfilled"})
SOLD_WORDS = frozenset({"sold", "sales", "consumption", "usage", "demand"})
DEFECT_WORDS = frozenset({"defect", "scrap", "reject", "rework"})
CANCELLED = frozenset({"cancelled", "canceled", "cancel", "void", "voided"})
OPEN = frozenset({"open", "pending", "in transit", "in_transit", "backorder", "ordered"})
DAY = pd.Timedelta(days=1)


# ---- column finders ---------------------------------------------------------------------------
def _cols(ctx: _Ctx, roles: tuple[str, ...], pred, numeric: bool) -> list[str]:
    out = []
    for r in roles:
        out += [c for c in ctx.rank(r, numeric=numeric, name_filter=pred) if pred(ctx.tokens(c)) and c not in out]
    return out


def _is_on_time_name(t: set[str]) -> bool:
    return ({"on", "time"} <= t) or bool(t & {"ontime", "otd"})


def _status(ctx: _Ctx) -> tuple[pd.Series | None, pd.Series | None, str | None]:
    """(cancelled mask, open mask, column) from a status column; (None, None, None) when absent."""
    for col in ctx.rank("category", numeric=False):
        if "status" not in ctx.tokens(col):
            continue
        v = ctx.df[col].astype("string").str.strip().str.lower()
        if set(v.dropna().unique()) & (CANCELLED | OPEN):
            return v.isin(CANCELLED).fillna(False), v.isin(OPEN).fillna(False), col
    return None, None, None


def is_procurement(ctx: _Ctx) -> bool:
    """Supplier data with an amount column and no customers: the amount is spend, not revenue."""
    return bool(ctx.rank("supplier", numeric=False)) and bool(ctx.rank("revenue")) \
        and not ctx.model.columns_with_role("customer")


# ---- on-time outcome (shared by on-time rate, delay and OTIF) ---------------------------------
def _on_time(ctx: _Ctx):
    """Returns (on_time bool series with NA = unknown, delay in days or None, columns, formula, notes, reason)."""
    notes: list[str] = []
    dates = ctx.rank("datetime", numeric=False)
    exp = [c for c in dates if ctx.tokens(c) & EXPECTED_DATE]
    act = [c for c in dates if ctx.tokens(c) & ACTUAL_DATE and c not in exp]
    if exp and act:
        e, a = exp[0], act[0]
        delay = (ctx.dt(a) - ctx.dt(e)).dt.total_seconds() / 86400
        on = pd.Series(pd.NA, index=ctx.df.index, dtype="boolean")
        known = delay.notna()
        on[known] = delay[known] <= 0
        miss = int((~known).sum())
        if miss:
            notes.append(f"{miss:,} rows have no '{a}' or '{e}' value (for example orders not yet received); "
                         "their outcome is unknown and they are excluded.")
        return on, delay, [e, a], f"on time when {a} <= {e}", notes + ctx.col_notes(e) + ctx.col_notes(a), ""
    nums = _cols(ctx, ("duration", "quantity"), lambda t: True, True)
    ad = [c for c in nums if ctx.tokens(c) & ACTUAL_DAYS and ctx.tokens(c) & {"days", "day", "shipping", "shipment", "delivery"}]
    sd = [c for c in nums if ctx.tokens(c) & SCHEDULED_DAYS and ctx.tokens(c) & {"days", "day", "shipping", "shipment", "delivery"}]
    if ad and sd:
        delay = ctx.num(ad[0]) - ctx.num(sd[0])
        on = pd.Series(pd.NA, index=ctx.df.index, dtype="boolean")
        on[delay.notna()] = delay[delay.notna()] <= 0
        return on, delay, [ad[0], sd[0]], f"on time when {ad[0]} <= {sd[0]}", ctx.col_notes(ad[0]), ""
    bad = ""
    for name, c in ctx.model.columns.items():
        t = ctx.tokens(name)
        if name not in ctx.df.columns or not (_is_on_time_name(t) or t & LATE_WORDS):
            continue
        flags, why = parse_flag(ctx.df[name])
        if flags is None:
            bad = bad or f"Column '{name}' looks like a delivery flag but {why}, so it was not used."
            continue
        late_style = bool(t & LATE_WORDS) and not _is_on_time_name(t)
        on = (~flags) if late_style else flags
        word = "late" if late_style else "on time"
        return on, None, [name], f"share of rows where {name} says {word}" if not late_style else \
            f"share of rows where {name} does not say late", notes, ""
    reason = "No expected/promised and received/delivered date pair, no actual vs scheduled days, and no valid on-time or late flag was found."
    return None, None, [], "", [], (bad + " " + reason).strip()


def _k_on_time(ctx: _Ctx) -> list[KPIResult]:
    on, delay, cols, how, notes, reason = _on_time(ctx)
    name = "On-time delivery rate"
    if on is None:
        return [ctx.skipped("on_time_rate", name, SC, "percent", reason),
                ctx.skipped("avg_delay", "Average delay", SC, "number", reason)]
    known = on.dropna()
    formula = f"on-time rows / rows with a known outcome x 100 ({how})"
    out = []
    if len(known) == 0:
        out.append(ctx.not_calculable("on_time_rate", name, SC, "percent", formula, cols, "no row has a known delivery outcome"))
    else:
        out.append(ctx.result("on_time_rate", name, SC, 100 * float(known.astype(int).mean()), "percent", formula, cols,
                              notes + [f"Based on {len(known):,} rows with a known outcome."]))
    if delay is None or delay.notna().sum() == 0:
        out.append(ctx.skipped("avg_delay", "Average delay", SC, "number",
                               "Needs a date or day-count pair (actual vs expected); a flag alone gives no day counts."))
    else:
        out.append(ctx.result("avg_delay", "Average delay (days vs expected)", SC, float(delay.mean()), "number",
                              f"AVG(actual - expected), {how}", cols,
                              notes + ["Negative values mean early; early and late rows are netted, so read it with the on-time rate.",
                                       f"Based on {int(delay.notna().sum()):,} rows."]))
    return out


# ---- lead time --------------------------------------------------------------------------------
def lead_days(ctx: _Ctx):
    """Lead time in days per row. Returns (series, columns, notes) or (None, [], [reason]).
    Uses a lead-time column, else order date to received/delivered date (negative gaps are excluded and reported)."""
    direct = [c for c in _cols(ctx, ("duration", "quantity", "unknown"), lambda t: "lead" in t and "time" in t, True)]
    if direct:
        x = ctx.num(direct[0])
        return x.dropna(), [direct[0]], ctx.col_notes(direct[0]) + [f"Based on {int(x.notna().sum()):,} rows."]
    dates = ctx.rank("datetime", numeric=False)
    od = [c for c in dates if ctx.tokens(c) & ORDERED_DATE and not ctx.tokens(c) & (EXPECTED_DATE | ACTUAL_DATE)]
    ad = [c for c in dates if ctx.tokens(c) & ACTUAL_DATE and not ctx.tokens(c) & EXPECTED_DATE]
    if od and ad:
        d = (ctx.dt(ad[0]) - ctx.dt(od[0])).dt.total_seconds() / 86400
        neg = int((d < 0).sum())
        d = d[d >= 0]
        notes = ctx.col_notes(od[0]) + ctx.col_notes(ad[0]) + [f"Based on {len(d):,} rows with both dates."]
        if neg:
            notes.append(f"Data check: {neg:,} rows have '{ad[0]}' before '{od[0]}'; they were excluded. Review them in the source.")
        return d, [od[0], ad[0]], notes
    return None, [], ["Needs a lead-time column, or an order date and a received/delivered date."]


def _k_lead_time(ctx: _Ctx) -> list[KPIResult]:
    key, name = "avg_lead_time", "Average lead time (days)"
    d, cols, notes = lead_days(ctx)
    if d is None:
        return [ctx.skipped(key, name, SC, "number", notes[0])]
    formula = f"AVG({cols[0]})" if len(cols) == 1 else f"AVG({cols[1]} - {cols[0]}) in days"
    if d.empty:
        return [ctx.not_calculable(key, name, SC, "number", formula, cols, "no row has both dates in a valid order", notes)]
    return [ctx.result(key, name, SC, d.mean(), "number", formula, cols, notes)]


# ---- quantities: fill rate and OTIF -----------------------------------------------------------
def _qty_pair(ctx: _Ctx):
    nums = _cols(ctx, ("quantity",), lambda t: bool(t & QTY_WORDS), True)
    ordered = [c for c in nums if ctx.tokens(c) & {"ordered", "order"} and not ctx.tokens(c) & RECEIVED_QTY]
    received = [c for c in nums if ctx.tokens(c) & RECEIVED_QTY]
    return (ordered[0], received[0]) if ordered and received else (None, None)


def _closed_rows(ctx: _Ctx, ordered: str, received: str):
    """Rows with both quantities, excluding cancelled and still-open orders. Returns (mask, notes)."""
    mask = ctx.num(ordered).notna() & ctx.num(received).notna()
    cancelled, opened, col = _status(ctx)
    notes = []
    if col is not None:
        excl = int((mask & (cancelled | opened)).sum())
        mask &= ~(cancelled | opened)
        if excl:
            notes.append(f"{excl:,} rows with a cancelled or open status in '{col}' were excluded (nothing received yet).")
    return mask, notes


def _k_fill_rate(ctx: _Ctx) -> list[KPIResult]:
    ordered, received = _qty_pair(ctx)
    name = "Fill rate"
    if ordered is None:
        return [ctx.skipped("fill_rate", name, SC, "percent", "Needs an ordered-quantity column and a received-quantity column.")]
    mask, notes = _closed_rows(ctx, ordered, received)
    o, r = ctx.num(ordered)[mask], ctx.num(received)[mask]
    formula = f"SUM(MIN({received}, {ordered})) / SUM({ordered}) x 100"
    if o.sum() <= 0:
        return [ctx.not_calculable("fill_rate", name, SC, "percent", formula, [ordered, received], "total ordered quantity is zero")]
    over = int((r > o).sum())
    if over:
        notes.append(f"Data check: {over:,} rows received more than ordered; over-delivery is capped per line so it cannot hide shortfalls elsewhere.")
    short = int((r < o).sum())
    return [ctx.result("fill_rate", name, SC, 100 * float(pd.concat([r, o], axis=1).min(axis=1).sum() / o.sum()), "percent",
                       formula, [ordered, received],
                       notes + [f"{short:,} of {len(o):,} lines were short-shipped."] + ctx.col_notes(ordered) + ctx.col_notes(received))]


def _k_otif(ctx: _Ctx) -> list[KPIResult]:
    name = "On-time in-full (OTIF)"
    ordered, received = _qty_pair(ctx)
    on, _, ocols, how, onotes, _ = _on_time(ctx)
    if ordered is None or on is None:
        return [ctx.skipped("otif", name, SC, "percent", "Needs both an on-time outcome (dates or a valid flag) and ordered and received quantities.")]
    mask, notes = _closed_rows(ctx, ordered, received)
    mask &= on.notna()
    n = int(mask.sum())
    formula = f"rows on time AND {received} >= {ordered}, divided by rows with both known x 100"
    cols = ocols + [ordered, received]
    if n == 0:
        return [ctx.not_calculable("otif", name, SC, "percent", formula, cols, "no row has a known outcome and quantities")]
    ok = on[mask].astype(bool) & (ctx.num(received)[mask] >= ctx.num(ordered)[mask])
    oid = _order_id(ctx)
    extra = [f"Each row is treated as one order; '{oid}' repeats, so rows may be order lines."] \
        if oid and ctx.df[oid].duplicated().any() else []
    return [ctx.result("otif", name, SC, 100 * float(ok.mean()), "percent", formula, cols,
                       notes + extra + onotes + [f"Based on {n:,} rows."])]


# ---- stock ------------------------------------------------------------------------------------
def _k_stockout(ctx: _Ctx) -> list[KPIResult]:
    name = "Stockout rate"
    cands = [(n, ctx.tokens(n)) for n in ctx.model.columns if n in ctx.df.columns]
    flag = [n for n, t in cands if "stockout" in t or "oos" in t or {"stock", "out"} <= t]
    bad = ""
    for col in flag:
        f, why = parse_flag(ctx.df[col])
        if f is None:
            bad = bad or f"Column '{col}' looks like a stockout flag but {why}."
            continue
        k = f.dropna()
        lim = ctx.col_notes(col) + [f"Based on {len(k):,} rows."]
        if k.sum() == 0:
            lim.append("No stockout is recorded in this column (every value is 0/No); check that the flag is populated in the source.")
        return [ctx.result("stockout_rate", name, SC, 100 * float(k.astype(int).mean()), "percent",
                           f"rows where {col} is Yes / rows x 100", [col], lim)]
    stock = _stock(ctx)
    if stock is None:
        return [ctx.skipped("stockout_rate", name, SC, "percent", (bad + " No stockout flag or stock-on-hand column was found.").strip())]
    s = ctx.num(stock).dropna()
    neg = int((s < 0).sum())
    lim = ctx.col_notes(stock) + [f"Derived from stock <= 0 in '{stock}' because no stockout flag exists."]
    if neg:
        lim.append(f"Data check: {neg:,} rows have negative stock in '{stock}'; they count as stockouts.")
    return [ctx.result("stockout_rate", name, SC, 100 * float((s <= 0).mean()), "percent", f"rows where {stock} <= 0 / rows x 100", [stock], lim)]


def _reorder(ctx: _Ctx) -> str | None:
    r = [c for c in ctx.rank("quantity") if ctx.tokens(c) & REORDER_TOKENS]
    return r[0] if r else None


def _k_below_reorder(ctx: _Ctx) -> list[KPIResult]:
    stock, rp = _stock(ctx), _reorder(ctx)
    name = "Share of stock rows below reorder point"
    if stock is None or rp is None:
        return [ctx.skipped("below_reorder_share", name, SC, "percent", "Needs a stock-on-hand column and a reorder-point column.")]
    both = ctx.num(stock).notna() & ctx.num(rp).notna()
    if not both.any():
        return [ctx.not_calculable("below_reorder_share", name, SC, "percent", f"{stock} < {rp}", [stock, rp], "no row has both values")]
    below = ctx.num(stock)[both] < ctx.num(rp)[both]
    return [ctx.result("below_reorder_share", name, SC, 100 * float(below.mean()), "percent",
                       f"rows where {stock} < {rp} / rows x 100 (all dates)", [stock, rp],
                       [f"{int(below.sum()):,} of {int(both.sum()):,} stock positions across all dates; "
                        "the separate 'Rows below reorder point' KPI uses the latest snapshot only."])]


def _sold(ctx: _Ctx) -> str | None:
    c = [c for c in _cols(ctx, ("quantity",), lambda t: bool(t & SOLD_WORDS) and "forecast" not in t, True)]
    return c[0] if c else None


def _daily(ctx: _Ctx, col: str, date: str, values: pd.Series | None = None) -> pd.Series:
    v = ctx.num(col) if values is None else values
    return v.groupby(ctx.dt(date).dt.normalize()).sum(min_count=1).dropna()


def _k_cover_and_turnover(ctx: _Ctx) -> list[KPIResult]:
    stock, sold, date = _stock(ctx), _sold(ctx), primary_date_column(ctx.df, ctx.model)
    why = ("Needs a stock-on-hand column, a units-sold column and a date column."
           if None in (stock, sold, date) else "")
    keys = (("days_of_cover", "Days of cover"), ("inventory_turnover", "Inventory turnover (per year)"))
    if why:
        return [ctx.skipped(k, n, SC, "number", why) for k, n in keys]
    st_d, sold_d = _daily(ctx, stock, date), _daily(ctx, sold, date)
    dates = st_d.index.intersection(sold_d.index)
    if len(dates) < 2:
        return [ctx.not_calculable(k, n, SC, "number", "", [stock, sold, date], "needs at least two dates of stock and sales")
                for k, n in keys]
    st_d, sold_d = st_d[dates], sold_d[dates]
    per_date = ctx.dt(date).dt.normalize().value_counts()
    notes = [f"Average of daily totals over {len(dates):,} dates."]
    if per_date.min() != per_date.max():
        notes.append("Data check: dates list different numbers of rows, so some items may be missing on some dates and the average stock is understated.")
    out = []
    daily_sales = float(sold_d.mean())
    if daily_sales <= 0:
        out.append(ctx.not_calculable("days_of_cover", keys[0][1], SC, "number", "AVG(daily stock) / AVG(daily units sold)",
                                      [stock, sold, date], "no units were sold"))
    else:
        out.append(ctx.result("days_of_cover", keys[0][1], SC, float(st_d.mean()) / daily_sales, "number",
                              f"AVG(daily total {stock}) / AVG(daily total {sold})", [stock, sold, date], notes))
    costs = [c for c in ctx.rank("cost") if not (ctx.tokens(c) & TOTAL_TOKENS)]
    unit_costs = [c for c in costs if ctx.tokens(c) & UNIT_COST_TOKENS] or costs
    if not unit_costs:
        out.append(ctx.skipped("inventory_turnover", keys[1][1], SC, "number", "Needs a unit cost column to value the stock and the cost of goods sold."))
        return out
    cc = unit_costs[0]
    cogs_d = _daily(ctx, cc, date, ctx.num(sold) * ctx.num(cc))
    inv_d = _daily(ctx, cc, date, ctx.num(stock) * ctx.num(cc))
    cogs, avg_inv = float(cogs_d[dates].sum()), float(inv_d[dates].mean())
    days = (dates.max() - dates.min()).days + 1
    if avg_inv <= 0:
        out.append(ctx.not_calculable("inventory_turnover", keys[1][1], SC, "number",
                                      f"(SUM({sold} x {cc}) / AVG(daily stock value)) x 365 / days", [stock, sold, cc, date],
                                      "average inventory value is zero"))
    else:
        out.append(ctx.result("inventory_turnover", keys[1][1], SC, cogs / avg_inv * 365 / days, "number",
                              f"(SUM({sold} x {cc}) / AVG(daily SUM({stock} x {cc}))) x 365 / {days} days",
                              [stock, sold, cc, date],
                              notes + [f"'{cc}' is assumed to be cost per unit. Period covers {days:,} days; the result is annualised."]))
    return out


# ---- quality and suppliers --------------------------------------------------------------------
def _k_defect_rate(ctx: _Ctx) -> list[KPIResult]:
    name = "Defect rate"
    d = _cols(ctx, ("quantity",), lambda t: bool(t & DEFECT_WORDS), True)
    u, _ = _units(ctx)
    if not d or u is None:
        return [ctx.skipped("defect_rate", name, SC, "percent", "Needs a defects column and a units-produced column.")]
    both = ctx.num(d[0]).notna() & ctx.num(u).notna()
    de, un = ctx.num(d[0])[both], ctx.num(u)[both]
    formula = f"SUM({d[0]}) / SUM({u}) x 100"
    if un.sum() <= 0:
        return [ctx.not_calculable("defect_rate", name, SC, "percent", formula, [d[0], u], "total units is zero")]
    rate = 100 * float(de.sum() / un.sum())
    over = int((de > un).sum())
    notes = ctx.col_notes(d[0]) + ctx.col_notes(u) + [f"Based on {len(de):,} rows."]
    if rate > 100:
        return [ctx.not_calculable("defect_rate", name, SC, "percent", formula, [d[0], u],
                                   f"defects ({de.sum():,.0f}) exceed units ({un.sum():,.0f}); a rate above 100% is not meaningful",
                                   notes)]
    if over:
        notes.append(f"Data check: {over:,} of {len(de):,} rows have more defects than units produced. If defects are counted per unit this is "
                     "impossible; treat the rate as unreliable until the source is checked.")
    return [ctx.result("defect_rate", name, SC, rate, "percent", formula, [d[0], u], notes)]


def _supplier_measure(ctx: _Ctx, keep: pd.Series):
    """(series, label, column) used to weigh suppliers: spend, revenue, ordered quantity, units, else row count."""
    rev, _ = _revenue(ctx)
    if rev is not None:
        return ctx.num(rev)[keep], ("spend" if is_procurement(ctx) else "value"), rev
    ordered, _ = _qty_pair(ctx)
    for c in ([ordered] if ordered else []) + _cols(ctx, ("quantity",), lambda t: bool(t & {"ordered", "order"}), True):
        return ctx.num(c)[keep], "ordered quantity", c
    u, _ = _units(ctx)
    if u is not None:
        return ctx.num(u)[keep], "units", u
    return pd.Series(1.0, index=ctx.df.index)[keep], "rows", None


def supplier_shares(ctx: _Ctx):
    """(totals per supplier sorted high to low, measure label, columns, notes) or (None, reason, [], [])."""
    sup = ctx.rank("supplier", numeric=False)
    if not sup:
        return None, "No column has the Supplier role.", [], []
    cancelled, _, _ = _status(ctx)
    keep = ~cancelled if cancelled is not None else pd.Series(True, index=ctx.df.index)
    m, label, mcol = _supplier_measure(ctx, keep)
    g = pd.DataFrame({"s": ctx.df[sup[0]][keep].astype("string"), "v": m}).dropna().groupby("s")["v"].sum()
    cols = [sup[0]] + ([mcol] if mcol else [])
    if len(g) < 2:
        return None, "Fewer than two distinct suppliers.", cols, []
    if g.sum() <= 0:
        return None, f"total {label} is zero or negative", cols, []
    notes = ctx.col_notes(sup[0]) + [f"Weighted by {label}; {len(g):,} suppliers. Largest: {g.sort_values(ascending=False).index[0]}."]
    if cancelled is not None and int(cancelled.sum()):
        notes.append(f"{int(cancelled.sum()):,} cancelled rows were excluded.")
    return g.sort_values(ascending=False), label, cols, notes


def _k_supplier_concentration(ctx: _Ctx) -> list[KPIResult]:
    keys = (("supplier_top_share", "Top supplier share"), ("supplier_top3_share", "Top 3 suppliers share"))
    g, label, cols, notes = supplier_shares(ctx)
    if g is None:
        sup = ctx.rank("supplier", numeric=False)
        if not sup or label.startswith("Fewer") or label == "No column has the Supplier role.":
            return [ctx.skipped(k, n, SC, "percent", label) for k, n in keys]
        return [ctx.not_calculable(k, n, SC, "percent", "", cols, label) for k, n in keys]
    formula = f"SUM({cols[1] if len(cols) > 1 else 'rows'}) of the largest supplier(s) by {cols[0]} / total x 100"
    shown = top_n_with_other(g, ctx.config.top_n)
    bd = tuple((str(k), float(v)) for k, v in shown.items())
    return [ctx.result(keys[0][0], keys[0][1], SC, 100 * g.iloc[0] / g.sum(), "percent", formula, cols, notes, breakdown=bd),
            ctx.result(keys[1][0], keys[1][1], SC, 100 * g.iloc[:3].sum() / g.sum(), "percent", formula, cols, notes, breakdown=bd)]


# ---- procurement spend (amount columns in supplier data are spend, not revenue) ---------------
def _k_spend(ctx: _Ctx) -> list[KPIResult]:
    if not is_procurement(ctx):
        return []
    col, alts = _revenue(ctx)
    cancelled, _, scol = _status(ctx)
    keep = ~cancelled if cancelled is not None else pd.Series(True, index=ctx.df.index)
    x = ctx.num(col)[keep]
    notes = [f"'{col}' is treated as purchase spend because the data has suppliers and no customers; it is not sales revenue."] \
        + ctx.col_notes(col, alts, money=True)
    if cancelled is not None:
        c = int(cancelled.sum())
        notes.append(f"{c:,} cancelled rows ('{scol}') worth {ctx.num(col)[cancelled].sum():,.0f} were excluded.")
    out = [ctx.result("total_spend", "Purchase spend", PROC, x.sum(min_count=1), "amount", f"SUM({col}) excluding cancelled orders", [col], notes)]
    sub = _Ctx(ctx.df[keep], ctx.model, ctx.config, ctx.flt)
    g = _growth(sub, "spend_growth", "Spend growth", col, PROC)
    out += g
    oid = _order_id(ctx)
    orders = ctx.df[oid][keep].nunique(dropna=True) if oid else int(x.notna().sum())
    if orders:
        out.append(ctx.result("avg_po_value", "Average order value (spend)", PROC, float(x.sum() / orders), "amount",
                              f"SUM({col}) / COUNT DISTINCT({oid})" if oid else f"SUM({col}) / rows", [col] + ([oid] if oid else []),
                              ["Cancelled orders are excluded."] if cancelled is not None else []))
    return out


SUPPLY_CHAIN_BUILDERS = (_k_spend, _k_on_time, _k_lead_time, _k_fill_rate, _k_otif, _k_stockout,
                         _k_below_reorder, _k_cover_and_turnover, _k_defect_rate, _k_supplier_concentration)
PROCUREMENT_REPLACED = frozenset({"total_revenue", "revenue_growth", "aov", "avg_transaction", "revenue_by_category"})
