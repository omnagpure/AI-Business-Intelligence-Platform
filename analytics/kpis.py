"""Role-driven KPI engine. Pure pandas, no Streamlit; input data is never modified.

A KPI is produced only when the semantic roles it needs exist. Roles come from the SemanticModel, so a
user's override on the Semantics page changes the KPIs. Every KPI records its formula, columns,
filters, limitations and calculation time. Ratios use totals (margin = SUM(profit) / SUM(revenue)),
zero denominators give value=None with a "Not calculable" limitation, and growth compares only two
complete periods.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analytics.descriptive import (EXCLUDED_NUMERIC_ROLES, FREQ_LABELS, aggregate_time, datetime_series,
                                   numeric_series, top_n_with_other)
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.logging_config import get_logger
from semantic.column_lexicon import token_set
from semantic.semantic_model import SemanticModel

log = get_logger("analytics.kpis")

OK, NOT_CALCULABLE, SKIPPED = "ok", "not_calculable", "skipped"
LOW_CONFIDENCE = 0.5

PREFER_DATE = frozenset({"order", "ordered", "transaction", "invoice", "sale", "sales", "purchase", "date"})
AVOID_DATE = frozenset({"ship", "shipping", "shipment", "delivery", "delivered", "received", "expected", "due"})
PREFER_UNITS = frozenset({"sold", "ordered", "units", "purchased", "produced", "qty", "quantity"})
AVOID_UNITS = frozenset({"inventory", "stock", "level", "reorder", "point", "forecast", "received",
                         "defects", "defect", "line", "count", "safety"})
STOCK_TOKENS = frozenset({"inventory", "stock", "onhand", "hand", "level"})
REORDER_TOKENS = frozenset({"reorder"})
UNIT_COST_TOKENS = frozenset({"unit", "each"})
TOTAL_TOKENS = frozenset({"total", "extended"})
ORDER_TOKENS = frozenset({"order", "invoice", "transaction", "purchase", "po"})
NOT_ORDER_TOKENS = frozenset({"item", "line", "product", "category", "customer", "department", "card",
                              "supplier", "warehouse", "sku"})
CUSTOMER_ID_TOKENS = frozenset({"id", "number", "no", "code", "key", "ref"})
CUSTOMER_NAME_TOKENS = frozenset({"fname", "lname", "first", "last", "street", "city", "name"})
CATEGORY_PREFER = ("category", "segment", "department", "class", "group", "type")


# ---- result types -----------------------------------------------------------------------------
@dataclass(frozen=True)
class KPIResult:
    key: str
    name: str
    group: str                       # Sales | Profit and cost | Customers | Inventory | General
    status: str                      # ok | not_calculable | skipped
    value: float | None
    unit: str                        # amount | count | percent | number
    formula: str
    columns: tuple[str, ...]
    filters: tuple[str, ...]
    limitations: tuple[str, ...]
    calculated_at: str
    reason: str = ""                 # why not calculable / skipped
    period: str = ""                 # e.g. "2017-12 vs 2017-11"
    breakdown: tuple[tuple[str, float], ...] = ()


KPIMetric = KPIResult                # alias used in the project notes


@dataclass(frozen=True)
class KPIReport:
    results: tuple[KPIResult, ...]
    rows_used: int
    rows_total: int
    domain: str
    notes: tuple[str, ...] = ()

    def ok(self) -> list[KPIResult]:
        return [r for r in self.results if r.status == OK]

    def unavailable(self) -> list[KPIResult]:
        return [r for r in self.results if r.status != OK]

    def get(self, key: str) -> KPIResult | None:
        return next((r for r in self.results if r.key == key), None)


@dataclass(frozen=True)
class DataFilter:
    """Row filter applied before any analysis. Hashable, with a stable key for cache invalidation."""
    categories: tuple[tuple[str, tuple[str, ...]], ...] = ()
    date_col: str | None = None
    date_start: str | None = None        # ISO date, inclusive
    date_end: str | None = None          # ISO date, inclusive

    @property
    def active(self) -> bool:
        return bool(self.categories) or (self.date_col is not None and (self.date_start or self.date_end))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.active:
            return df
        mask = pd.Series(True, index=df.index)
        for col, values in self.categories:
            if col in df.columns:
                mask &= df[col].astype("string").isin(list(values)).fillna(False)
        if self.date_col and self.date_col in df.columns and (self.date_start or self.date_end):
            d = datetime_series(df[self.date_col])
            if self.date_start:
                mask &= d >= pd.Timestamp(self.date_start)
            if self.date_end:
                mask &= d < pd.Timestamp(self.date_end) + pd.Timedelta(days=1)
        return df[mask]

    def describe(self) -> tuple[str, ...]:
        out = [f"{c} in ({', '.join(v[:5])}{', ...' if len(v) > 5 else ''})" for c, v in self.categories]
        if self.date_col and (self.date_start or self.date_end):
            out.append(f"{self.date_col} from {self.date_start or 'start'} to {self.date_end or 'end'}")
        return tuple(out)

    def key(self) -> str:
        return json.dumps([self.categories, self.date_col, self.date_start, self.date_end], default=str)


def cache_key(*parts: object) -> str:
    """Stable short hash of anything JSON-like. Use (dataset fingerprint, cleaning fingerprint,
    model.fingerprint(), overrides, filter.key(), config) so any change invalidates stale results."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


# ---- engine context ---------------------------------------------------------------------------
class _Ctx:
    def __init__(self, df: pd.DataFrame, model: SemanticModel, config: AnalyticsConfig, flt: DataFilter):
        self.df, self.model, self.config, self.flt = df, model, config, flt
        self.filters = flt.describe()
        self.now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._num: dict[str, pd.Series] = {}
        self._dt: dict[str, pd.Series] = {}
        self.base_limits = ((f"Based on only {len(df)} rows; treat as indicative.",)
                            if len(df) < config.min_rows * 5 else ())

    def num(self, col: str) -> pd.Series:
        if col not in self._num:
            self._num[col] = numeric_series(self.df[col])
        return self._num[col]

    def dt(self, col: str) -> pd.Series:
        if col not in self._dt:
            self._dt[col] = datetime_series(self.df[col])
        return self._dt[col]

    def tokens(self, col: str) -> set[str]:
        return token_set(col)

    def rank(self, role: str, *, prefer=(), avoid=(), numeric=True, extra=None) -> list[str]:
        """Columns with `role`, best first: user-chosen, then name preference, then confidence."""
        scored = []
        for i, (name, c) in enumerate(self.model.columns.items()):
            if c.role != role or name not in self.df.columns:
                continue
            if numeric and self.num(name).notna().sum() == 0:
                continue
            t = self.tokens(name)
            bonus = extra(name) if extra else 0
            scored.append((-(c.source == "user"), -len(t & set(prefer)), len(t & set(avoid)), -bonus,
                           -c.confidence, i, name))
        scored.sort()
        return [s[-1] for s in scored]

    def col_notes(self, col: str, alternatives: list[str] | None = None, money: bool = False) -> list[str]:
        notes = []
        c = self.model.columns.get(col)
        if c is not None and c.source != "user" and c.confidence < LOW_CONFIDENCE:
            notes.append(f"Role of '{col}' is a low-confidence suggestion; confirm it on the Semantics page.")
        if alternatives:
            notes.append("Other columns with this role were not used: " + ", ".join(alternatives[:5])
                         + ("..." if len(alternatives) > 5 else "") + ".")
        s = self.df[col]
        miss = int(s.isna().sum())
        if miss:
            notes.append(f"{miss:,} of {len(s):,} rows have no value in '{col}' and are excluded.")
        if money:
            neg = int((self.num(col) < 0).sum())
            if neg:
                notes.append(f"{neg:,} rows have negative values in '{col}' and are included in the total.")
        return notes

    def result(self, key, name, group, value, unit, formula, cols, limits=(), period="", breakdown=()) -> KPIResult:
        return KPIResult(key, name, group, OK, None if value is None else float(value), unit, formula,
                         tuple(cols), self.filters, tuple(self.base_limits) + tuple(limits), self.now,
                         period=period, breakdown=tuple(breakdown))

    def not_calculable(self, key, name, group, unit, formula, cols, reason, limits=()) -> KPIResult:
        return KPIResult(key, name, group, NOT_CALCULABLE, None, unit, formula, tuple(cols), self.filters,
                         tuple(limits) + (f"Not calculable: {reason}",), self.now, reason=reason)

    def skipped(self, key, name, group, unit, reason, formula="") -> KPIResult:
        return KPIResult(key, name, group, SKIPPED, None, unit, formula, (), self.filters, (), self.now,
                         reason=reason)


# ---- column selection (also used by the dashboard) --------------------------------------------
def primary_date_column(df: pd.DataFrame, model: SemanticModel) -> str | None:
    c = _Ctx(df, model, DEFAULT_ANALYTICS, DataFilter())
    ranked = c.rank("datetime", prefer=PREFER_DATE, avoid=AVOID_DATE, numeric=False)
    for col in ranked:
        if c.dt(col).notna().any():
            return col
    return None


def primary_column(df: pd.DataFrame, model: SemanticModel, role: str) -> str | None:
    c = _Ctx(df, model, DEFAULT_ANALYTICS, DataFilter())
    r = c.rank(role, numeric=role not in {"category", "customer", "product", "location", "datetime"})
    return r[0] if r else None


def _order_id(ctx: _Ctx) -> str | None:
    cands = [n for n, c in ctx.model.columns.items()
             if c.role == "identifier" and n in ctx.df.columns
             and (ctx.tokens(n) & ORDER_TOKENS) and not (ctx.tokens(n) & NOT_ORDER_TOKENS)]
    return sorted(cands, key=lambda n: -ctx.model.columns[n].confidence)[0] if cands else None


def _customer(ctx: _Ctx) -> tuple[str | None, list[str]]:
    def bonus(n: str) -> float:
        t = ctx.tokens(n)
        return 2 * bool(t & CUSTOMER_ID_TOKENS) - 2 * bool(t & CUSTOMER_NAME_TOKENS - CUSTOMER_ID_TOKENS) \
            + ctx.df[n].nunique(dropna=True) / max(len(ctx.df), 1)
    r = ctx.rank("customer", numeric=False, extra=bonus)
    return (r[0], r[1:]) if r else (None, [])


def _units(ctx: _Ctx) -> tuple[str | None, list[str]]:
    r = ctx.rank("quantity", prefer=PREFER_UNITS, avoid=AVOID_UNITS)
    good = [c for c in r if not (ctx.tokens(c) & AVOID_UNITS)]
    return (good[0], [c for c in r if c != good[0]]) if good else (None, r)


def _stock(ctx: _Ctx) -> str | None:
    r = [c for c in ctx.rank("quantity", prefer=STOCK_TOKENS) if ctx.tokens(c) & STOCK_TOKENS]
    return r[0] if r else None


def _category(ctx: _Ctx) -> tuple[str | None, list[str]]:
    def bonus(n: str) -> float:
        t = ctx.tokens(n)
        for i, p in enumerate(CATEGORY_PREFER):
            if p in t:
                return len(CATEGORY_PREFER) - i
        return 0
    r = [c for c in ctx.rank("category", numeric=False, extra=bonus)
         if 2 <= ctx.df[c].nunique(dropna=True) <= ctx.config.max_group_cardinality]
    return (r[0], r[1:]) if r else (None, [])


# ---- KPI builders -----------------------------------------------------------------------------
def _revenue(ctx: _Ctx) -> tuple[str | None, list[str]]:
    r = ctx.rank("revenue")
    return (r[0], r[1:]) if r else (None, [])


def _k_rows(ctx: _Ctx) -> list[KPIResult]:
    return [ctx.result("row_count", "Rows analysed", "General", len(ctx.df), "count",
                       "COUNT(rows after filters)", (), ["Each row may be a line item rather than a unique order."])]


def _k_revenue(ctx: _Ctx) -> list[KPIResult]:
    col, alts = _revenue(ctx)
    if col is None:
        return [ctx.skipped("total_revenue", "Total revenue", "Sales", "amount", "No column has the Revenue role.")]
    x = ctx.num(col)
    return [ctx.result("total_revenue", "Total revenue", "Sales", x.sum(), "amount", f"SUM({col})", [col],
                       ctx.col_notes(col, alts, money=True))]


def _k_orders(ctx: _Ctx) -> list[KPIResult]:
    col = _order_id(ctx)
    if col is None:
        return [ctx.skipped("order_count", "Orders", "Sales", "count",
                            "No order, invoice or transaction identifier column was found.")]
    n = ctx.df[col].nunique(dropna=True)
    miss = int(ctx.df[col].isna().sum())
    return [ctx.result("order_count", "Orders", "Sales", n, "count", f"COUNT DISTINCT({col})", [col],
                       [f"{miss:,} rows have no value in '{col}' and are not counted."] if miss else [])]


def _k_units(ctx: _Ctx) -> list[KPIResult]:
    col, alts = _units(ctx)
    if col is None:
        return [ctx.skipped("units", "Units", "Sales", "count",
                            "No quantity column with a sold, ordered or produced meaning was found.")]
    return [ctx.result("units", "Units", "Sales", ctx.num(col).sum(), "count", f"SUM({col})", [col],
                       ctx.col_notes(col, alts))]


def _k_estimated_revenue(ctx: _Ctx) -> list[KPIResult]:
    if ctx.model.columns_with_role("revenue"):
        return []
    q, _ = _units(ctx)
    prices = [c for c in ctx.rank("price") if ctx.tokens(c) & {"unit", "price", "selling", "sale"}]
    if q is None or not prices:
        return []
    p = prices[0]
    prod = ctx.num(q) * ctx.num(p)
    return [ctx.result("estimated_revenue", "Estimated revenue", "Sales", prod.sum(min_count=1), "amount",
                       f"SUM({q} x {p})", [q, p],
                       ["Estimated from units and unit price because no revenue column exists; "
                        "discounts, returns and taxes are not reflected."] + ctx.col_notes(p))]


def _k_aov(ctx: _Ctx) -> list[KPIResult]:
    rev, _ = _revenue(ctx)
    if rev is None:
        return []
    oid = _order_id(ctx)
    total = ctx.num(rev).sum()
    if oid is None:
        n = int(ctx.num(rev).notna().sum())
        if n == 0:
            return [ctx.not_calculable("avg_transaction", "Average value per row", "Sales", "amount",
                                       f"SUM({rev}) / COUNT(rows)", [rev], "no rows have a revenue value")]
        return [ctx.result("avg_transaction", "Average value per row", "Sales", total / n, "amount",
                           f"SUM({rev}) / COUNT(rows with {rev})", [rev],
                           ["No order identifier was found, so this is per row. If rows are order lines, "
                            "it is not the average order value."])]
    orders = ctx.df[oid].nunique(dropna=True)
    if orders == 0:
        return [ctx.not_calculable("aov", "Average order value", "Sales", "amount",
                                   f"SUM({rev}) / COUNT DISTINCT({oid})", [rev, oid], "there are no orders")]
    return [ctx.result("aov", "Average order value", "Sales", total / orders, "amount",
                       f"SUM({rev}) / COUNT DISTINCT({oid})", [rev, oid],
                       ["Revenue of rows with a missing order id is still in the numerator."]
                       if ctx.df[oid].isna().any() else [])]


def _k_category_share(ctx: _Ctx) -> list[KPIResult]:
    rev, _ = _revenue(ctx)
    cat, alts = _category(ctx)
    if rev is None or cat is None:
        return [ctx.skipped("revenue_by_category", "Top category share of revenue", "Sales", "percent",
                            "Needs a Revenue column and a Category column with 2 to "
                            f"{ctx.config.max_group_cardinality} values.")]
    g = pd.DataFrame({"c": ctx.df[cat].astype("string"), "v": ctx.num(rev)}).dropna(subset=["c"]).groupby("c")["v"].sum()
    total = g.sum()
    if total <= 0:
        return [ctx.not_calculable("revenue_by_category", "Top category share of revenue", "Sales", "percent",
                                   f"MAX(SUM({rev}) by {cat}) / SUM({rev})", [rev, cat],
                                   "total revenue is zero or negative")]
    shown = top_n_with_other(g, ctx.config.top_n)
    top = g.idxmax()
    lim = ctx.col_notes(cat, alts) + [f"Top category: {top}. Breakdown shows the top {ctx.config.top_n} and 'Other'."]
    return [ctx.result("revenue_by_category", "Top category share of revenue", "Sales", 100 * g.max() / total,
                       "percent", f"MAX(SUM({rev}) by {cat}) / SUM({rev}) x 100", [rev, cat], lim,
                       breakdown=tuple((str(k), float(v)) for k, v in shown.items()))]


def _growth(ctx: _Ctx, key: str, name: str, value_col: str, group: str) -> list[KPIResult]:
    date = primary_date_column(ctx.df, ctx.model)
    formula = f"(SUM({value_col}, latest complete period) - SUM(comparison period)) / SUM(comparison period) x 100"
    if date is None:
        return [ctx.skipped(key, name, group, "percent", "No usable date column for a comparison period.")]
    freq = ctx.config.time_freq
    try:
        t = aggregate_time(ctx.df, date, value_col, freq, "sum")
    except Exception as e:           # no valid dates
        return [ctx.skipped(key, name, group, "percent", str(getattr(e, "user_message", e)))]
    label = FREQ_LABELS[freq]
    done = t[t["complete"]].reset_index(drop=True)
    if len(done) < 2:
        return [ctx.skipped(key, name, group, "percent",
                            f"Fewer than two complete {label}s of data are available.")]
    cur = done.iloc[-1]
    cur_p = cur["period"]
    if ctx.config.comparison == "year_over_year":
        prev_p = cur_p - (pd.Timedelta(weeks=52) if freq == "W" else pd.DateOffset(years=1))
        cmp_label = "same period one year earlier"
    else:
        prev_p = {"D": cur_p - pd.Timedelta(days=1), "W": cur_p - pd.Timedelta(weeks=1),
                  "M": cur_p - pd.DateOffset(months=1)}[freq]
        cmp_label = f"previous {label}"
    prev = done[done["period"] == prev_p]
    fmt = "%Y-%m-%d" if freq != "M" else "%Y-%m"
    if prev.empty:
        return [ctx.skipped(key, name, group, "percent",
                            f"No complete {cmp_label} ({prev_p.strftime(fmt)}) exists in the data.")]
    pv, cv = float(prev.iloc[0]["value"]), float(cur["value"])
    lim = ctx.col_notes(date)
    if not t.iloc[-1]["complete"]:
        lim.append(f"The latest {label} in the data is partial and was excluded from the comparison.")
    if pv <= 0 or prev.iloc[0]["rows"] == 0:
        return [ctx.skipped(key, name, group, "percent",
                            f"The comparison period total is zero or negative, so growth is undefined.")]
    return [ctx.result(key, name, group, 100 * (cv - pv) / pv, "percent", formula, [value_col, date], lim,
                       period=f"{cur_p.strftime(fmt)} vs {prev_p.strftime(fmt)} ({cmp_label})")]


def _k_growth(ctx: _Ctx) -> list[KPIResult]:
    rev, _ = _revenue(ctx)
    if rev is None:
        return [ctx.skipped("revenue_growth", "Revenue growth", "Sales", "percent", "No column has the Revenue role.")]
    return _growth(ctx, "revenue_growth", "Revenue growth", rev, "Sales")


def _cost_total(ctx: _Ctx) -> tuple[pd.Series | None, list[str], list[str], str, str]:
    """Row-level total cost series, columns used, limitations, formula, reason-if-none."""
    r = ctx.rank("cost")
    if not r:
        return None, [], [], "", "No column has the Cost role."
    col = r[0]
    notes = ctx.col_notes(col, r[1:], money=True)
    t = ctx.tokens(col)
    if t & UNIT_COST_TOKENS and not (t & TOTAL_TOKENS):
        q, _ = _units(ctx)
        if q is None:
            return None, [col], [], "", f"'{col}' looks like a unit cost and no units column exists to scale it."
        return (ctx.num(col) * ctx.num(q), [col, q], notes + [f"'{col}' is treated as a per-unit cost and multiplied by '{q}'."],
                f"SUM({col} x {q})", "")
    return ctx.num(col), [col], notes, f"SUM({col})", ""


def _k_cost_profit(ctx: _Ctx) -> list[KPIResult]:
    out: list[KPIResult] = []
    rev, _ = _revenue(ctx)
    cost, ccols, cnotes, cformula, creason = _cost_total(ctx)
    if cost is None:
        out.append(ctx.skipped("total_cost", "Total cost", "Profit and cost", "amount", creason))
    else:
        out.append(ctx.result("total_cost", "Total cost", "Profit and cost", cost.sum(min_count=1), "amount",
                              cformula, ccols, cnotes))
    prof = ctx.rank("profit")
    if prof:
        pcol = prof[0]
        p = ctx.num(pcol)
        out.append(ctx.result("total_profit", "Total profit", "Profit and cost", p.sum(min_count=1), "amount",
                              f"SUM({pcol})", [pcol], ctx.col_notes(pcol, prof[1:], money=True)))
        profit_series, pcols, pformula, pnotes = p, [pcol], pcol, []
    elif cost is not None and rev is not None and len(ccols) == 1:
        profit_series = ctx.num(rev) - cost
        pcols, pformula = [rev] + ccols, f"{rev} - {ccols[0]}"
        pnotes = ["Derived as revenue minus cost because no profit column exists; assumes both are in the "
                  "same unit and cost covers all expenses."]
        out.append(ctx.result("total_profit", "Net profit (revenue - cost)", "Profit and cost",
                              profit_series.sum(min_count=1), "amount", f"SUM({pformula})", pcols, pnotes))
    else:
        out.append(ctx.skipped("total_profit", "Total profit", "Profit and cost", "amount",
                               "No Profit column, and not enough Revenue and Cost data to derive it."))
        profit_series = None
    if rev is None or profit_series is None:
        out.append(ctx.skipped("profit_margin", "Profit margin", "Profit and cost", "percent",
                               "Needs both Revenue and Profit (or Cost) columns."))
        return out
    r = ctx.num(rev)
    both = r.notna() & profit_series.notna()
    denominator = float(r[both].sum())
    formula = f"SUM({pformula}) / SUM({rev}) x 100 (rows with both values)"
    cols = list(dict.fromkeys(pcols + [rev]))
    excluded = int((~both).sum())
    lim = ([f"{excluded:,} rows lack a revenue or profit value and are left out of both sums."] if excluded else [])
    lim.append("Computed from totals, not as the average of row-level margins.")
    if both.sum() == 0 or denominator == 0:
        out.append(ctx.not_calculable("profit_margin", "Profit margin", "Profit and cost", "percent", formula, cols,
                                      "total revenue is zero or no rows have both values", lim))
    else:
        out.append(ctx.result("profit_margin", "Profit margin", "Profit and cost",
                              100 * float(profit_series[both].sum()) / denominator, "percent", formula, cols, lim))
    return out


def _k_customers(ctx: _Ctx) -> list[KPIResult]:
    col, alts = _customer(ctx)
    if col is None:
        return [ctx.skipped("distinct_customers", "Distinct customers", "Customers", "count",
                            "No column has the Customer role.")]
    notes = ctx.col_notes(col, alts)
    s = ctx.df[col]
    n = int(s.nunique(dropna=True))
    out = [ctx.result("distinct_customers", "Distinct customers", "Customers", n, "count",
                      f"COUNT DISTINCT({col})", [col], notes)]
    rev, _ = _revenue(ctx)
    if rev is None:
        return out + [ctx.skipped(k, nm, "Customers", u, "Needs a Revenue column.") for k, nm, u in
                      (("avg_spend", "Average spend per customer", "amount"),
                       ("concentration", "Customer revenue concentration", "percent"))] + _k_repeat(ctx, col)
    spend = pd.DataFrame({"c": s, "v": ctx.num(rev)}).dropna(subset=["c"])
    per = spend.groupby("c")["v"].sum()
    if n == 0:
        out.append(ctx.not_calculable("avg_spend", "Average spend per customer", "Customers", "amount",
                                      f"SUM({rev}) / COUNT DISTINCT({col})", [rev, col], "there are no customers"))
    else:
        out.append(ctx.result("avg_spend", "Average spend per customer", "Customers", per.sum() / n, "amount",
                              f"SUM({rev}) / COUNT DISTINCT({col})", [rev, col],
                              ["Rows with no customer value are excluded from the revenue total."]))
    share = ctx.config.concentration_top_share
    cname = f"Top {share:.0%} of customers' share of revenue"
    cformula = f"SUM({rev} of top {share:.0%} customers) / SUM({rev}) x 100"
    if n < 10:
        out.append(ctx.skipped("concentration", cname, "Customers", "percent", "Fewer than 10 customers."))
    elif per.sum() <= 0:
        out.append(ctx.not_calculable("concentration", cname, "Customers", "percent", cformula, [rev, col],
                                      "total revenue is zero or negative"))
    else:
        k = max(1, math.ceil(share * n))
        top = per.sort_values(ascending=False).head(k).sum()
        out.append(ctx.result("concentration", cname, "Customers", 100 * top / per.sum(), "percent", cformula,
                              [rev, col], [f"Top group = {k:,} of {n:,} customers."]))
    return out + _k_repeat(ctx, col)


def _k_repeat(ctx: _Ctx, cust: str) -> list[KPIResult]:
    oid = _order_id(ctx)
    name = "Repeat customer rate"
    if oid is None:
        return [ctx.skipped("repeat_rate", name, "Customers", "percent",
                            "Needs an order identifier to count orders per customer.")]
    f = pd.DataFrame({"c": ctx.df[cust], "o": ctx.df[oid]}).dropna()
    per = f.groupby("c")["o"].nunique()
    formula = f"COUNT(customers with 2+ distinct {oid}) / COUNT(customers) x 100"
    if len(per) == 0:
        return [ctx.not_calculable("repeat_rate", name, "Customers", "percent", formula, [cust, oid], "no customers")]
    return [ctx.result("repeat_rate", name, "Customers", 100 * float((per >= 2).mean()), "percent", formula,
                       [cust, oid], ["Counts orders inside the selected filters and data range only."])]


def _k_inventory(ctx: _Ctx) -> list[KPIResult]:
    stock = _stock(ctx)
    if stock is None:
        return [ctx.skipped(k, n, "Inventory", u, "No stock-on-hand quantity column (inventory, stock or level) found.")
                for k, n, u in (("stock_units", "Stock on hand", "count"), ("stock_value", "Stock value", "amount"),
                                ("reorder_candidates", "Rows below reorder point", "count"))]
    date = primary_date_column(ctx.df, ctx.model)
    mask = pd.Series(True, index=ctx.df.index)
    snap_note = "No date column: every row is treated as a separate stock position."
    if date is not None:
        d = ctx.dt(date)
        last = d.max()
        mask = d.dt.normalize() == last.normalize()
        snap_note = f"Latest snapshot only ({last.date()}); earlier dates are not added up."
    qty = ctx.num(stock)[mask]
    cols = [stock] + ([date] if date else [])
    notes = ctx.col_notes(stock) + [snap_note]
    out = [ctx.result("stock_units", "Stock on hand", "Inventory", qty.sum(min_count=1), "count",
                      f"SUM({stock}) at latest snapshot", cols, notes,
                      breakdown=tuple((k, float(v)) for k, v in zip(
                          ("min", "25%", "median", "75%", "max"),
                          qty.quantile([0, .25, .5, .75, 1]).tolist()))) if qty.notna().any() else
           ctx.not_calculable("stock_units", "Stock on hand", "Inventory", "count", f"SUM({stock})", cols,
                              "no stock values in the latest snapshot")]
    costs = [c for c in ctx.rank("cost") if not (ctx.tokens(c) & TOTAL_TOKENS)]
    unit_costs = [c for c in costs if ctx.tokens(c) & UNIT_COST_TOKENS] or costs
    if not unit_costs:
        out.append(ctx.skipped("stock_value", "Stock value", "Inventory", "amount", "Needs a unit cost column."))
    else:
        cc = unit_costs[0]
        val = (ctx.num(stock) * ctx.num(cc))[mask]
        out.append(ctx.result("stock_value", "Stock value", "Inventory", val.sum(min_count=1), "amount",
                              f"SUM({stock} x {cc}) at latest snapshot", cols + [cc],
                              notes + [f"'{cc}' is assumed to be a cost per unit."] + ctx.col_notes(cc)))
    rp = [c for c in ctx.rank("quantity", prefer=REORDER_TOKENS) if ctx.tokens(c) & REORDER_TOKENS]
    if not rp:
        out.append(ctx.skipped("reorder_candidates", "Rows below reorder point", "Inventory", "count",
                               "No reorder-point column found."))
    else:
        both = ctx.num(stock).notna() & ctx.num(rp[0]).notna() & mask
        below = int((ctx.num(stock)[both] < ctx.num(rp[0])[both]).sum())
        out.append(ctx.result("reorder_candidates", "Rows below reorder point", "Inventory", below, "count",
                              f"COUNT({stock} < {rp[0]}) at latest snapshot", cols + [rp[0]],
                              notes + [f"{int(both.sum()):,} rows in the snapshot were compared."]))
    return out


def _k_general_numeric(ctx: _Ctx) -> list[KPIResult]:
    cands = [n for n, c in ctx.model.columns.items()
             if c.kind == "numeric" and c.role not in EXCLUDED_NUMERIC_ROLES and n in ctx.df.columns
             and ctx.num(n).nunique() > 1]
    out = []
    for col in cands[:ctx.config.max_general_numeric]:
        x = ctx.num(col)
        role = ctx.model.role_of(col)
        note = [f"No business meaning is assumed for '{col}'; totals of rates or levels may not be meaningful."]
        if role != "percentage":
            out.append(ctx.result(f"sum_{col}", f"Total of {col}", "General", x.sum(min_count=1), "number",
                                  f"SUM({col})", [col], note + ctx.col_notes(col)))
        out.append(ctx.result(f"mean_{col}", f"Average of {col}", "General", x.mean(), "number",
                              f"AVG({col})", [col], note + ctx.col_notes(col)))
    return out


_BUILDERS = (_k_revenue, _k_estimated_revenue, _k_orders, _k_units, _k_aov, _k_category_share, _k_growth,
             _k_cost_profit, _k_customers, _k_inventory)


def compute_kpis(df: pd.DataFrame, model: SemanticModel, config: AnalyticsConfig = DEFAULT_ANALYTICS,
                 data_filter: DataFilter | None = None) -> KPIReport:
    """Compute every KPI whose roles exist. One failing KPI never stops the others."""
    flt = data_filter or DataFilter()
    filtered = flt.apply(df)
    total = len(df)
    if len(filtered) == 0:
        msg = "The dataset is empty." if total == 0 else "No rows match the current filters."
        return KPIReport((), 0, total, model.domain.domain, (msg,))
    ctx = _Ctx(filtered, model, config, flt)
    results = _k_rows(ctx)
    for fn in _BUILDERS:
        try:
            results.extend(fn(ctx))
        except Exception:
            log.exception("KPI builder %s failed", fn.__name__)
            results.append(ctx.skipped(fn.__name__.removeprefix("_k_"), fn.__name__.removeprefix("_k_").replace("_", " ").title(),
                                       "General", "number", "Could not be calculated for this dataset."))
    if not any(r.status == OK and r.key != "row_count" for r in results):
        try:
            results.extend(_k_general_numeric(ctx))
        except Exception:
            log.exception("General KPI builder failed")
    notes = []
    if flt.active:
        notes.append(f"Filters active: {'; '.join(flt.describe())}. Showing {len(filtered):,} of {total:,} rows.")
    return KPIReport(tuple(results), len(filtered), total, model.domain.domain, tuple(notes))
