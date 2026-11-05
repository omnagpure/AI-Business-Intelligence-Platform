"""Descriptive analytics. Pure pandas, no Streamlit. Inputs are never modified.

Identifier, constant, location-coordinate and datetime columns are excluded from numeric summaries so
that, for example, an order id is never averaged. High-cardinality columns are capped and say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from cleaning.type_inference import to_numeric_clean
from config.analytics_config import DEFAULT_ANALYTICS, AnalyticsConfig
from core.exceptions import InsufficientDataError
from ingestion.profiler import DatasetProfile
from ingestion.schema_detector import parse_datetime_series

EXCLUDED_NUMERIC_ROLES = frozenset({"identifier", "datetime", "location", "text", "customer", "product", "category",
                                   "supplier", "warehouse", "flag"})
OTHER_LABEL = "Other"
FREQ_LABELS = {"D": "day", "W": "week", "M": "month"}


# ---- column preparation ---------------------------------------------------------------------
def numeric_series(s: pd.Series) -> pd.Series:
    """Float series with text numbers parsed and +/-inf turned into NaN. Does not modify `s`."""
    if pdt.is_bool_dtype(s):
        x = s.astype("float64")
    elif pdt.is_numeric_dtype(s):
        x = s.astype("float64")
    else:
        x = to_numeric_clean(s).astype("float64")
    return x.replace([np.inf, -np.inf], np.nan)


def datetime_series(s: pd.Series) -> pd.Series:
    """Datetimes (NaT where invalid). Already-datetime columns are returned as they are."""
    if pdt.is_datetime64_any_dtype(s):
        out = s
    else:
        out = parse_datetime_series(s)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out


def analysable_numeric_columns(df: pd.DataFrame, profile: DatasetProfile, model=None) -> list[str]:
    """Numeric columns that make sense to summarise: not identifiers, constants, coordinates or dates."""
    out = []
    for c in profile.columns_profile:
        if c.name not in df.columns or c.kind != "numeric" or c.is_identifier or c.is_constant:
            continue
        if model is not None and c.name in model.columns and model.role_of(c.name) in EXCLUDED_NUMERIC_ROLES:
            continue
        out.append(c.name)
    return out


def groupable_columns(df: pd.DataFrame, profile: DatasetProfile, model=None,
                      config: AnalyticsConfig = DEFAULT_ANALYTICS) -> list[str]:
    """Categorical columns with a usable number of distinct values (2..max_group_cardinality)."""
    out = []
    for c in profile.columns_profile:
        if c.name not in df.columns or c.is_identifier or c.is_constant or c.kind not in ("categorical", "boolean"):
            continue
        if model is not None and c.name in model.columns and model.role_of(c.name) in {"identifier", "text", "datetime"}:
            continue
        if 2 <= c.unique <= config.max_group_cardinality:
            out.append(c.name)
    return out


# ---- numeric / categorical / date summaries ---------------------------------------------------
NUMERIC_SUMMARY_COLUMNS = ["count", "missing", "mean", "median", "std", "min", "q1", "q3", "max", "skew"]


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per column. Skew is left empty when it is not meaningful (fewer than 3 values or no spread)."""
    rows = {}
    for col in columns:
        x = numeric_series(df[col])
        v = x.dropna()
        n = len(v)
        if n == 0:
            rows[col] = {k: np.nan for k in NUMERIC_SUMMARY_COLUMNS} | {"count": 0, "missing": len(x)}
            continue
        std = float(v.std(ddof=1)) if n > 1 else np.nan
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75]).tolist()
        skew = float(v.skew()) if n >= 3 and std and std > 0 else np.nan
        rows[col] = {"count": n, "missing": int(len(x) - n), "mean": float(v.mean()), "median": med, "std": std,
                     "min": float(v.min()), "q1": q1, "q3": q3, "max": float(v.max()), "skew": skew}
    return pd.DataFrame.from_dict(rows, orient="index", columns=NUMERIC_SUMMARY_COLUMNS)


@dataclass(frozen=True)
class CategoricalSummary:
    column: str
    distinct: int
    non_null: int
    missing: int
    top: pd.DataFrame = field(compare=False)       # value, count, pct (share of non-null rows)
    other_count: int = 0                            # rows in values not shown
    note: str = ""


def categorical_summary(df: pd.DataFrame, columns: list[str],
                        config: AnalyticsConfig = DEFAULT_ANALYTICS) -> dict[str, CategoricalSummary]:
    out = {}
    for col in columns:
        s = df[col]
        nn = s.dropna()
        vc = nn.astype(str).value_counts()
        total = int(vc.sum())
        top = vc.head(config.top_n)
        table = pd.DataFrame({"value": top.index, "count": top.to_numpy(),
                              "pct": (100 * top.to_numpy() / total) if total else np.zeros(len(top))})
        note = (f"Showing the {config.top_n} most frequent of {len(vc):,} distinct values."
                if len(vc) > config.top_n else "")
        out[col] = CategoricalSummary(col, int(len(vc)), total, int(s.isna().sum()), table,
                                      int(total - top.sum()), note)
    return out


@dataclass(frozen=True)
class DateSummary:
    column: str
    valid: int
    invalid: int
    min: pd.Timestamp | None
    max: pd.Timestamp | None
    span_days: int
    inferred_freq: str                     # "daily" | "weekly" | "monthly" | "irregular" | "n/a"
    missing_periods: int                   # empty days (daily) / months (monthly) between min and max
    gaps: tuple[tuple[str, str], ...] = ()  # up to 10 (start, end) runs of missing periods


def date_summary(df: pd.DataFrame, column: str) -> DateSummary:
    d = datetime_series(df[column])
    invalid = int(d.isna().sum() - df[column].isna().sum())
    v = d.dropna()
    if v.empty:
        return DateSummary(column, 0, max(invalid, 0), None, None, 0, "n/a", 0)
    days = np.sort(v.dt.normalize().unique())
    span = int((v.max() - v.min()).days)
    freq, missing, gaps = "irregular", 0, []
    if len(days) >= 3:
        step = pd.Series(days).diff().dropna().dt.days
        mode = int(step.mode().iloc[0])
        if mode == 1:
            freq = "daily"
            full = pd.date_range(days[0], days[-1], freq="D")
            absent = full.difference(pd.DatetimeIndex(days))
            missing = len(absent)
            gaps = _runs(absent, pd.Timedelta(days=1))
        elif mode == 7:
            freq = "weekly"
        elif 28 <= mode <= 31:
            freq = "monthly"
            present = pd.PeriodIndex(days, freq="M").unique()
            full = pd.period_range(present.min(), present.max(), freq="M")
            absent = full.difference(present)
            missing = len(absent)
            gaps = _runs(absent.to_timestamp(), pd.Timedelta(days=31))
    return DateSummary(column, int(len(v)), max(invalid, 0), v.min(), v.max(), span, freq, missing, tuple(gaps[:10]))


def _runs(absent: pd.DatetimeIndex, step: pd.Timedelta) -> list[tuple[str, str]]:
    runs, start, prev = [], None, None
    for t in absent:
        if start is None:
            start = prev = t
        elif t - prev <= step:
            prev = t
        else:
            runs.append((str(start.date()), str(prev.date())))
            start = prev = t
    if start is not None:
        runs.append((str(start.date()), str(prev.date())))
    return runs


# ---- aggregation ------------------------------------------------------------------------------
def top_n_with_other(values: pd.Series, n: int = 10, label: str = OTHER_LABEL) -> pd.Series:
    """Keep the n largest entries (by absolute value) and sum the rest into one 'Other' entry."""
    v = values.dropna()
    if len(v) <= n:
        return v.sort_values(ascending=False)
    order = v.abs().sort_values(ascending=False).index
    head = v.loc[order[:n]].sort_values(ascending=False)
    return pd.concat([head, pd.Series({label: v.loc[order[n:]].sum()})])


def aggregate_time(df: pd.DataFrame, date_col: str, value_col: str | None = None, freq: str = "M",
                   agg: str = "sum") -> pd.DataFrame:
    """Aggregate by day/week/month. Columns: period (start timestamp), value, rows, complete.

    `complete` is False for the first or last period when the data does not cover all of it, so growth
    is never computed from a partial month. Periods with no rows are included with value 0 (sum/count)
    or NaN (mean/median) so gaps stay visible. Rows with an invalid date are excluded.
    """
    if freq not in FREQ_LABELS:
        raise ValueError(f"freq must be one of {sorted(FREQ_LABELS)}")
    if agg not in {"sum", "mean", "median", "count"}:
        raise ValueError("agg must be sum, mean, median or count")
    d = datetime_series(df[date_col])
    frame = pd.DataFrame({"d": d})
    if value_col is not None and agg != "count":
        frame["v"] = numeric_series(df[value_col])
    frame = frame.dropna(subset=["d"])
    if frame.empty:
        raise InsufficientDataError("No valid dates are available for a time aggregation.")
    p = frame["d"].dt.to_period(freq)
    grouped = frame.groupby(p)
    rows = grouped.size()
    if agg == "count" or value_col is None:
        vals = rows.astype("float64")
    else:
        vals = grouped["v"].sum(min_count=1) if agg == "sum" else getattr(grouped["v"], agg)()
    full = pd.period_range(rows.index.min(), rows.index.max(), freq=freq)
    vals = vals.reindex(full)
    rows = rows.reindex(full, fill_value=0)
    if agg in ("sum", "count") or value_col is None:
        vals = vals.fillna(0.0)
    out = pd.DataFrame({"period": full.to_timestamp(), "value": vals.to_numpy(dtype="float64"),
                        "rows": rows.to_numpy()})
    lo, hi = frame["d"].min(), frame["d"].max()
    start = full.start_time
    end = full.end_time
    complete = np.ones(len(full), dtype=bool)
    if freq != "D":
        complete[0] = lo.normalize() <= start[0]
        complete[-1] = hi.normalize() >= end[-1].normalize()
    out["complete"] = complete
    return out


def grouped_summary(df: pd.DataFrame, by: str, value: str | None = None, agg: str = "sum",
                    config: AnalyticsConfig = DEFAULT_ANALYTICS) -> pd.DataFrame:
    """Top-N groups plus 'Other'. Columns: group, value, rows, share_pct. Missing group keys are excluded."""
    if agg not in {"sum", "mean", "median", "count"}:
        raise ValueError("agg must be sum, mean, median or count")
    key = df[by].astype("string")
    frame = pd.DataFrame({"g": key})
    if value is not None and agg != "count":
        frame["v"] = numeric_series(df[value])
    frame = frame.dropna(subset=["g"])
    g = frame.groupby("g", observed=True)
    rows = g.size()
    if agg == "count" or value is None:
        vals = rows.astype("float64")
    else:
        vals = getattr(g["v"], agg)()
    if agg in ("sum", "count") or value is None:
        shown = top_n_with_other(vals, config.top_n)
        named = [i for i in shown.index if i != OTHER_LABEL]
        r = rows.reindex(shown.index)
        if OTHER_LABEL in shown.index:
            r[OTHER_LABEL] = int(rows.sum() - rows.loc[named].sum())
        total = shown.sum()
        share = (100 * shown / total) if total else shown * np.nan
    else:      # means/medians cannot be summed into "Other": show the top N groups only
        shown = vals.dropna().sort_values(ascending=False).head(config.top_n)
        r = rows.reindex(shown.index)
        share = pd.Series(np.nan, index=shown.index)
    return pd.DataFrame({"group": shown.index.astype(str), "value": shown.to_numpy(dtype="float64"),
                         "rows": r.to_numpy(dtype="int64"), "share_pct": share.to_numpy(dtype="float64")})


def crosstab(df: pd.DataFrame, row: str, col: str, value: str | None = None, agg: str = "sum",
             config: AnalyticsConfig = DEFAULT_ANALYTICS) -> pd.DataFrame:
    """Cross-tabulation limited to the top-N values on each axis (the rest are dropped, not merged)."""
    r = df[row].astype("string")
    c = df[col].astype("string")
    keep_r = r.value_counts().head(config.top_n).index
    keep_c = c.value_counts().head(config.top_n).index
    mask = r.isin(keep_r) & c.isin(keep_c)
    if value is None or agg == "count":
        return pd.crosstab(r[mask], c[mask])
    v = numeric_series(df[value])[mask]
    return pd.crosstab(r[mask], c[mask], values=v, aggfunc=agg).fillna(0.0)
