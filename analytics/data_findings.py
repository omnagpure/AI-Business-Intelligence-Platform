"""Descriptive patterns read straight from the data, for the insight engine. Pure pandas; never mutates the input.

Each section fails on its own (returns nothing) so one unusual column cannot hide the others."""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analytics.data_checks import parse_flag
from analytics.descriptive import datetime_series, numeric_series
from analytics.kpis import primary_column, primary_date_column
from config.analytics_config import DEFAULT_INSIGHTS, InsightConfig
from core.logging_config import get_logger
from semantic.column_lexicon import token_set

log = get_logger("analytics.findings")
GROUP_ROLES = ("supplier", "warehouse", "location", "category", "product")
ENTITY_ROLES = ("product", "supplier", "customer")
FORECAST_TOKENS = frozenset({"forecast", "forecasted", "predicted", "prediction"})
ACTUAL_TOKENS = frozenset({"sold", "sales", "actual", "units", "demand"})


@dataclass(frozen=True)
class GroupGap:
    group: str
    measure: str
    high_label: str
    low_label: str
    high_mean: float
    low_mean: float
    overall: float
    gap_pct: float
    n_groups: int
    min_rows: int


@dataclass(frozen=True)
class FlagEffect:
    flag: str
    measure: str
    mean_flag: float
    mean_other: float
    lift_pct: float
    share_flagged: float
    n_flag: int
    n_other: int


@dataclass(frozen=True)
class WeekdayPattern:
    measure: str
    peak_day: str
    low_day: str
    peak_mean: float
    low_mean: float
    mean: float
    spread_pct: float
    n_days: int


@dataclass(frozen=True)
class Trend:
    measure: str
    latest: float
    previous: float
    change_pct: float
    period: str


@dataclass(frozen=True)
class Concentration:
    entity: str
    measure: str
    n_entities: int
    top_n: int
    share_pct: float


@dataclass(frozen=True)
class ForecastAccuracy:
    forecast_col: str
    actual_col: str
    wape_pct: float
    bias_pct: float
    n_rows: int


@dataclass(frozen=True)
class DataFindings:
    measure: str | None
    group_gaps: tuple[GroupGap, ...] = ()
    flag_effects: tuple[FlagEffect, ...] = ()
    weekday: WeekdayPattern | None = None
    trend: Trend | None = None
    concentrations: tuple[Concentration, ...] = ()
    forecast_accuracy: ForecastAccuracy | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _fin(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


def _group_gaps(df, model, measures, cfg) -> list[GroupGap]:
    out = []
    for g in [c for r in GROUP_ROLES for c in model.columns_with_role(r) if c in df.columns]:
        key = df[g].astype("string")
        if not 2 <= key.nunique() <= cfg.group_max_groups:
            continue
        for m in measures:
            if m == g:
                continue
            x = numeric_series(df[m])
            frame = pd.DataFrame({"k": key, "x": x}).dropna()
            stats = frame.groupby("k")["x"].agg(["mean", "count"])
            stats = stats[stats["count"] >= cfg.group_min_rows]
            overall = float(frame["x"].mean()) if len(frame) else float("nan")
            if len(stats) < 2 or not _fin(overall) or overall <= 0:
                continue
            hi, lo = stats["mean"].idxmax(), stats["mean"].idxmin()
            gap = 100 * float(stats.loc[hi, "mean"] - stats.loc[lo, "mean"]) / overall
            if _fin(gap):
                out.append(GroupGap(g, m, str(hi), str(lo), float(stats.loc[hi, "mean"]), float(stats.loc[lo, "mean"]),
                                    overall, gap, int(len(stats)), int(stats["count"].min())))
    return sorted(out, key=lambda r: (-r.gap_pct, r.group, r.measure))


def _flag_effects(df, model, measure, cfg) -> list[FlagEffect]:
    out = []
    x = numeric_series(df[measure])
    for c in model.columns_with_role("flag"):
        if c not in df.columns or c == measure:
            continue
        flags, _ = parse_flag(df[c])
        if flags is None:
            continue
        frame = pd.DataFrame({"f": flags, "x": x}).dropna()
        n1, n0 = int(frame["f"].sum()), int((~frame["f"].astype(bool)).sum())
        total = n1 + n0
        if total == 0 or min(n1, n0) / total < cfg.flag_min_share:
            continue
        m1, m0 = float(frame.loc[frame["f"].astype(bool), "x"].mean()), float(frame.loc[~frame["f"].astype(bool), "x"].mean())
        if _fin(m1, m0) and m0 > 0:
            out.append(FlagEffect(c, measure, m1, m0, 100 * (m1 - m0) / m0, n1 / total, n1, n0))
    return sorted(out, key=lambda r: (-abs(r.lift_pct), r.flag))


def _weekday(df, date_col, measure, cfg) -> WeekdayPattern | None:
    d, x = datetime_series(df[date_col]), numeric_series(df[measure])
    frame = pd.DataFrame({"d": d.dt.normalize(), "x": x}).dropna()
    daily = frame.groupby("d")["x"].sum()
    if len(daily) < cfg.weekday_min_days:
        return None
    by = daily.groupby(daily.index.dayofweek).agg(["mean", "count"])
    if len(by) < 7 or by["count"].min() < 8 or float(daily.mean()) <= 0:
        return None
    hi, lo = int(by["mean"].idxmax()), int(by["mean"].idxmin())
    spread = 100 * float(by.loc[hi, "mean"] - by.loc[lo, "mean"]) / float(daily.mean())
    return WeekdayPattern(measure, calendar.day_name[hi], calendar.day_name[lo], float(by.loc[hi, "mean"]),
                          float(by.loc[lo, "mean"]), float(daily.mean()), spread, int(len(daily)))


def _trend(df, date_col, measure) -> Trend | None:
    d, x = datetime_series(df[date_col]), numeric_series(df[measure])
    frame = pd.DataFrame({"d": d, "x": x}).dropna()
    if frame.empty:
        return None
    sums = frame.groupby(frame["d"].dt.to_period("M"))["x"].sum()
    first, last = frame["d"].min(), frame["d"].max()
    if first.day != 1:
        sums = sums.drop(first.to_period("M"), errors="ignore")             # partial first month
    if last.day != last.days_in_month:
        sums = sums.drop(last.to_period("M"), errors="ignore")              # partial last month
    if len(sums) < 2 or sums.index[-1].ordinal - sums.index[-2].ordinal != 1 or sums.iloc[-2] <= 0:
        return None
    latest, prev = float(sums.iloc[-1]), float(sums.iloc[-2])
    return Trend(measure, latest, prev, 100 * (latest - prev) / prev, f"{sums.index[-1]} vs {sums.index[-2]}")


def _concentration(df, model, measure, cfg) -> list[Concentration]:
    out, x = [], numeric_series(df[measure])
    for e in [c for r in ENTITY_ROLES for c in model.columns_with_role(r) if c in df.columns
              and not token_set(str(c)) & {"image", "url", "link", "photo", "thumbnail", "picture", "path"}]:
        sums = pd.DataFrame({"k": df[e].astype("string"), "x": x}).dropna().groupby("k")["x"].sum()
        if len(sums) < cfg.pareto_min_entities or (sums < 0).any() or float(sums.sum()) <= 0:
            continue
        top_n = max(1, math.ceil(0.10 * len(sums)))
        out.append(Concentration(e, measure, int(len(sums)), top_n, 100 * float(sums.nlargest(top_n).sum() / sums.sum())))
    out.sort(key=lambda r: (-r.share_pct, "name" not in token_set(str(r.entity)), r.entity))
    seen, unique = set(), []
    for r in out:                       # two columns with the same count and share are the same thing (Product Id / Product Name)
        key = (r.n_entities, round(r.share_pct, 1))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _forecast_accuracy(df, model, cfg) -> ForecastAccuracy | None:
    nums = [c for c in df.columns if model.role_of(c) in ("quantity", "revenue", "unknown")]
    fc = [c for c in nums if token_set(str(c)) & FORECAST_TOKENS]
    actual = sorted((c for c in nums if c not in fc and token_set(str(c)) & ACTUAL_TOKENS and model.role_of(c) == "quantity"),
                    key=lambda c: (-len(token_set(str(c)) & ACTUAL_TOKENS), c))
    if not fc or not actual:
        return None
    f, a = numeric_series(df[fc[0]]), numeric_series(df[actual[0]])
    both = pd.DataFrame({"f": f, "a": a}).dropna()
    if len(both) < cfg.group_min_rows or float(both["a"].sum()) <= 0:
        return None
    sa = float(both["a"].sum())
    return ForecastAccuracy(fc[0], actual[0], 100 * float((both["f"] - both["a"]).abs().sum()) / sa,
                            100 * float(both["f"].sum() - sa) / sa, int(len(both)))


def compute_findings(df: pd.DataFrame, model, config: InsightConfig = DEFAULT_INSIGHTS) -> DataFindings:
    """Group gaps, flag effects, weekday pattern, latest-month trend, concentration and forecast-column accuracy."""
    measure = primary_column(df, model, "revenue") or primary_column(df, model, "quantity")
    date_col = primary_date_column(df, model)
    duration = primary_column(df, model, "duration")
    measures = [m for m in dict.fromkeys([measure, duration]) if m]
    notes: list[str] = []
    res: dict = {}

    def part(name, fn):
        try:
            return fn()
        except Exception:
            log.exception("Data finding %s failed", name)
            notes.append(f"The {name} check could not be completed on this data.")
            return None
    if measures:
        res["group_gaps"] = tuple(part("group comparison", lambda: _group_gaps(df, model, measures, config)) or ())
    if measure:
        res["flag_effects"] = tuple(part("flag comparison", lambda: _flag_effects(df, model, measure, config)) or ())
        res["concentrations"] = tuple(part("concentration", lambda: _concentration(df, model, measure, config)) or ())
        if date_col:
            res["weekday"] = part("weekday pattern", lambda: _weekday(df, date_col, measure, config))
            res["trend"] = part("trend", lambda: _trend(df, date_col, measure))
    res["forecast_accuracy"] = part("forecast accuracy", lambda: _forecast_accuracy(df, model, config))
    return DataFindings(measure, notes=tuple(notes), **res)
