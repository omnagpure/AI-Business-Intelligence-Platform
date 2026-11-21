"""Deterministic insight rules. Each rule reads computed results only and returns zero or more insights.

A rule never recomputes statistics itself; it reads KPI results, the data profile and the Phase 6 results,
compares them with the thresholds in InsightConfig and states which threshold was crossed."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from analytics.anomaly_detection import METHOD_LABELS
from analytics.formatting import format_value
from analytics.kpi_relevance import is_supply_chain
from analytics.kpis import NOT_CALCULABLE, OK, SKIPPED, KPIReport, KPIResult
from config.analytics_config import InsightConfig
from insights.evidence import Evidence, Insight
from semantic.column_lexicon import token_set

EMPTY_PCT = 99.5                     # a column this empty is ignored by every analysis: a note, not a problem
TIME_WORDS = frozenset({"day", "days", "time", "lead", "transit", "shipping", "delivery", "duration", "hour", "hours"})
SERVICE_WORDS = frozenset({"mode", "class", "service", "method", "priority", "carrier", "shipping"})
SUPPLY_CHAIN_KEYS = {"on_time_rate", "fill_rate", "otif", "stockout_rate", "defect_rate", "below_reorder_share",
                     "supplier_top_share", "avg_delay"}


@dataclass(frozen=True)
class InsightInputs:
    """Results the engine may read. Any field can be None; rules that need it are reported as skipped."""
    kpis: KPIReport | None = None
    profile: object | None = None          # ingestion.profiler.DatasetProfile
    forecast: object | None = None         # analytics.forecasting.ForecastResult
    anomalies: object | None = None        # analytics.anomaly_detection.AnomalyReport
    segmentation: object | None = None     # analytics.segmentation.SegmentationResult
    drivers: object | None = None          # analytics.drivers.DriverResult
    findings: object | None = None         # analytics.data_findings.DataFindings
    unavailable: tuple[tuple[str, str], ...] = ()   # (input name, reason) for analyses that were tried but could not run


# ---- helpers -----------------------------------------------------------------------------------
def _conf(rows: int, cfg: InsightConfig) -> str:
    return "high" if rows >= cfg.confidence_rows_high else "medium" if rows >= cfg.confidence_rows_medium else "low"


def _ok(report: KPIReport, key: str) -> KPIResult | None:
    r = report.get(key)
    return r if r is not None and r.status == OK and r.value is not None and math.isfinite(r.value) else None


def _kpi_ev(r: KPIResult) -> Evidence:
    return Evidence("kpi", r.key, format_value(r.value, r.unit), r.formula, tuple(r.columns), r.period)


def _group(key: str) -> str:
    return "Supply chain" if key in SUPPLY_CHAIN_KEYS else "Performance"


# ---- KPI rules ---------------------------------------------------------------------------------
def rule_growth(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    out = []
    for key, label in (("revenue_growth", "Revenue"), ("spend_growth", "Spend")):
        r = _ok(inp.kpis, key)
        if r is None or abs(r.value) < cfg.growth_medium_pct:
            continue                                   # small changes are not reported
        g = r.value
        out.append(Insight(
            id=f"growth:{key}", rule="growth", group="Performance",
            title=f"{label} {'rose' if g > 0 else 'fell'} {abs(g):.1f}% against the comparison period",
            description=f"{label} changed by {g:+.1f}% ({r.period}). Only complete periods are compared.",
            evidence=(_kpi_ev(r),), severity="high" if abs(g) >= cfg.growth_high_pct else "medium",
            confidence=_conf(inp.kpis.rows_used, cfg), subject=label.lower(), direction="up" if g > 0 else "down",
            period=r.period, thresholds=f"Medium at {cfg.growth_medium_pct:g}% change or more, high at {cfg.growth_high_pct:g}%.",
            limitations=tuple(r.limitations)))
    return out


def rule_category_share(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = _ok(inp.kpis, "revenue_by_category")
    if r is None or r.value < cfg.category_share_medium:
        return []
    named = [(k, v) for k, v in r.breakdown if k != "Other"]
    top = max(named, key=lambda kv: kv[1])[0] if named else "the top category"
    return [Insight(
        id="category_share:revenue_by_category", rule="category_share", group="Performance",
        title=f"One category holds {r.value:.1f}% of revenue",
        description=f"'{top}' accounts for {r.value:.1f}% of total revenue ({' / '.join(r.columns)}).",
        evidence=(_kpi_ev(r),), severity="high" if r.value >= cfg.category_share_high else "medium",
        confidence=_conf(inp.kpis.rows_used, cfg), subject=str(top), direction="high",
        thresholds=f"Medium at {cfg.category_share_medium:g}% or more, high at {cfg.category_share_high:g}%.",
        limitations=tuple(r.limitations))]


def _threshold_rule(table_attr: str, low: bool):
    def rule(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
        out = []
        for key, medium, high in getattr(cfg, table_attr):
            r = _ok(inp.kpis, key)
            if r is None:
                continue
            v = r.value
            bad_high = v < high if low else v >= high
            bad_med = v < medium if low else v >= medium
            if not bad_med:
                continue
            word = "below" if low else "at or above"
            out.append(Insight(
                id=f"threshold:{key}", rule="threshold", group=_group(key),
                title=f"{r.name} is {v:.1f}%, {'well ' if bad_high else ''}{'below' if low else 'above'} the {medium:g}% reference level",
                description=f"{r.name} is {v:.1f}%, {word} the {medium:g}% reference level set in this app.",
                evidence=(_kpi_ev(r),), severity="high" if bad_high else "medium",
                confidence=_conf(inp.kpis.rows_used, cfg), subject=r.name, direction="low" if low else "high",
                period=r.period,
                thresholds=f"Medium {word} {medium:g}%, high {'below' if low else 'at or above'} {high:g}% (adjustable reference levels, not industry standards).",
                limitations=tuple(r.limitations)))
        return out
    return rule


rule_low_kpis = _threshold_rule("low_is_bad", True)
rule_high_kpis = _threshold_rule("high_is_bad", False)


def rule_margin(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = _ok(inp.kpis, "profit_margin")
    if r is None or r.value >= cfg.low_margin_pct:
        return []
    neg = r.value < 0
    return [Insight(
        id="margin:profit_margin", rule="margin", group="Performance",
        title="Overall profit margin is negative" if neg else f"Overall profit margin is low ({r.value:.1f}%)",
        description=f"Profit margin is {r.value:.1f}% (sum of profit / sum of revenue over rows with both values).",
        evidence=(_kpi_ev(r),), severity="high" if neg else "medium", confidence=_conf(inp.kpis.rows_used, cfg),
        subject="profit margin", direction="low", thresholds=f"Flagged below {cfg.low_margin_pct:g}%; negative is always high.",
        limitations=tuple(r.limitations))]


def rule_kpi_unavailable(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    """KPIs whose columns exist but whose value cannot be calculated reliably (medium), and, for supply-chain data,
    core delivery/stock/quality KPIs that are missing because the columns are absent (low)."""
    rep = inp.kpis
    broken = [r for r in rep.results if r.status == NOT_CALCULABLE]
    absent = [r for r in rep.results if r.status == SKIPPED and r.key in SUPPLY_CHAIN_KEYS] if is_supply_chain(rep) else []
    items = broken + absent
    if not items:
        return []
    ev = tuple(Evidence("kpi", r.key, "Not calculated", r.reason or "required columns not found", tuple(r.columns))
               for r in items[:cfg.max_items_listed])
    names = ", ".join(r.name for r in items[:cfg.max_items_listed])
    return [Insight(
        id="kpi_unavailable:relevant", rule="kpi_unavailable", group="Data quality",
        title=f"{len(items)} important KPI{'s' if len(items) != 1 else ''} could not be calculated",
        description=f"Not calculated for this data: {names}. See the reasons in the evidence.",
        evidence=ev, severity="medium" if broken else "low", confidence="high", subject=names, direction="missing",
        thresholds="Medium when a KPI's columns exist but its value cannot be calculated; low when only columns are absent.")]


# ---- data quality rules ------------------------------------------------------------------------
def rule_missing(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    p = inp.profile
    cols = sorted((c for c in p.columns_profile if c.missing_pct >= cfg.missing_medium_pct),
                  key=lambda c: (-c.missing_pct, c.name))
    if not cols:
        return []
    ev = tuple(Evidence("profile", "missing_pct", f"{c.missing_pct:.1f}%", f"{c.missing:,} of {p.rows:,} values are empty",
                        (c.name,)) for c in cols[:cfg.max_items_listed])
    partial = [c for c in cols if c.missing_pct < EMPTY_PCT]
    empty = [c for c in cols if c.missing_pct >= EMPTY_PCT]
    worst = partial[0] if partial else cols[0]
    s = "s" if len(cols) != 1 else ""
    if partial:
        title = f"{len(cols)} column{s} with {cfg.missing_medium_pct:g}% or more missing values"
        desc = (f"'{worst.name}' is the most incomplete ({worst.missing_pct:.1f}% empty). Analyses that use these columns "
                "rest on fewer rows.")
        if empty:
            desc += (f" {len(empty)} of them ({', '.join(repr(c.name) for c in empty[:3])}) "
                     f"{'is' if len(empty) == 1 else 'are'} completely empty and simply ignored.")
        sev = "high" if worst.missing_pct >= cfg.missing_high_pct else "medium"
    else:
        title = f"{len(cols)} column{s} {'is' if len(cols) == 1 else 'are'} completely empty"
        desc = (f"{', '.join(repr(c.name) for c in cols[:3])}{' and others' if len(cols) > 3 else ''} have no values. "
                "Every analysis ignores them, so they do no harm; they can be dropped from the export.")
        sev = "low"
    return [Insight(
        id="missing:columns", rule="missing", group="Data quality", title=title, description=desc,
        evidence=ev, severity=sev, confidence="high", subject=worst.name, direction="high" if partial else "empty",
        thresholds=f"Medium at {cfg.missing_medium_pct:g}% missing, high at {cfg.missing_high_pct:g}%; completely empty "
                   "columns are low.")]


def rule_duplicates(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    p = inp.profile
    if p.duplicate_pct < cfg.duplicate_medium_pct:
        return []
    return [Insight(
        id="duplicates:rows", rule="duplicates", group="Data quality",
        title=f"{p.duplicate_pct:.1f}% of rows are exact duplicates",
        description=f"{p.duplicate_rows:,} of {p.rows:,} rows repeat another row exactly. Totals may be overstated if they are errors.",
        evidence=(Evidence("profile", "duplicate_pct", f"{p.duplicate_pct:.1f}%", f"{p.duplicate_rows:,} duplicate rows of {p.rows:,}"),),
        severity="medium", confidence="high", subject="duplicate rows", direction="high",
        thresholds=f"Flagged at {cfg.duplicate_medium_pct:g}% or more.",
        limitations=("Identical rows can be genuine (for example two identical sales); nothing was deleted.",))]


def rule_small_dataset(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    p = inp.profile
    if p.rows >= cfg.min_rows_high:
        return []
    return [Insight(
        id="small_dataset:rows", rule="small_dataset", group="Data quality", title=f"Only {p.rows:,} rows: results are weak",
        description="With so few rows most statistics and every model are unreliable.",
        evidence=(Evidence("profile", "rows", f"{p.rows:,}", "number of rows analysed"),),
        severity="high", confidence="high", subject="dataset size", direction="low",
        thresholds=f"High below {cfg.min_rows_high} rows.")]


# ---- Phase 6 rules -----------------------------------------------------------------------------
def rule_forecast_direction(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = inp.forecast
    h = len(r.forecast)
    hist = r.history["value"].dropna()
    if r.chosen == r.baseline or h == 0 or len(hist) < h:
        return []                       # a baseline forecast is flat by construction; it says nothing about direction
    base, fc = float(hist.tail(h).mean()), float(r.forecast["forecast"].mean())
    if not (math.isfinite(base) and math.isfinite(fc)) or abs(base) < 1e-9:
        return []
    pct = 100 * (fc - base) / abs(base)
    if abs(pct) < cfg.forecast_change_medium_pct:
        return []
    target = r.value_col or "row count"
    imp = r.improvement_vs_baseline_pct
    conf = "high" if imp is not None and imp >= cfg.forecast_good_improvement_pct else "medium"
    return [Insight(
        id=f"forecast_direction:{target}", rule="forecast_direction", group="Forecast",
        title=f"Forecast: {target} {'rises' if pct > 0 else 'falls'} about {abs(pct):.0f}% versus the latest {h} periods",
        description=f"The validated model's average forecast for the next {h} periods is {pct:+.1f}% against the average of the "
                    f"latest {h} observed periods.",
        evidence=(Evidence("forecast", "forecast_vs_recent", f"{pct:+.1f}%", f"mean forecast {fc:,.2f} vs mean of last {h} periods {base:,.2f}",
                           (r.date_col,) + ((r.value_col,) if r.value_col else ())),
                  Evidence("forecast", "improvement_vs_baseline", "n/a" if imp is None else f"{imp:.1f}%",
                           f"selected model '{r.chosen}' versus baseline '{r.baseline}' in out-of-sample validation")),
        severity="high" if abs(pct) >= cfg.forecast_change_high_pct else "medium", confidence=conf, subject=target,
        direction="up" if pct > 0 else "down",
        thresholds=f"Medium at {cfg.forecast_change_medium_pct:g}% change or more, high at {cfg.forecast_change_high_pct:g}%.",
        limitations=tuple(r.warnings) + tuple(r.limitations))]


def rule_forecast_reliability(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = inp.forecast
    imp = r.improvement_vs_baseline_pct
    if r.chosen != r.baseline and imp is not None and imp > 0:
        return []
    return [Insight(
        id="forecast_reliability:baseline", rule="forecast_reliability", group="Forecast",
        title="No forecasting model clearly beat the simple baseline",
        description="In out-of-sample validation the simple baseline was as good as the more complex models, so the forecast "
                    "adds little over repeating recent values.",
        evidence=(Evidence("forecast", "improvement_vs_baseline", "n/a" if imp is None else f"{imp:.1f}%",
                           f"selected '{r.chosen}', baseline '{r.baseline}'"),),
        severity="low", confidence="high", subject=r.value_col or "row count", direction="weak",
        thresholds="Reported when the selected model is the baseline or its improvement is not positive.",
        limitations=tuple(r.warnings))]


def rule_anomalies(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    rep = inp.anomalies
    s = rep.summary
    s = s[s["status"] == "ok"]
    if s.empty:
        return []
    best = s.sort_values(["share", "feature", "method"], ascending=[False, True, True]).drop_duplicates("feature")
    best = best[best["share"] >= cfg.anomaly_share_medium]
    if best.empty:
        return []
    ev = tuple(Evidence("anomaly", "flag_share", f"{row.share:.1%}",
                        f"{int(row.flagged):,} of {int(row.examined):,} values flagged by {METHOD_LABELS.get(row.method, row.method)}",
                        (str(row.feature),)) for row in best.head(cfg.max_items_listed).itertuples())
    top = best.iloc[0]
    return [Insight(
        id="anomalies:extreme_values", rule="anomalies", group="Anomalies",
        title=f"{len(best)} column{'s' if len(best) != 1 else ''} with an unusually large share of extreme values",
        description=f"'{top['feature']}' has the largest share ({top['share']:.1%}) of values far from typical.",
        evidence=ev, severity="high" if float(top["share"]) >= cfg.anomaly_share_high else "medium", confidence="medium",
        subject=str(top["feature"]), direction="high",
        thresholds=f"Medium at {cfg.anomaly_share_medium:.0%} of values flagged, high at {cfg.anomaly_share_high:.0%} (sensitivity {rep.sensitivity}).",
        limitations=(rep.note, "Skewed columns (such as order values) naturally have a long tail, which inflates flag counts.") + tuple(rep.warnings))]


def rule_segments(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = inp.segmentation
    if r.silhouette < cfg.seg_min_silhouette:
        return []
    out = []
    prof = r.profile
    for cluster in prof.index:
        share = float(prof.loc[cluster, "share"])
        if share > cfg.seg_small_share:
            continue
        for f in r.features:
            if r.units.get(f) in ("percent", "days") or f not in prof.columns:     # rates and times have no "total"
                continue
            overall, mean = float(r.overall[f]), float(prof.loc[cluster, f])
            if not (math.isfinite(overall) and math.isfinite(mean)) or overall <= 0 or mean / overall < cfg.seg_value_ratio:
                continue
            ratio, value_share = mean / overall, share * mean / overall
            out.append(Insight(
                id=f"segments:{cluster}:{f}", rule="segments", group="Segments",
                title=f"A small segment ({share:.0%} of entities) averages {ratio:.1f}x the overall {r.labels.get(f, f)}",
                description=f"Segment {cluster} ({int(prof.loc[cluster, 'entities'])} entities) has a mean {r.labels.get(f, f)} of "
                            f"{mean:,.2f} against {overall:,.2f} overall; roughly {value_share:.0%} of the total. {r.traits.get(cluster, '')}".strip(),
                evidence=(Evidence("segmentation", "segment_vs_overall", f"{ratio:.2f}x", f"segment mean {mean:,.2f} / overall mean {overall:,.2f}", (f,)),
                          Evidence("segmentation", "silhouette", f"{r.silhouette:.2f}", f"{r.k} clusters, {r.scaler} scaling")),
                severity="high" if value_share >= cfg.seg_value_share_high else "medium",
                confidence="high" if r.silhouette >= 0.5 else "medium", subject=f"segment {cluster}", direction="high",
                thresholds=f"Segment at most {cfg.seg_small_share:.0%} of entities with a mean at least {cfg.seg_value_ratio:g}x overall.",
                limitations=(r.note,) + tuple(r.warnings)))
    out.sort(key=lambda i: i.id)
    return out[:cfg.max_items_listed]


def _driver_group_line(r, feature: str) -> tuple[str, float]:
    """('yes for 95.5% of First Class rows against 55.4% overall', relative gap) from the driver's group table, or ('', 0)."""
    g = (getattr(r, "groups", None) or {}).get(feature)
    overall = getattr(r, "overall", float("nan"))
    if g is None or len(g) < 2 or not math.isfinite(overall) or overall == 0:
        return "", 0.0
    hi = g.loc[g["target_mean"].idxmax()]
    lo = g.loc[g["target_mean"].idxmin()]
    pick = hi if abs(hi["target_mean"] - overall) >= abs(lo["target_mean"] - overall) else lo
    v = float(pick["target_mean"])
    if r.task == "classification":
        text = f"{r.target} is yes for {v:.1%} of rows where {feature} is '{pick['group']}', against {overall:.1%} overall"
    else:
        text = f"{r.target} averages {v:,.2f} where {feature} is '{pick['group']}', against {overall:,.2f} overall"
    return text, abs(v - overall) / abs(overall)


def rule_drivers(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = inp.drivers
    imp = r.importance
    if not r.reliable or imp is None or imp.empty or getattr(r, "near_perfect", False):
        return []
    top = imp.sort_values("importance", ascending=False).iloc[0]
    share = float(top["share"])
    if share < cfg.driver_top_share:
        return []
    dominated = share >= 0.90
    m = r.metrics.iloc[0]
    feature = str(top["feature"])
    line, gap = _driver_group_line(r, feature)
    direction = str(top.get("direction", ""))
    if not line and direction.startswith("higher values go with"):
        line = f"higher {feature} goes with {'higher' if 'higher target' in direction else 'lower'} {r.target}"
    lim = [r.note, "Importance shows association in this data, not cause and effect."] + list(r.warnings)
    if dominated:
        lim.append("One column holds almost all the importance; it may be a near-copy of the target, so check it before treating it as a driver.")
    desc = f"Of the columns in this data, '{feature}' says the most about {r.target} ({share:.0%} of the model's explanatory power)."
    if line:
        desc += f" {line[:1].upper() + line[1:]}."
    return [Insight(
        id=f"drivers:{r.target}", rule="drivers", group="Drivers",
        title=f"'{feature}' is most associated with {r.target} ({share:.0%} of importance)",
        description=desc,
        evidence=(Evidence("drivers", "importance_share", f"{share:.1%}", "permutation importance share on held-out rows", (feature,)),
                  Evidence("drivers", str(m["metric"]), f"{float(m['model']):.3f}", f"baseline {float(m['baseline']):.3f}; {r.split}", (r.target,))),
        severity="medium" if gap >= 0.25 and not dominated else "info", confidence="low" if dominated else "medium",
        subject=r.target, direction="association",
        thresholds=f"Reported when the model is reliable and the top factor holds at least {cfg.driver_top_share:.0%} of importance; "
                   "medium when its strongest group differs from the overall level by 25% or more.",
        limitations=tuple(lim))]


# ---- data-pattern rules (read straight from the data) --------------------------------------------
def _rows_conf(n: int, cfg: InsightConfig) -> str:
    return _conf(n, cfg)


def rule_group_gap(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    out, seen = [], set()
    gaps = sorted((g for g in inp.findings.group_gaps if g.gap_pct >= cfg.group_gap_medium_pct), key=lambda g: -g.gap_pct)
    for g in gaps:
        if g.measure in seen or len(out) >= 3:
            continue                                     # one finding per measure: the largest gap says it
        seen.add(g.measure)
        by_design = bool(token_set(g.measure) & TIME_WORDS) and bool(token_set(g.group) & SERVICE_WORDS)
        lim = ("Differences in group averages are descriptive; they are not tested for significance and other factors may explain them.",)
        if by_design:
            lim = (f"Faster {g.group} options are meant to be faster, so this gap is largely by design. Compare actual against "
                   "promised times per option (Drivers page) to see which option misses its promise.",) + lim
        out.append(Insight(
            id=f"group_gap:{g.group}:{g.measure}", rule="group_gap", group="Performance",
            title=f"{g.measure} varies by {g.gap_pct:.0f}% of its average across {g.group}",
            description=f"By {g.group}, '{g.high_label}' has the highest mean {g.measure} ({g.high_mean:,.2f}) and '{g.low_label}' the "
                        f"lowest ({g.low_mean:,.2f}); the overall mean is {g.overall:,.2f} across {g.n_groups} groups.",
            evidence=(Evidence("data", "group_gap_pct", f"{g.gap_pct:.1f}%",
                               f"(highest group mean - lowest group mean) / overall mean, groups with at least {cfg.group_min_rows} rows",
                               (g.group, g.measure)),),
            severity="low" if by_design else "high" if g.gap_pct >= cfg.group_gap_high_pct else "medium",
            confidence=_rows_conf(g.min_rows, cfg), subject=f"{g.measure} by {g.group}", direction="by_design" if by_design else "gap",
            thresholds=f"Medium at {cfg.group_gap_medium_pct:g}% or more, high at {cfg.group_gap_high_pct:g}%; low when the "
                       "groups are service levels and the measure is a time.",
            limitations=lim))
    return out


def rule_flag_effect(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    out = []
    for f in [f for f in inp.findings.flag_effects if abs(f.lift_pct) >= cfg.flag_lift_medium_pct][:3]:
        up = f.lift_pct > 0
        out.append(Insight(
            id=f"flag_effect:{f.flag}:{f.measure}", rule="flag_effect", group="Performance",
            title=f"Rows where {f.flag} is set average {abs(f.lift_pct):.0f}% {'higher' if up else 'lower'} {f.measure}",
            description=f"Mean {f.measure} is {f.mean_flag:,.2f} where {f.flag} is set ({f.n_flag:,} rows, {f.share_flagged:.0%} of rows) "
                        f"and {f.mean_other:,.2f} where it is not ({f.n_other:,} rows).",
            evidence=(Evidence("data", "flag_lift_pct", f"{f.lift_pct:+.1f}%", "(mean with flag - mean without) / mean without",
                               (f.flag, f.measure)),),
            severity="high" if abs(f.lift_pct) >= cfg.flag_lift_high_pct else "medium", confidence=_rows_conf(min(f.n_flag, f.n_other), cfg),
            subject=f.flag, direction="up" if up else "down",
            thresholds=f"Medium at {cfg.flag_lift_medium_pct:g}% difference or more, high at {cfg.flag_lift_high_pct:g}%.",
            limitations=("This is an association in this data, not proof that the flag causes the difference.",)))
    return out


def rule_weekday(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    w = inp.findings.weekday
    if w is None or w.spread_pct < cfg.weekday_spread_medium_pct:
        return []
    return [Insight(
        id=f"weekday:{w.measure}", rule="weekday", group="Performance",
        title=f"{w.measure} peaks on {w.peak_day} and is weakest on {w.low_day}",
        description=f"Average daily total {w.measure} is {w.peak_mean:,.2f} on {w.peak_day} and {w.low_mean:,.2f} on {w.low_day} "
                    f"(the overall daily average is {w.mean:,.2f}, {w.n_days} days).",
        evidence=(Evidence("data", "weekday_spread_pct", f"{w.spread_pct:.1f}%", "(best weekday mean - worst weekday mean) / average daily total", (w.measure,)),),
        severity="high" if w.spread_pct >= cfg.weekday_spread_high_pct else "medium",
        confidence="high" if w.n_days >= 180 else "medium", subject=w.measure, direction="pattern",
        thresholds=f"Medium at {cfg.weekday_spread_medium_pct:g}% spread or more, high at {cfg.weekday_spread_high_pct:g}%.")]


def rule_trend(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    t = inp.findings.trend
    if t is None or abs(t.change_pct) < cfg.growth_medium_pct:
        return []
    for key in ("revenue_growth", "spend_growth"):               # the KPI engine already reports these
        r = _ok(inp.kpis, key) if inp.kpis is not None else None
        if r is not None and t.measure in r.columns:
            return []
    up = t.change_pct > 0
    return [Insight(
        id=f"trend:{t.measure}", rule="trend", group="Performance",
        title=f"{t.measure} {'rose' if up else 'fell'} {abs(t.change_pct):.1f}% in the latest complete month",
        description=f"Total {t.measure} was {t.latest:,.2f} against {t.previous:,.2f} in the month before ({t.period}). Partial months are excluded.",
        evidence=(Evidence("data", "monthly_change_pct", f"{t.change_pct:+.1f}%", f"sum of {t.measure} by month, latest complete vs previous", (t.measure,), t.period),),
        severity="high" if abs(t.change_pct) >= cfg.growth_high_pct else "medium", confidence="medium",
        subject=t.measure, direction="up" if up else "down", period=t.period,
        thresholds=f"Medium at {cfg.growth_medium_pct:g}% change or more, high at {cfg.growth_high_pct:g}%.",
        limitations=("A single month-to-month change can reflect calendar effects or seasonality.",))]


def rule_concentration(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    out = []
    for c in [c for c in inp.findings.concentrations if c.share_pct >= cfg.pareto_top_share_medium][:2]:
        out.append(Insight(
            id=f"concentration:{c.entity}:{c.measure}", rule="concentration", group="Performance",
            title=f"The top {c.top_n} of {c.n_entities} {c.entity} values hold {c.share_pct:.0f}% of {c.measure}",
            description=f"The largest 10% of {c.entity} values (top {c.top_n}) account for {c.share_pct:.1f}% of total {c.measure}; "
                        "an even spread would be about 10%.",
            evidence=(Evidence("data", "top_decile_share", f"{c.share_pct:.1f}%", f"sum of {c.measure} for the top 10% of {c.entity} / total", (c.entity, c.measure)),),
            severity="high" if c.share_pct >= cfg.pareto_top_share_high else "medium", confidence="high",
            subject=c.entity, direction="high",
            thresholds=f"Medium at {cfg.pareto_top_share_medium:g}% or more, high at {cfg.pareto_top_share_high:g}%."))
    return out


def rule_forecast_accuracy(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    a = inp.findings.forecast_accuracy
    if a is None:
        return []
    ev = Evidence("data", "wape_pct", f"{a.wape_pct:.1f}%", f"sum |{a.forecast_col} - {a.actual_col}| / sum {a.actual_col}", (a.forecast_col, a.actual_col))
    lim = (f"Compares the '{a.forecast_col}' column in your file with '{a.actual_col}' on {a.n_rows:,} rows; it says nothing about this app's forecast.",)
    out = []
    if a.wape_pct >= cfg.fc_wape_medium_pct:
        out.append(Insight(
            id=f"forecast_accuracy:{a.forecast_col}", rule="forecast_accuracy", group="Supply chain",
            title=f"'{a.forecast_col}' misses actual {a.actual_col} by {a.wape_pct:.0f}%",
            description=f"The planning forecast in the data has a weighted absolute error of {a.wape_pct:.1f}% against actual {a.actual_col}.",
            evidence=(ev,), severity="high" if a.wape_pct >= cfg.fc_wape_high_pct else "medium", confidence="high",
            subject=a.forecast_col, direction="error", thresholds=f"Medium at {cfg.fc_wape_medium_pct:g}% or more, high at {cfg.fc_wape_high_pct:g}%.",
            limitations=lim))
    if abs(a.bias_pct) >= cfg.fc_bias_pct:
        over = a.bias_pct > 0
        out.append(Insight(
            id=f"forecast_bias:{a.forecast_col}", rule="forecast_accuracy", group="Supply chain",
            title=f"'{a.forecast_col}' {'over' if over else 'under'}-forecasts demand by {abs(a.bias_pct):.0f}% overall",
            description=f"Total {a.forecast_col} is {a.bias_pct:+.1f}% against total actual {a.actual_col}.",
            evidence=(Evidence("data", "bias_pct", f"{a.bias_pct:+.1f}%", f"(sum {a.forecast_col} - sum {a.actual_col}) / sum {a.actual_col}", (a.forecast_col, a.actual_col)),),
            severity="medium", confidence="high", subject=a.forecast_col, direction="over" if over else "under",
            thresholds=f"Flagged at {cfg.fc_bias_pct:g}% bias or more.", limitations=lim))
    return out


def rule_cover(inp: InsightInputs, cfg: InsightConfig) -> list[Insight]:
    r = _ok(inp.kpis, "days_of_cover")
    if r is None:
        return []
    v = r.value
    if v < cfg.cover_low_medium_days:
        sev, d, word = ("high" if v < cfg.cover_low_high_days else "medium"), "low", "low"
    elif v > cfg.cover_excess_days:
        sev, d, word = "medium", "high", "high"
    else:
        return []
    return [Insight(
        id="cover:days_of_cover", rule="cover", group="Supply chain",
        title=f"Days of cover is {word}: about {v:.0f} days of sales in stock",
        description=f"Average stock equals about {v:.1f} days of average daily sales."
                    + (" Stock may run out before it is replenished." if d == "low" else " Capital may be tied up in excess stock."),
        evidence=(_kpi_ev(r),), severity=sev, confidence=_conf(inp.kpis.rows_used, cfg), subject="days of cover", direction=d,
        thresholds=f"Low below {cfg.cover_low_medium_days:g} days (high below {cfg.cover_low_high_days:g}); excess above {cfg.cover_excess_days:g} days.",
        limitations=tuple(r.limitations))]


# ---- registry ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    name: str
    needs: str                      # field of InsightInputs that must be present
    fn: Callable[[InsightInputs, InsightConfig], list[Insight]]


RULES: tuple[Rule, ...] = (
    Rule("growth", "kpis", rule_growth),
    Rule("category_share", "kpis", rule_category_share),
    Rule("low_kpis", "kpis", rule_low_kpis),
    Rule("high_kpis", "kpis", rule_high_kpis),
    Rule("margin", "kpis", rule_margin),
    Rule("kpi_unavailable", "kpis", rule_kpi_unavailable),
    Rule("cover", "kpis", rule_cover),
    Rule("group_gap", "findings", rule_group_gap),
    Rule("flag_effect", "findings", rule_flag_effect),
    Rule("weekday", "findings", rule_weekday),
    Rule("trend", "findings", rule_trend),
    Rule("concentration", "findings", rule_concentration),
    Rule("forecast_accuracy", "findings", rule_forecast_accuracy),
    Rule("missing", "profile", rule_missing),
    Rule("duplicates", "profile", rule_duplicates),
    Rule("small_dataset", "profile", rule_small_dataset),
    Rule("forecast_direction", "forecast", rule_forecast_direction),
    Rule("forecast_reliability", "forecast", rule_forecast_reliability),
    Rule("anomalies", "anomalies", rule_anomalies),
    Rule("segments", "segmentation", rule_segments),
    Rule("drivers", "drivers", rule_drivers),
)
