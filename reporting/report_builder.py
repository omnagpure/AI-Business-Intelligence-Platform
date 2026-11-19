"""Collects already-computed results into one renderer-neutral structure (ReportData).

Nothing is calculated here that is not a direct read, count or comparison of a result the app produced. Every
section is either included with content or listed as "not included" with the reason, so a report is never padded
with placeholders. The HTML and PDF writers both read this one structure, so they cannot disagree.
No Streamlit, no I/O.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from analytics.formatting import format_value
from core.logging_config import get_logger
from reporting import charts

log = get_logger("reporting.builder")

MAX_CARDS = 12
MAX_ROWS = 12
SEV_ORDER = ("high", "medium", "low", "info")
SEV_RANK = {s: i for i, s in enumerate(SEV_ORDER)}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ---- structure ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Table:
    title: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str = ""
    numeric_from: int = 99             # columns at or after this index are right-aligned


@dataclass(frozen=True)
class Card:
    title: str
    tag: str                           # e.g. "HIGH · Performance"
    severity: str
    body: str
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Section:
    key: str
    number: int
    title: str
    lead: str = ""
    status: str = "ok"                 # ok | skipped
    reason: str = ""
    facts: tuple[tuple[str, str], ...] = ()
    tables: tuple[Table, ...] = ()
    charts: tuple[tuple[str, bytes], ...] = ()      # (caption, PNG)
    cards: tuple[Card, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportData:
    title: str
    subtitle: str
    prepared_by: str
    dataset_name: str
    is_sample: bool
    generated_at: str
    report_id: str
    headline: tuple[tuple[str, str], ...]
    summary: tuple[str, ...]
    sections: tuple[Section, ...]
    methodology: tuple[str, ...]
    included: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)


# ---- small helpers ------------------------------------------------------------------------------
def clean(text: object, limit: int = 0) -> str:
    """Printable text: control characters removed, optionally shortened. Escaping happens in each writer."""
    s = _CTRL.sub("", "" if text is None else str(text)).replace("\r", " ").replace("\n", " ").strip()
    return s if not limit or len(s) <= limit else s[: limit - 1].rstrip() + "…"


def safe_filename(name: str, fallback: str = "report") -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(name)).strip("._-")
    return (base or fallback)[:60]


def _num(v, digits: int = 2) -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return clean(v)
    if not math.isfinite(f):
        return "–"
    if abs(f) >= 1e6:
        return format_value(f, "number")
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:,.{digits}f}"


def _pct(v, digits: int = 1) -> str:
    return "–" if v is None or not math.isfinite(float(v)) else f"{float(v):.{digits}f}%"


def _period(ts, freq: str) -> str:
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return clean(ts)
    return t.strftime("%b %Y") if str(freq).upper().startswith(("M", "Q", "A", "Y")) else t.strftime("%d %b %Y")


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


# ---- sections -----------------------------------------------------------------------------------
def _skipped(key: str, title: str, reason: str) -> Section:
    return Section(key, 0, title, status="skipped", reason=clean(reason) or "This analysis was not run.")


def _dataset_section(profile, model, cleaning, analysis_rows: int, domain: str) -> Section:
    facts = [("Rows analysed", f"{analysis_rows:,}"), ("Columns", f"{profile.columns:,}"), ("Detected domain", clean(domain).replace("_", " ").title()),
             ("Duplicate rows", f"{profile.duplicate_rows:,} ({_pct(profile.duplicate_pct)})"),
             ("Missing cells", f"{profile.total_missing:,} ({_pct(profile.missing_cells_pct)})")]
    if cleaning is not None:
        facts.append(("Cleaning", f"{cleaning.applied_count} step(s) applied; {cleaning.rows_before:,} to {cleaning.rows_after:,} rows"))
    else:
        facts.append(("Cleaning", "None applied; the original data was analysed"))
    kinds = tuple((k.title(), f"{v:,}") for k, v in sorted(profile.kind_counts.items(), key=lambda kv: -kv[1]) if v)
    tables = [Table("Column types", ("Type", "Columns"), kinds, numeric_from=1)]
    if model is not None:
        rows = []
        for name, sem in model.columns.items():
            if sem.role != "unknown":
                rows.append((clean(name, 40), sem.role.replace("_", " "), f"{sem.confidence:.0%}", "set by you" if sem.source == "user" else "detected"))
        rows.sort(key=lambda r: r[1])
        if rows:
            shown = tuple(rows[:20])
            note = f"{len(rows) - len(shown)} further column(s) with a role are not listed." if len(rows) > len(shown) else ""
            tables.append(Table("Column roles used by the analysis", ("Column", "Role", "Confidence", "Origin"), shown, note, numeric_from=2))
    return Section("dataset", 0, "Dataset", "What was analysed and how the columns were interpreted.", facts=tuple(facts), tables=tuple(tables))


def _quality_section(profile, quality) -> Section:
    sev = {s: len(quality.by_severity(s)) for s in ("high", "medium", "low")}
    lead = (f"Overall status: {quality.status}. Completeness is {_pct(quality.completeness_pct)}; "
            f"{sev['high']} high, {sev['medium']} medium and {sev['low']} low severity finding(s).")
    issues = sorted(quality.issues, key=lambda i: SEV_RANK.get(i.severity, 9))
    rows = tuple((i.severity.title(), clean(i.column or "Whole file", 32), clean(i.message, 150), f"{i.affected_rows:,}" if i.affected_rows else "–",
                  clean(i.suggestion, 120)) for i in issues[:MAX_ROWS])
    tables = []
    if rows:
        note = f"{len(issues) - len(rows)} further finding(s) are on the Data quality page." if len(issues) > len(rows) else ""
        tables.append(Table("Findings, most severe first", ("Severity", "Column", "Finding", "Rows", "Suggested action"), rows, note, numeric_from=99))
    chart_list = []
    miss = sorted(((c.name, c.missing_pct) for c in profile.columns_profile if c.missing_pct and c.missing_pct > 0), key=lambda t: -t[1])[:8]
    if miss:
        png = charts.hbar([m[0] for m in miss], [m[1] for m in miss], fmt=lambda v: f"{v:.1f}", unit="%")
        if png:
            chart_list.append(("Share of values missing, by column (top 8)", png))
    notes = () if rows else ("No data-quality findings were raised.",)
    return Section("quality", 0, "Data quality", lead, tables=tuple(tables), charts=tuple(chart_list), notes=notes)


def _kpi_section(kpis) -> Section:
    ok = [r for r in kpis.results if r.status == "ok" and r.value is not None]
    if not ok:
        return _skipped("kpis", "Key metrics", "No metric could be calculated from the available columns.")
    rows = tuple((r.group, clean(r.name, 46), format_value(r.value, r.unit), clean(r.period, 46) or "All rows") for r in ok)
    calc = tuple((clean(r.name, 40), clean(r.formula, 140), clean(", ".join(r.columns), 60) or "–") for r in ok)
    tables = [Table("Headline figures", ("Group", "Metric", "Value", "Period"), rows, numeric_from=2),
              Table("How each figure was calculated", ("Metric", "Formula", "Columns"), calc)]
    off = [r for r in kpis.results if r.status != "ok" or r.value is None]
    if off:
        tables.append(Table("Metrics that could not be calculated", ("Metric", "Reason"),
                            tuple((clean(r.name, 46), clean(r.reason or "; ".join(r.limitations), 150) or "Not calculable") for r in off)))
    chart_list = []
    br = next((r for r in ok if r.breakdown), None)
    if br is not None:
        png = charts.hbar([b[0] for b in br.breakdown], [b[1] for b in br.breakdown], fmt=lambda v: format_value(v, "amount" if br.unit != "percent" else "number"))
        if png:
            chart_list.append((f"{clean(br.name)}: breakdown", png))
    lead = f"{len(ok)} of {len(kpis.results)} metrics could be calculated from {kpis.rows_used:,} of {kpis.rows_total:,} rows."
    return Section("kpis", 0, "Key metrics", lead, tables=tuple(tables), charts=tuple(chart_list), notes=tuple(clean(n, 200) for n in kpis.notes))


def _insight_sections(rep) -> tuple[Section, Section]:
    ins = list(rep.insights)
    if not ins:
        note = "; ".join(rep.notes) or "No finding crossed a threshold."
        return (Section("insights", 0, "Insights", note), _skipped("recs", "Recommendations", "There were no insights to base recommendations on."))
    counts = {s: sum(1 for i in ins if i.severity == s) for s in SEV_ORDER}
    lead = (f"{len(ins)} insight(s): " + ", ".join(f"{counts[s]} {s}" for s in SEV_ORDER if counts[s]) +
            ". Insights come from fixed, deterministic rules over computed results; each lists its evidence.")
    cards = []
    for i in ins[:MAX_CARDS]:
        det = [("Evidence", " · ".join(f"{clean(e.metric)}: {clean(e.value)}" for e in i.evidence[:4]))]
        if i.period:
            det.append(("Period", clean(i.period)))
        if i.next_step:
            det.append(("Next step", clean(i.next_step)))
        det.append(("Confidence", i.confidence.title()))
        if i.thresholds:
            det.append(("Threshold", clean(i.thresholds, 200)))
        if i.limitations:
            det.append(("Limitations", " ".join(clean(t, 160) for t in i.limitations[:2])))
        cards.append(Card(clean(i.title, 140), f"{i.severity.upper()} · {i.group}", i.severity, clean(i.description, 500), tuple(det)))
    notes = (f"{len(ins) - MAX_CARDS} lower-priority insight(s) are not shown here; see the Insights page.",) if len(ins) > MAX_CARDS else ()
    sec = Section("insights", 0, "Insights", lead, cards=tuple(cards), notes=notes)
    by_id = {i.id: i for i in ins}
    recs = [(r, by_id[r.insight_id]) for r in rep.recommendations if r.insight_id in by_id]
    recs.sort(key=lambda t: -t[0].priority)
    rrows = tuple((str(n), clean(i.title, 90), i.severity.title(), clean(r.text, 260)) for n, (r, i) in enumerate(recs[:10], 1))
    rsec = (Section("recs", 0, "Recommendations", "Each action is template-based and tied to the finding that triggered it.",
                    tables=(Table("Recommended actions, highest priority first", ("#", "Finding", "Severity", "Action"), rrows),))
            if rrows else _skipped("recs", "Recommendations", "No recommendation was generated."))
    return sec, rsec


def _forecast_section(fc) -> Section:
    target = clean(fc.value_col) if fc.value_col else "number of rows"
    chosen = next((s for s in fc.scores if s.key == fc.chosen), None)
    label = chosen.label if chosen else clean(fc.chosen)
    h = len(fc.forecast)
    lead = ""
    if h and len(fc.history) >= h:
        past = float(fc.history["value"].tail(h).mean())
        fut = float(fc.forecast["forecast"].mean())
        if past:
            chg = (fut - past) / abs(past) * 100
            lead = (f"{label} projects an average of {_num(fut)} per period over the next {_plural(h, 'period')}, "
                    f"{abs(chg):.1f}% {'above' if chg >= 0 else 'below'} the average of the last {h} observed period(s) ({_num(past)}).")
    facts = [("Target", f"{target} ({fc.agg})"), ("Date column", clean(fc.date_col)), ("Frequency", clean(fc.freq)), ("Horizon", _plural(fc.horizon, "period")),
             ("Selected model", label), ("Baseline", clean(next((s.label for s in fc.scores if s.key == fc.baseline), fc.baseline))),
             ("Improvement vs baseline", _pct(fc.improvement_vs_baseline_pct) if fc.improvement_vs_baseline_pct is not None else "not established"),
             ("Validation", f"{fc.folds} rolling-origin fold(s) of {_plural(fc.validation_horizon, 'period')}, chronological"),
             ("Seasonality", f"supported (period {fc.seasonal_period})" if fc.seasonality_supported else "not supported by the data"),
             ("Interval level", f"{fc.interval_level:.0%}")]
    score_rows = tuple((s.label + (" (selected)" if s.key == fc.chosen else ""), _num(s.mae), _num(s.rmse), _num(s.mase), _num(s.mape) if s.mape is not None else "n/a",
                        s.status if s.status != "ok" else "") for s in sorted(fc.scores, key=lambda x: (x.mae is None, x.mae or 0.0)))
    fr = tuple((_period(r.period, fc.freq), _num(r.forecast), _num(r.lower), _num(r.upper)) for r in fc.forecast.head(MAX_ROWS).itertuples())
    tables = [Table("Model comparison (out-of-sample, lower is better)", ("Model", "MAE", "RMSE", "MASE", "MAPE %", "Status"), score_rows, numeric_from=1),
              Table("Forecast values", ("Period", "Forecast", "Lower", "Upper"), fr, f"First {len(fr)} of {h} period(s)." if h > len(fr) else "", numeric_from=1)]
    chart_list = []
    png = charts.forecast_chart(fc.history, fc.forecast, target)
    if png:
        chart_list.append((f"{target}: history and forecast", png))
    notes = tuple(clean(t, 220) for t in (*fc.warnings, *fc.limitations))
    return Section("forecast", 0, "Forecast", lead, facts=tuple(facts), tables=tuple(tables), charts=tuple(chart_list), notes=notes)


def _flag_total(summary) -> int:
    """Values flagged by the most sensitive method, per column, summed. Summing across methods would count a value twice."""
    return int(summary.groupby("feature")["flagged"].max().sum()) if len(summary) else 0


def _anomaly_section(an) -> Section:
    s = an.summary
    flagged = _flag_total(s)
    lead = (f"The most sensitive method flagged {flagged:,} value(s) across {s['feature'].nunique() if len(s) else 0} column(s). "
            "A flagged value is unusual, not necessarily wrong; nothing was removed from the data.")
    srow = tuple((clean(r.method), clean(r.feature, 32), f"{int(r.examined):,}", f"{int(r.flagged):,}", _pct(float(r.share) * 100), _num(r.threshold),
                  clean(r.note, 90) if r.status != "ok" else "") for r in s.itertuples())
    tables = [Table("Screening summary", ("Method", "Column", "Examined", "Flagged", "Share", "Threshold", "Note"), srow, numeric_from=2)]
    f = an.findings.head(MAX_ROWS)
    if len(f):
        frows = tuple((clean(r.row, 16), clean(r.feature, 28), _num(r.value), clean(r.method), _num(r.score), clean(r.direction)) for r in f.itertuples())
        tables.append(Table("Strongest flagged values", ("Row", "Column", "Value", "Method", "Score", "Direction"), frows,
                            f"Top {len(frows)} of {len(an.findings):,} listed findings, by score.", numeric_from=99))
    chart_list = []
    if len(s) and flagged:
        per = s.groupby("feature")["flagged"].max().sort_values(ascending=False).head(8)
        png = charts.hbar(list(per.index), list(per.values), fmt=lambda v: f"{v:,.0f}")
        if png:
            chart_list.append(("Flagged values by column (highest count across methods)", png))
    return Section("anomalies", 0, "Anomalies", lead, tables=tuple(tables), charts=tuple(chart_list),
                   notes=tuple(clean(t, 220) for t in (*an.warnings, an.note)))


def _segment_section(sg) -> Section:
    sizes = sg.profile["entities"]
    feats = list(sg.features)
    shown = feats[:4]
    lead = (f"{int(sizes.sum()):,} entities were grouped into {sg.k} segment(s). Silhouette {sg.silhouette:.2f}: {clean(sg.quality)}")
    header = ("Segment", "Entities", "Share") + tuple(clean(sg.labels.get(f, f), 22) for f in shown)

    def cell(v, f):
        u = sg.units.get(f, "number")
        return _pct(v) if u == "percent" else _num(v)
    rows = tuple((f"Segment {int(i)}", f"{int(r['entities']):,}", _pct(float(r["share"]) * 100), *[cell(r[f], f) for f in shown]) for i, r in sg.profile.iterrows())
    facts = (("Grouped by", clean(sg.assignments.columns[0])), ("Segments", f"{sg.k} ({sg.k_mode})"), ("Scaling", clean(sg.scaler)),
             ("Features", ", ".join(clean(sg.labels.get(f, f), 30) for f in feats)))
    tables = [Table("Segment profile (mean of each feature)", header, rows, f"{len(feats) - len(shown)} further feature(s) are shown on the Segmentation page." if len(feats) > len(shown) else "", numeric_from=1)]
    png = charts.hbar([f"Segment {int(i)}" for i in sizes.index], [float(v) for v in sizes.values], fmt=lambda v: f"{v:,.0f}")
    traits = tuple(f"Segment {int(k)}: {clean(v, 260)}" for k, v in sorted(sg.traits.items()))
    return Section("segments", 0, "Segments", lead, facts=facts, tables=tuple(tables), charts=((("Entities per segment", png),) if png else ()),
                   notes=traits + tuple(clean(t, 220) for t in (*sg.warnings, sg.note)))


def _driver_section(dr) -> Section:
    imp = dr.importance.head(8)
    lead = (f"Association with '{clean(dr.target)}' ({dr.task}), tested on {dr.n_test:,} held-out row(s) ({clean(dr.split)}). "
            + ("The model beats its baseline, so the ranking is usable as a guide." if dr.reliable else "The model does not clearly beat its baseline, so treat the ranking as weak.")
            + " These are associations, not causes.")
    mrows = tuple((clean(r.metric), _num(r.model, 3), _num(r.baseline, 3)) for r in dr.metrics.itertuples())
    irows = tuple((clean(r.feature, 34), clean(r.kind), _pct(float(r.share) * 100), clean(r.direction, 60)) for r in imp.itertuples())
    tables = [Table("Held-out performance", ("Metric", "Model", "Baseline"), mrows, numeric_from=1),
              Table("Most influential columns (permutation importance)", ("Column", "Type", "Share", "Direction"), irows, numeric_from=99)]
    png = charts.hbar(list(imp["feature"]), [float(s) * 100 for s in imp["share"]], fmt=lambda v: f"{v:.1f}", unit="%")
    excl = (f"{len(dr.excluded)} column(s) were excluded to avoid leakage, e.g. " + "; ".join(f"{clean(k, 24)} ({clean(v, 60)})" for k, v in list(dr.excluded.items())[:3]),) if dr.excluded else ()
    return Section("drivers", 0, "Drivers", lead, tables=tuple(tables), charts=((("Share of explained influence, top columns", png),) if png else ()),
                   notes=excl + tuple(clean(t, 220) for t in (*dr.warnings, dr.note)))


def _cleaning_section(cleaning) -> Section:
    if cleaning is None:
        return _skipped("cleaning", "Transformation log", "No cleaning was applied; the analysis used the original data.")
    lead = f"{cleaning.applied_count} of {len(cleaning.log)} step(s) were applied. Rows: {cleaning.rows_before:,} before, {cleaning.rows_after:,} after."
    rows = tuple((str(e.step), clean(e.op), clean(e.column or "–", 28), e.status, f"{e.cells_changed:,}", clean(e.detail, 130)) for e in cleaning.log[:20])
    note = f"{len(cleaning.log) - len(rows)} further step(s) are in the transformation log CSV download." if len(cleaning.log) > len(rows) else ""
    return Section("cleaning", 0, "Transformation log", lead, tables=(Table("Cleaning steps", ("Step", "Operation", "Column", "Status", "Cells", "Detail"), rows, note, numeric_from=99),))


METHOD = (
    "All numbers come from the uploaded data through deterministic calculations. No language model or external service was used, and no value in this report is typed in by hand.",
    "Insights are produced by fixed rules with adjustable thresholds. The thresholds are reference levels of this application, not industry standards.",
    "Forecasts are chosen by chronological, rolling-origin validation against a baseline. A forecast is a projection of past patterns, not a guarantee.",
    "Anomaly screening uses robust statistics (median and MAD, IQR, z-score where valid). A flagged value may be a genuine extreme; nothing is deleted.",
    "Segments and driver rankings describe patterns and associations. They do not prove causes, and clusters are one reasonable grouping, not the only one.",
    "Column roles are detected from names and values and can be wrong; roles you set manually are marked as such.",
)


# ---- executive summary --------------------------------------------------------------------------
def _summary(filename, profile, quality, domain, analysis_rows, kpis, rep, forecast, anomalies, segmentation, skipped) -> tuple[str, ...]:
    out = [f"{clean(filename)} was analysed with {analysis_rows:,} rows and {profile.columns:,} columns (detected domain: {clean(domain).replace('_', ' ')}). "
           f"Data quality is rated '{quality.status}' with {_pct(quality.completeness_pct)} completeness."]
    if kpis is not None:
        ok = [r for r in kpis.results if r.status == "ok" and r.value is not None and r.key != "row_count"]
        if ok:
            out.append("Headline figures: " + "; ".join(f"{clean(r.name)} {format_value(r.value, r.unit)}" for r in ok[:4]) + ".")
    if rep is not None and rep.insights:
        top = rep.insights[:3]
        out.append(f"{len(rep.insights)} insight(s) were found. Highest priority: " + " ".join(f"({n}) {clean(i.title, 110)} [{i.severity}]." for n, i in enumerate(top, 1)))
    elif rep is not None:
        out.append("No insight crossed its threshold on this data.")
    if forecast is not None:
        label = next((s.label for s in forecast.scores if s.key == forecast.chosen), forecast.chosen)
        out.append(f"Forecast: {clean(label)} was selected by chronological validation for {_plural(forecast.horizon, 'period')} ahead.")
    if anomalies is not None and len(anomalies.summary):
        out.append(f"Anomaly screening flagged {_flag_total(anomalies.summary):,} value(s) for review (most sensitive method, per column).")
    if segmentation is not None:
        out.append(f"{segmentation.k} segment(s) were found (silhouette {segmentation.silhouette:.2f}).")
    if skipped:
        out.append("Not included: " + "; ".join(f"{t.lower()} ({clean(r, 100).rstrip('.')})" for t, r in skipped[:4]) + ".")
    return tuple(out)


# ---- entry point --------------------------------------------------------------------------------
def build_report(*, filename: str, is_sample: bool, fingerprint: str, profile, quality, analysis_rows: int,
                 kpis=None, insight_report=None, forecast=None, anomalies=None, segmentation=None, drivers=None,
                 cleaning=None, semantic_model=None, unavailable: dict[str, str] | None = None,
                 title: str = "Business Intelligence Report", prepared_by: str = "", generated_at: datetime | None = None,
                 user_settings: tuple[str, ...] = ()) -> ReportData:
    """Assemble a report from computed results. Any result may be None; its section is then marked not included."""
    unavailable = unavailable or {}
    when = generated_at or datetime.now()
    domain = (kpis.domain if kpis is not None and kpis.domain else None) or (semantic_model.domain.domain if semantic_model is not None else "general")

    def guard(key, label, result, fn):
        if result is None:
            return _skipped(key, label, unavailable.get(key, "This analysis was not run for this report."))
        try:
            return fn(result)
        except Exception:                                  # one section must never stop the report
            log.exception("Report section %s failed", key)
            return _skipped(key, label, "The section could not be built from this result.")

    ins_sec, rec_sec = (_skipped("insights", "Insights", unavailable.get("insights", "Insights were not generated.")),
                        _skipped("recs", "Recommendations", unavailable.get("insights", "Insights were not generated.")))
    if insight_report is not None:
        try:
            ins_sec, rec_sec = _insight_sections(insight_report)
        except Exception:
            log.exception("Report insight sections failed")
    parts = [
        guard("dataset", "Dataset", profile, lambda p: _dataset_section(p, semantic_model, cleaning, analysis_rows, domain)),
        guard("quality", "Data quality", quality, lambda q: _quality_section(profile, q)),
        guard("kpis", "Key metrics", kpis, _kpi_section),
        ins_sec, rec_sec,
        guard("forecast", "Forecast", forecast, _forecast_section),
        guard("anomalies", "Anomalies", anomalies, _anomaly_section),
        guard("segments", "Segments", segmentation, _segment_section),
        guard("drivers", "Drivers", drivers, _driver_section),
        _cleaning_section(cleaning),
    ]
    parts = [Section(**{**s.__dict__, "number": n}) for n, s in enumerate(parts, 2)]
    skipped = tuple((s.title, s.reason) for s in parts if s.status == "skipped" and s.key != "cleaning")
    included = tuple(s.title for s in parts if s.status == "ok")
    ok_k = len(kpis.ok()) if kpis is not None else 0
    high = sum(1 for i in insight_report.insights if i.severity == "high") if insight_report is not None else 0
    headline = (("Rows analysed", f"{analysis_rows:,}"), ("Columns", f"{profile.columns:,}"), ("Completeness", _pct(quality.completeness_pct)),
                ("Metrics calculated", f"{ok_k} of {len(kpis.results)}" if kpis is not None else "–"),
                ("High-priority insights", f"{high}" if insight_report is not None else "–"))
    rid = hashlib.sha256(f"{fingerprint}|{when.isoformat()}".encode()).hexdigest()[:8].upper()
    ok = {s.key for s in parts if s.status == "ok"}          # only summarise results whose section was built
    summary = _summary(filename, profile, quality, domain, analysis_rows, kpis if "kpis" in ok else None, insight_report if "insights" in ok else None,
                       forecast if "forecast" in ok else None, anomalies if "anomalies" in ok else None,
                       segmentation if "segments" in ok else None, skipped)
    meth = METHOD + (f"Rows and columns in this report come from the {'cleaned' if cleaning is not None else 'original'} dataset.",
                     ("Settings chosen on the " + ", ".join(user_settings) + " page(s) were used for those analyses; the others ran with default settings.")
                     if user_settings else "All analyses in this report ran with default settings.")
    return ReportData(clean(title, 90) or "Business Intelligence Report", clean(filename, 120), clean(prepared_by, 80), clean(filename, 120), bool(is_sample),
                      when.strftime("%d %b %Y, %H:%M"), f"{rid[:4]}-{rid[4:]}", headline, summary, tuple(parts), meth, included, skipped)
