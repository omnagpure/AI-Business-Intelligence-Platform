"""Collects already-computed results into one renderer-neutral structure (ReportData).

Nothing is calculated here that is not a direct read, count or comparison of a result the app produced (or of the
app's own engines re-run on a date window, see reporting.data_views). The report reads like a business report:

  cover (title from the data, key message, headline figures) · report at a glance · executive summary ·
  1 Key findings · 2 Performance scorecard · 3 Performance over time · 4 breakdown in detail · 5 Drivers · 6 Segments ·
  7 Outlook · 8 Unusual values · 9 Recommendations and action plan · 10 Conclusion and next steps ·
  appendix: data overview, data quality, metric definitions, glossary, transformation log, methodology.

Every section has a short title (contents, running header), a headline that states its main point, and a one-line
description with real numbers for the contents page. Analyses that could not run are listed once, with the reason.
The HTML and PDF writers both read this one structure, so they cannot disagree. No Streamlit, no I/O.
"""
from __future__ import annotations

import hashlib
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import PurePath

import pandas as pd

from analytics.formatting import format_value
from analytics.kpi_relevance import prioritise
from analytics.kpi_trend import lower_is_better
from analytics.recommendations import KINDS, plan, related_driver
from config.analytics_config import DEFAULT_ANALYTICS, DEFAULT_INSIGHTS, InsightConfig
from config.settings import APP_NAME
from core.logging_config import get_logger
from reporting import charts, theme
from reporting.data_views import Extras, change, gather

log = get_logger("reporting.builder")

MAX_CARDS = 8                      # findings shown in full; the rest are listed in a table
MAX_ROWS = 12
MAX_TILES = 4
MAX_ACTIONS = 5                    # priority actions in the executive summary
SEV_ORDER = ("high", "medium", "low", "info")
SEV_RANK = {s: i for i, s in enumerate(SEV_ORDER)}
SEV_WORD = {"high": "High", "medium": "Medium", "low": "Low", "info": "Info"}
SEV_TONE = {"high": "bad", "medium": "warn", "low": "neutral", "info": "neutral"}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
DEFAULT_TITLE = "Business Intelligence Report"
AUTO_TITLES = {"supply_chain": "Supply Chain Performance Review", "operations": "Operations Performance Review",
               "inventory": "Inventory Performance Review", "retail": "Sales Performance Review", "sales_crm": "Sales Pipeline Review",
               "finance": "Financial Performance Review", "marketing": "Marketing Performance Review",
               "saas": "Product and Revenue Review", "hr": "Workforce Review", "general": "Business Performance Review"}
_GROUP_WORDS = {"Supply chain": "delivery performance", "Sales": "sales", "Profit and cost": "profitability", "Customers": "customers",
                "Inventory": "inventory", "Procurement": "procurement spend"}
_NAMES: ContextVar[tuple] = ContextVar("report_names", default=())


# ---- structure ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Table:
    title: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str = ""
    numeric_from: int = 99             # columns at or after this index are right-aligned (unless `right` is given)
    right: tuple[int, ...] = ()        # explicit right-aligned columns
    tones: tuple[str, ...] = ()        # per row: good | warn | bad | neutral | "" (colours the `tone_col` cell)
    tone_col: int = -1
    sub: tuple[str, ...] = ()          # optional second line under column 1 of each row (greyed)

    def is_right(self, i: int) -> bool:
        return i in self.right if self.right else i >= self.numeric_from


@dataclass(frozen=True)
class Chart:
    title: str
    png: bytes
    note: str = ""                     # one or two sentences saying what the chart shows, computed from its data
    half: bool = False                 # drawn for a two-up row

    def __iter__(self):                # (title, png) unpacking, as the earlier tuples allowed
        return iter((self.title, self.png))


@dataclass(frozen=True)
class Card:
    title: str
    tag: str                           # e.g. "Needs attention · Supply chain"
    severity: str
    body: str
    details: tuple[tuple[str, str], ...] = ()
    kind: str = ""                     # risk | opportunity | pattern | data
    why: str = ""
    action: str = ""
    steps: tuple[str, ...] = ()
    clue: str = ""                     # a related finding that helps explain this one


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
    charts: tuple[Chart, ...] = ()
    cards: tuple[Card, ...] = ()
    notes: tuple[str, ...] = ()
    part: str = "main"                 # main | appendix
    label: str = ""                    # "1", "2" ... in the body, "A", "B" ... in the appendix; "" when not shown
    headline: str = ""                 # the section's main point, shown as its large title
    desc: str = ""                     # one line for the contents page
    callouts: tuple[tuple[str, str, str], ...] = ()   # (value, label, tone) big-number highlights
    steps: tuple[str, ...] = ()        # numbered list (conclusion next steps)
    paragraphs: tuple[str, ...] = ()   # extra body paragraphs after the lead


@dataclass(frozen=True)
class Tile:
    label: str
    value: str
    note: str = ""
    status: str = "neutral"            # good | warn | bad | neutral
    delta: str = ""                    # "▲ 2.1 pts vs Feb 2016 – Jan 2017"
    delta_tone: str = "neutral"


@dataclass(frozen=True)
class Action:
    severity: str
    action: str
    why: str
    finding: str
    steps: tuple[str, ...] = ()
    owner: str = ""
    timeframe: str = ""
    measure: str = ""                  # measure of success


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
    summary: tuple[str, ...]                 # key takeaways as plain sentences (plus what was not included)
    sections: tuple[Section, ...]
    methodology: tuple[str, ...]
    included: tuple[str, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    lede: str = ""                           # first executive-summary paragraph
    tiles: tuple[Tile, ...] = ()             # KPI scorecard on the summary page
    takeaways: tuple[tuple[str, str], ...] = ()   # (kind, text)
    actions: tuple[Action, ...] = ()
    period: str = ""                         # "Jan 2015 – Jan 2018"
    domain_label: str = ""
    quality_line: str = ""
    method_label: str = ""                   # appendix letter of the methodology section
    app_name: str = APP_NAME
    key_message: str = ""                    # one sentence on the cover
    summary_paras: tuple[str, ...] = ()      # further executive-summary paragraphs
    about: tuple[tuple[str, str], ...] = ()  # "About the data" facts
    cover_png: bytes | None = None
    organisation: str = ""
    logo: bytes | None = None
    accent: str = theme.ACCENT
    classification: str = "Internal"
    eyebrow: str = ""                        # small line above the cover title

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)

    def shown(self, part: str) -> tuple[Section, ...]:
        return tuple(s for s in self.sections if s.part == part and s.status == "ok")


# ---- small helpers ------------------------------------------------------------------------------
def readable(name: str) -> str:
    """'Late_delivery_risk' -> 'Late delivery risk', 'OrderDate' -> 'Order Date'; other names unchanged."""
    s = str(name)
    if "_" in s:
        s = re.sub(r"\s+", " ", s.replace("_", " ")).strip()
    elif re.fullmatch(r"[A-Za-z]+", s) and re.search(r"[a-z][A-Z]", s):
        s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return s or str(name)


def _name_map(columns) -> tuple:
    pairs = [(c, readable(c)) for c in columns if isinstance(c, str) and readable(c) != c and len(c) >= 3]
    if not pairs:
        return ()
    pairs.sort(key=lambda p: -len(p[0]))
    rx = re.compile("|".join(r"(?<![\w])" + re.escape(c) + r"(?![\w])" for c, _ in pairs))
    return (rx, dict(pairs))


def clean(text: object, limit: int = 0, raw: bool = False) -> str:
    """Printable text: control characters removed, column names made readable, optionally shortened. Escaping
    happens in each writer. `raw` keeps column names exactly as in the data (appendix tables)."""
    s = _CTRL.sub("", "" if text is None else str(text)).replace("\r", " ").replace("\n", " ").strip()
    names = _NAMES.get()
    if names and not raw:
        rx, m = names
        s = rx.sub(lambda g: m[g.group(0)], s)
    return s if not limit or len(s) <= limit else s[: limit - 1].rstrip() + "…"


def safe_filename(name: str, fallback: str = "report") -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(name, raw=True)).strip("._-")
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


def _human(metric: str) -> str:
    s = clean(metric).replace("_", " ")
    return s[:1].upper() + s[1:]


def _and(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _lc(s: str) -> str:
    """Lower-case the first letter unless the first word looks like a name or acronym."""
    w = s.split(" ", 1)[0]
    return s if (len(w) > 1 and (w[1:2].isupper() or any(ch.isdigit() for ch in w))) else s[:1].lower() + s[1:]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _sentence(s: str) -> str:
    s = _cap(s.strip())
    return s if not s or s.endswith((".", "!", "?")) else s + "."


def _first_sentence(s: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", s.strip())
    return (m.group(1) if m else s).rstrip(".")


_FREQ_WORD = {"D": "day", "W": "week", "M": "month", "MS": "month", "ME": "month", "Q": "quarter", "QS": "quarter", "Y": "year"}


# ---- KPI status against the app's reference levels ------------------------------------------------
def kpi_status(r, cfg: InsightConfig = DEFAULT_INSIGHTS) -> tuple[str, str]:
    """(tone, text) for a KPI: where it stands against the reference levels the insight rules use. ("neutral", "")
    when there is no reference for it."""
    v = r.value
    for key, medium, high in cfg.low_is_bad:
        if r.key == key:
            if v < medium:
                return ("bad" if v < high else "warn"), f"Below {medium:g}% reference"
            return "good", f"Meets {medium:g}% reference"
    for key, medium, high in cfg.high_is_bad:
        if r.key == key:
            if v >= medium:
                return ("bad" if v >= high else "warn"), f"Above {medium:g}% reference"
            return "good", f"Within {medium:g}% reference"
    if r.key == "profit_margin":
        if v < 0:
            return "bad", "Negative margin"
        return ("warn", f"Below {cfg.low_margin_pct:g}% floor") if v < cfg.low_margin_pct else ("good", f"Above {cfg.low_margin_pct:g}% floor")
    if r.key == "revenue_growth":
        if v <= -cfg.growth_medium_pct:
            return "bad", "Falling"
        return ("good", "Growing") if v >= cfg.growth_medium_pct else ("neutral", "Stable")
    return "neutral", ""


def kpi_target(key: str, name: str, cfg: InsightConfig = DEFAULT_INSIGHTS) -> tuple[str, float | None]:
    """('On-time delivery rate at least 90%', 90.0) for a KPI with a reference level, else ('', None)."""
    for k, medium, _ in cfg.low_is_bad:
        if k == key:
            return f"{name} at least {medium:g}%", medium
    for k, medium, _ in cfg.high_is_bad:
        if k == key:
            return f"{name} below {medium:g}%", medium
    if key == "profit_margin":
        return f"{name} above {cfg.low_margin_pct:g}%", cfg.low_margin_pct
    return "", None


def _data_period(profile, model) -> str:
    """'Jan 2015 – Jan 2018' from the first date column the analysis uses, or ''."""
    names = []
    if model is not None:
        names = [c for c, sem in model.columns.items() if sem.role == "datetime"]
    names += [c.name for c in profile.columns_profile if c.kind == "temporal"]
    for name in names:
        try:
            c = profile.column(name)
        except Exception:
            continue
        t = c.temporal or {}
        try:
            lo, hi = pd.Timestamp(t["min"]), pd.Timestamp(t["max"])
        except Exception:
            continue
        if pd.notna(lo) and pd.notna(hi):
            return f"{lo:%b %Y} – {hi:%b %Y}" if (lo.year, lo.month) != (hi.year, hi.month) else f"{lo:%d %b} – {hi:%d %b %Y}"
    return ""


def _fmt_change(unit: str, prior: float, recent: float) -> tuple[str, float | None]:
    c, cu = change(unit, prior, recent)
    if c is None:
        return "–", None
    arrow = "▲" if c > 0 else "▼" if c < 0 else "■"
    return (f"{arrow} {abs(c):.1f} {cu}" if cu == "pts" else f"{arrow} {abs(c):.1f}%"), c


def _change_tone(key: str, c: float | None, unit: str, cfg: InsightConfig = DEFAULT_INSIGHTS) -> str:
    if c is None or abs(c) < (0.1 if unit == "percent" else 0.5):
        return "neutral"
    if "cover" in key or "turnover" in key:            # more stock cover is neither simply good nor bad
        return "neutral"
    lower = lower_is_better(key) or any(k == key for k, _, _ in cfg.high_is_bad)
    better = (c < 0) if lower else (c > 0)
    return "good" if better else "bad"


# ---- sections -----------------------------------------------------------------------------------
def _skipped(key: str, title: str, reason: str, part: str = "main") -> Section:
    return Section(key, 0, title, status="skipped", reason=clean(reason) or "This analysis was not run.", part=part)


def _about(profile, quality, cleaning, analysis_rows: int, domain: str, period: str) -> tuple[tuple[str, str], ...]:
    out = [("Records analysed", f"{analysis_rows:,}"), ("Fields", f"{profile.columns:,}")]
    if period:
        out.append(("Period covered", period))
    out.append(("Business area", clean(domain).replace("_", " ").capitalize()))
    if quality is not None:
        out.append(("Completeness", f"{_pct(quality.completeness_pct)} of cells filled"))
    out.append(("Cleaning", f"{cleaning.applied_count} step(s) applied" if cleaning is not None else "None; original data analysed"))
    return tuple(out)


def _dataset_section(profile, model, cleaning, analysis_rows: int, domain: str, period: str) -> Section:
    facts = [("Rows analysed", f"{analysis_rows:,}"), ("Columns", f"{profile.columns:,}")]
    if period:
        facts.append(("Data covers", period))
    facts += [("Detected domain", clean(domain).replace("_", " ").title()),
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
                rows.append((clean(name, 40, raw=True), sem.role.replace("_", " ").capitalize(), f"{sem.confidence:.0%}",
                             "Set by you" if sem.source == "user" else "Detected"))
        rows.sort(key=lambda r: (r[1], r[0]))
        if rows:
            shown = tuple(rows[:24])
            note = f"{len(rows) - len(shown)} further column(s) with a role are not listed." if len(rows) > len(shown) else ""
            tables.append(Table("How the columns were read (names as in the data)", ("Column", "Role", "Confidence", "Origin"), shown, note,
                                right=(2,)))
    return Section("dataset", 0, "Data overview", "What was analysed and how each column was interpreted.",
                   facts=tuple(facts), tables=tuple(tables), part="appendix",
                   desc=f"{analysis_rows:,} rows and {profile.columns} columns, and how each column was read")


def _quality_section(profile, quality) -> Section:
    sev = {s: len(quality.by_severity(s)) for s in ("high", "medium", "low")}
    lead = (f"Overall the data is rated {quality.status}, with {_pct(quality.completeness_pct)} of cells filled. The checks raised "
            f"{sev['high']} high, {sev['medium']} medium and {sev['low']} low severity issue(s).")
    issues = sorted(quality.issues, key=lambda i: SEV_RANK.get(i.severity, 9))
    rows = tuple((i.severity.title(), clean(i.column or "Whole file", 32, raw=True), clean(i.message, 150), clean(i.suggestion, 120))
                 for i in issues[:MAX_ROWS])
    tones = tuple(SEV_TONE.get(i.severity, "neutral") for i in issues[:MAX_ROWS])
    tables = []
    if rows:
        note = f"{len(issues) - len(rows)} further issue(s) are listed on the Data quality page." if len(issues) > len(rows) else ""
        tables.append(Table("Issues, most severe first", ("Severity", "Column", "Issue", "Suggested action"), rows, note,
                            tones=tones, tone_col=0))
    chart_list = []
    miss = sorted(((c.name, c.missing_pct) for c in profile.columns_profile if c.missing_pct and c.missing_pct > 0), key=lambda t: -t[1])[:8]
    if miss:
        png = charts.hbar([clean(m[0]) for m in miss], [m[1] for m in miss], fmt=lambda v: f"{v:.1f}", unit="%")
        if png:
            note = f"{clean(miss[0][0])} is the least complete column ({miss[0][1]:.1f}% missing)"
            note += f"; {len(miss) - 1} other column(s) have gaps." if len(miss) > 1 else "."
            chart_list.append(Chart("Share of values missing, by column", png, note))
    notes = () if rows else ("No data-quality issues were raised.",)
    return Section("quality", 0, "Data quality", lead, tables=tuple(tables), charts=tuple(chart_list), notes=notes, part="appendix",
                   headline=f"Data rated {quality.status}: {_pct(quality.completeness_pct)} of cells filled",
                   desc=f"Rated {quality.status}; {sev['high']} high and {sev['medium']} medium severity issue(s)")


def _kpi_section(kpis) -> Section:
    ok = [r for r in kpis.results if r.status == "ok" and r.value is not None]
    if not ok:
        return _skipped("kpis", "Performance scorecard", "No metric could be calculated from the available columns.")
    disp = prioritise(kpis, MAX_TILES)
    order = list(disp.headline) + [r for r in disp.more if r.key != "row_count"]
    order += [r for r in ok if r not in order]                            # anything prioritise left out, row count last
    rows, tones = [], []
    for r in order:
        tone, text = kpi_status(r)
        rows.append((clean(r.group), clean(r.name, 46), format_value(r.value, r.unit), text or "–",
                     "All data" if not r.period or r.period == "All rows" else clean(r.period, 46)))
        tones.append(tone if text else "")
    tables = [Table("Key performance indicators", ("Area", "Metric", "Value", "Status", "Period"), tuple(rows),
                    "Status compares each metric with this app's adjustable reference levels, not industry standards.",
                    right=(2,), tones=tuple(tones), tone_col=3)]
    chart_list = []
    br = next((r for r in ok if r.breakdown), None)
    if br is not None:
        vals = [b[1] for b in br.breakdown]
        fmt = lambda v: format_value(v, "amount" if br.unit != "percent" else "number")  # noqa: E731
        png = charts.hbar([clean(b[0]) for b in br.breakdown], vals, fmt=fmt)
        if png:
            tot = sum(v for v in vals if v is not None and math.isfinite(v)) or 1.0
            top = br.breakdown[0]
            note = f"{clean(top[0])} leads with {fmt(top[1])}, {100 * top[1] / tot:.1f}% of the total shown"
            if len(br.breakdown) > 1:
                nxt = br.breakdown[1]
                note += f"; the next is {clean(nxt[0])} ({fmt(nxt[1])})" + (f", so the leader is {top[1] / nxt[1]:.1f}× larger" if nxt[1] else "")
            chart_list.append(Chart(f"{clean(br.name)}: breakdown", png, note + "."))
    stat = [(r, *kpi_status(r)) for r in order]
    flagged = [(clean(r.name), t) for r, tone, t in stat if tone in ("bad", "warn")]
    good = [r for r, tone, t in stat if tone == "good"]
    lead = f"{len(ok)} metrics could be calculated from {kpis.rows_used:,} rows."
    if flagged:
        lead += " Outside their reference level: " + _and([f"{n} ({t.lower()})" for n, t in flagged[:4]]) + "."
    else:
        lead += " None of the metrics with a reference level is outside it."
    rated = len(flagged) + len(good)
    headline = (f"{len(flagged)} of {rated} metrics with a reference level fall short" if flagged and rated
                else "Every metric with a reference level meets it" if rated else f"{len(ok)} metrics calculated from the data")
    callouts = ((f"{len(ok)}", "metrics calculated", "neutral"), (f"{len(flagged)}", "outside reference", "bad" if flagged else "neutral"),
                (f"{len(good)}", "meet reference", "good" if good else "neutral"))
    return Section("kpis", 0, "Performance scorecard", lead, tables=tuple(tables), charts=tuple(chart_list),
                   notes=tuple(clean(n, 200) for n in kpis.notes), headline=headline, callouts=callouts,
                   desc=f"{len(ok)} metrics with their status against reference levels; {len(flagged)} fall short")


def _trend_section(kpis, ex: Extras) -> Section:
    keys = [r.key for r in prioritise(kpis, MAX_TILES).headline] if kpis is not None else []
    keys += [k for k in ex.histories if k not in keys]
    chart_list = []
    for k in keys:
        h = ex.histories.get(k)
        r = kpis.get(k) if kpis is not None else None
        if h is None or r is None:
            continue
        fmt = (lambda v, u=r.unit: format_value(v, u))
        tgt_text, tgt = kpi_target(k, r.name)
        png = charts.trend(h.periods, h.points, fmt=fmt, ref=tgt, ref_label="Reference" if tgt is not None else "")
        if not png:
            continue
        lo = min(range(len(h.points)), key=lambda i: h.points[i])
        hi = max(range(len(h.points)), key=lambda i: h.points[i])
        p = lambda i: pd.Period(h.periods[i]).strftime("%b %Y")             # noqa: E731
        note = (f"Ranged from {fmt(h.points[lo])} ({p(lo)}) to {fmt(h.points[hi])} ({p(hi)}); "
                f"{fmt(h.points[-1])} in the latest month ({p(-1)}).")
        chart_list.append(Chart(f"{clean(r.name)}, by month", png, note, half=True))
        if len(chart_list) >= 4:
            break
    tables, headline, callouts = [], "", []
    comp = ex.comparison
    if comp is not None:
        rows, tones, moves = [], [], []
        for key, name, unit, a, b in comp.rows:
            txt, c = _fmt_change(unit, a, b)
            tone = _change_tone(key, c, unit)
            rows.append((clean(name, 46), format_value(a, unit), format_value(b, unit), txt))
            tones.append(tone)
            rel = abs(change(unit, a, b)[0] or 0) / (1 if unit == "percent" else 1)
            moves.append((key, name, unit, a, b, c, tone, rel))
        tables.append(Table(f"Latest {comp.window} against the {comp.window} before", ("Metric", comp.prior_label, comp.recent_label, "Change"),
                            tuple(rows), "Each window is recomputed with the same formulas as the scorecard. Green is an improvement, red a "
                            "deterioration; for delays, cancellations and costs a fall is an improvement.", right=(1, 2, 3),
                            tones=tuple(tones), tone_col=3))
        prio = {k: i for i, k in enumerate(keys)}
        lead_move = sorted([m for m in moves if m[5] is not None], key=lambda m: (m[0] not in prio, prio.get(m[0], 99)))
        material = lambda m: abs(m[5]) >= (1.0 if m[2] == "percent" else 5.0)     # noqa: E731
        head = [m for m in lead_move if m[0] in prio and material(m)]
        money = [m for k in _HEADLINE_FALLBACK for m in moves if m[0] == k and m[5] is not None and material(m)]

        def say(m):
            key, name, unit, a, b, c, tone, _ = m
            amt = f"{abs(c):.1f} points" if unit == "percent" else f"{abs(c):.1f}%"
            return f"{_lc(clean(name))} {'rose' if c > 0 else 'fell'} {amt}"
        if head:
            s = say(head[0])
            headline = f"Over the latest {comp.window}, {s}" + (f" and {say(money[0])}" if money and money[0][0] != head[0][0] else "")
        elif money:
            names = _and([_lc(clean(m[1])) for m in lead_move if m[0] in prio][:2])
            headline = (f"{names[:1].upper() + names[1:]} held steady over the latest {comp.window}, while {say(money[0])}" if names
                        else f"Over the latest {comp.window}, {say(money[0])}")
        elif lead_move:
            headline = f"The headline metrics held steady over the latest {comp.window}"
        for m in lead_move[:3]:
            key, name, unit, a, b, c, tone, _ = m
            callouts.append((_fmt_change(unit, a, b)[0], f"{clean(name)}, latest {comp.window}", tone))
    if not chart_list and not tables:
        return _skipped("trends", "Performance over time", "The data has no usable date column or too few complete months to show trends.")
    lead = "How the headline metrics moved month by month"
    lead += f" over the last {len(ex.histories[next(iter(ex.histories))].points)} complete months" if ex.histories else ""
    lead += (f", and how the latest {comp.window} ({comp.recent_label}) compare with the {comp.window} before ({comp.prior_label})."
             if comp is not None else ".")
    desc = "Monthly trend of the headline metrics" + (f"; latest {comp.window} against the {comp.window} before" if comp is not None else "")
    return Section("trends", 0, "Performance over time", lead, charts=tuple(chart_list), tables=tuple(tables),
                   headline=headline or "How the headline metrics moved over time", desc=desc, callouts=tuple(callouts))


_HEADLINE_FALLBACK = ("total_revenue", "total_spend", "total_profit", "order_count", "units")
_SC_VIEWS = ("on_time", "lead_time", "stock", "defects", "supplier")


def _breakdown_section(ex: Extras) -> Section:
    chart_list = []
    for v in ex.views:
        png = None
        if v.kind == "bar":
            d = v.data
            fmt = (lambda x: f"{x:.1f}") if v.unit == "%" else charts._num
            png = charts.hbar([clean(x) for x in d[v.x]], [float(x) for x in d[v.y[0]]], fmt=fmt, unit=v.unit if v.unit == "%" else "",
                              ref=v.ref, ref_label="Overall", half=True, highlight=0)
        elif v.kind == "histogram":
            png = charts.histogram(v.data[v.x], unit=v.unit, half=True)
        elif v.kind == "line":
            png = charts.trend(list(v.data[v.x]), [float(x) for x in v.data[v.y[0]]], fmt=(lambda x: f"{x:.1f}%") if v.unit == "%" else charts._num,
                               ref=v.ref, ref_label="Average")
        if png:
            chart_list.append(Chart(clean(v.title), png, clean(v.caption), half=True))
    if not chart_list:
        return _skipped("breakdown", "Breakdown in detail", "No breakdown view could be drawn from the columns available.")
    sc = any(v.key.startswith(_SC_VIEWS) for v in ex.views)
    title = "Operations in detail" if sc else "Breakdown in detail"
    headline = _first_sentence(chart_list[0].note).split(";")[0]
    lead = ("Where performance differs: the same breakdowns the Dashboard suggests for this data, each with what it shows. "
            "Groups with very few rows are left out so one odd record cannot drive a comparison.")
    return Section("breakdown", 0, title, lead, charts=tuple(chart_list), headline=headline,
                   desc=_and([c.title for c in chart_list[:3]]) + (" and more" if len(chart_list) > 3 else ""))


def _definitions_section(kpis) -> Section:
    ok = [r for r in kpis.results if r.status == "ok" and r.value is not None]
    calc = tuple((clean(r.name, 40), clean(r.formula, 140, raw=True), clean(", ".join(r.columns), 60, raw=True) or "–") for r in ok)
    tables = [Table("How each metric is calculated", ("Metric", "Formula", "Columns used"), calc)] if calc else []
    off = [r for r in kpis.results if r.status != "ok" or r.value is None]
    if off:
        tables.append(Table("Metrics that could not be calculated", ("Metric", "Reason"),
                            tuple((clean(r.name, 46), clean(r.reason or "; ".join(r.limitations), 150) or "Not calculable") for r in off)))
    if not tables:
        return _skipped("definitions", "Metric definitions", "No metric was calculated.", "appendix")
    return Section("definitions", 0, "Metric definitions", "The formula and columns behind every figure in the scorecard.",
                   tables=tuple(tables), part="appendix",
                   desc=f"Formulas for {len(ok)} metrics" + (f"; {len(off)} that could not be calculated, and why" if off else ""))


def _finding_card(i, insights) -> Card:
    p = plan(i)
    evidence = "; ".join(f"{_human(e.metric)} {clean(e.value)}" for e in i.evidence[:3])
    det = [("Evidence", evidence)]
    if i.period:
        det.append(("Period", clean(i.period)))
    det.append(("Confidence", i.confidence.title()))
    if i.limitations:
        det.append(("Keep in mind", clean(i.limitations[0], 220)))
    rel = related_driver(i, insights)
    return Card(_cap(clean(i.title, 140)), f"{KINDS[p.kind]} · {clean(i.group)}", i.severity, clean(i.description, 500), tuple(det),
                p.kind, clean(p.why, 240), clean(p.action, 140), tuple(clean(s, 200) for s in p.steps),
                clean(rel.description, 300) if rel else "")


def _findings_section(rep) -> Section:
    ins = list(rep.insights)
    if not ins:
        note = "; ".join(rep.notes) or "No finding crossed a threshold."
        return Section("insights", 0, "Key findings", note, headline="No finding crossed its threshold",
                       desc="No risks, unusual patterns or data problems were found")
    kinds = [plan(i).kind for i in ins]
    n = {k: kinds.count(k) for k in KINDS}
    words = {"risk": ("need attention", "needs attention"), "opportunity": ("opportunities", "opportunity"),
             "pattern": ("patterns", "pattern"), "data": ("data notes", "data note")}
    parts = [f"{n[k]} {words[k][0] if n[k] != 1 else words[k][1]}" for k in KINDS if n[k]]
    lead = (f"The analysis raised {_plural(len(ins), 'finding')}: {_and(parts)}. They are listed by priority. Each states what was "
            "found, why it matters and the recommended action, with the evidence behind it; owners and timeframes are in the action plan.")
    cards = tuple(_finding_card(i, ins) for i in ins[:MAX_CARDS])
    tables = []
    rest = ins[MAX_CARDS:]
    if rest:
        tables.append(Table("Further findings", ("Priority", "Finding", "Type"),
                            tuple((SEV_WORD[i.severity], clean(i.title, 120), KINDS[plan(i).kind]) for i in rest),
                            tones=tuple(SEV_TONE[i.severity] for i in rest), tone_col=0))
    top = next((c for c in cards if c.kind == "risk"), cards[0])
    callouts = ((str(n["risk"]), "need attention", "bad" if n["risk"] else "neutral"),
                (str(n["opportunity"]), "opportunities", "good" if n["opportunity"] else "neutral"),
                (str(n["pattern"] + n["data"]), "patterns and data notes", "neutral"))
    return Section("insights", 0, "Key findings", lead, cards=cards, tables=tuple(tables), headline=top.title, callouts=callouts,
                   desc=f"{_plural(len(ins), 'finding')}, {n['risk']} needing attention, each with why it matters and what to do")


# ---- action plan ----------------------------------------------------------------------------------
_OWNER = {"growth": "Sales lead", "trend": "Sales lead", "category_share": "Category manager", "concentration": "Account manager",
          "margin": "Finance lead", "kpi_unavailable": "Data owner", "missing": "Data owner", "duplicates": "Data owner",
          "small_dataset": "Data owner", "forecast_direction": "Demand planner", "forecast_reliability": "Demand planner",
          "forecast_accuracy": "Demand planner", "anomalies": "Data analyst", "segments": "Account manager",
          "group_gap": "Operations manager", "flag_effect": "Operations manager", "weekday": "Operations manager",
          "drivers": "Operations manager", "cover": "Inventory planner"}
_GROUP_OWNER = {"Supply chain": "Logistics manager", "Inventory": "Inventory planner", "Sales": "Sales lead", "Customers": "Account manager",
                "Profit and cost": "Finance lead", "Procurement": "Procurement lead", "Data quality": "Data owner"}
_WHEN = {"high": "Within 30 days", "medium": "30–90 days", "low": "Next review", "info": "Next review"}


def _kpi_for(i, kpis):
    if kpis is None:
        return None
    t = i.title.lower()
    hits = [r for r in kpis.ok() if r.name.lower() in t]
    return max(hits, key=lambda r: len(r.name), default=None)


def _measure(i, r) -> str:
    subj = clean(i.subject) or "this item"
    if r is not None:
        txt, _ = kpi_target(r.key, r.name)
        if txt:
            return txt
    m = {"missing": f"'{subj}' complete, or dropped if not needed", "duplicates": "No unexplained duplicate rows",
         "kpi_unavailable": "All key metrics calculable", "anomalies": "Every flagged value reviewed",
         "small_dataset": "Enough history for reliable results", "segments": "An approach agreed for each segment",
         "forecast_reliability": "Forecast error below the simple baseline", "forecast_accuracy": "Forecast error tracked monthly"}
    if i.rule in m:
        return m[i.rule]
    if i.rule == "missing" and i.direction == "empty":
        return "Empty columns removed from the export"
    if i.rule in ("growth", "trend"):
        return f"{subj[:1].upper() + subj[1:]} growing again" if i.direction == "down" else f"{subj[:1].upper() + subj[1:]} growth sustained"
    if i.rule == "group_gap":
        return f"Gap in {subj} between groups narrowed"
    if i.rule == "drivers":
        return f"{subj[:1].upper() + subj[1:]} improved in the weakest group"
    return "Target to be set by the owner"


def _actions(insights, limit: int, kpis=None) -> list[Action]:
    out, seen = [], set()
    for i in insights:
        if i.severity == "info" and len(insights) > 3:
            continue
        p = plan(i)
        if not p.action or p.action in seen:
            continue
        seen.add(p.action)
        r = _kpi_for(i, kpis)
        owner = (_GROUP_OWNER.get(r.group) if (i.rule == "threshold" and r is not None) else None) or _OWNER.get(i.rule) \
            or _GROUP_OWNER.get(i.group, "Business owner")
        out.append(Action(i.severity, clean(p.action, 140), clean(p.why, 220), clean(i.title, 140), tuple(clean(s, 200) for s in p.steps),
                          owner, _WHEN.get(i.severity, "Next review"), clean(_measure(i, r), 90)))
        if len(out) >= limit:
            break
    return out


def _plan_section(rep, kpis) -> Section:
    acts = _actions(list(rep.insights), 15, kpis)
    if not acts:
        return _skipped("plan", "Recommendations and action plan", "No finding called for an action.")
    rows = tuple((str(k), a.action, a.owner, a.timeframe, a.measure, SEV_WORD[a.severity]) for k, a in enumerate(acts, 1))
    tbl = Table("Action plan", ("#", "Action", "Suggested owner", "When", "Measure of success", "Priority"), rows,
                "Owners are suggested roles and timeframes follow the priority of each finding; adjust both to your organisation. "
                "Measures with a figure use this app's reference levels.",
                tones=tuple(SEV_TONE[a.severity] for a in acts), tone_col=5, sub=tuple(a.why or a.finding for a in acts))
    n30 = sum(a.timeframe == _WHEN["high"] for a in acts)
    n90 = sum(a.timeframe == _WHEN["medium"] for a in acts)
    later = len(acts) - n30 - n90
    lead = (f"{_plural(len(acts), 'action')} follow from the findings, ordered by priority: {n30} to start within 30 days, {n90} within "
            f"90 days and {later} for the next review. Each has a suggested owner and a measure that shows when it is done.")
    callouts = ((str(n30), "within 30 days", "bad" if n30 else "neutral"), (str(n90), "within 90 days", "warn" if n90 else "neutral"),
                (str(later), "at the next review", "neutral"))
    return Section("plan", 0, "Recommendations and action plan", lead, tables=(tbl,), callouts=callouts,
                   headline=f"{_plural(n30, 'action')} to start within 30 days" if n30 else f"{_plural(len(acts), 'action')} to plan in",
                   desc=f"{_plural(len(acts), 'action')} with suggested owners, timeframes and measures of success")


def _conclusion_section(rep, kpis, ex: Extras, domain_word: str) -> Section:
    ins = list(rep.insights) if rep is not None else []
    stat = [(r, *kpi_status(r)) for r in (kpis.ok() if kpis is not None else [])]
    bad = [r for r, t, _ in stat if t == "bad"]
    warn = [r for r, t, _ in stat if t == "warn"]
    good = [(r, txt) for r, t, txt in stat if t == "good"]
    high = [i for i in ins if i.severity == "high" and plan(i).kind in ("risk", "data")]
    if bad or any(plan(i).kind == "risk" for i in high):
        verdict = "under pressure"
    elif warn or any(i.severity == "medium" for i in ins):
        verdict = "mixed"
    else:
        verdict = "sound"
    paras = []
    p1 = f"Overall, {domain_word} performance in this data is {verdict}."
    if bad or warn:
        p1 += " " + _sentence(_and([f"{clean(r.name) if k == 0 else _lc(clean(r.name))} ({format_value(r.value, r.unit)})"
                                    for k, r in enumerate((bad + warn)[:3])])
                              + (" is" if len(bad + warn) == 1 else " are") + " outside the reference level")
    risks = [i for i in ins if plan(i).kind == "risk"]
    if risks:
        p1 += f" The most important issue is that {_lc(clean(risks[0].title))}."
    paras.append(p1)
    if good:
        paras.append("Strengths: " + _and([f"{_lc(clean(r.name))} at {format_value(r.value, r.unit)} ({txt.lower()})" for r, txt in good[:3]]) + ".")
    comp = ex.comparison
    if comp is not None:
        moves = []
        for key, name, unit, a, b in comp.rows[:6]:
            c, cu = change(unit, a, b)
            if c is not None and abs(c) >= (1 if cu == "pts" else 5):
                moves.append(f"{_lc(clean(name))} {'rose' if c > 0 else 'fell'} {abs(c):.1f}{' points' if cu == 'pts' else '%'}")
        if moves:
            paras.append(f"Compared with the previous {comp.window}, " + _and(moves[:3]) + ".")
    acts = _actions(ins, 3, kpis)
    steps = tuple(f"{a.action}. Owner: {a.owner.lower()}, {a.timeframe.lower()}. "
                  + (f"Success measure: {_lc(a.measure)}." if not a.measure.startswith("Target to be set") else "Success measure: to be agreed by the owner.")
                  for a in acts)
    closing = ("Re-run this report each month with fresh data to track the measures of success in the action plan. Reference levels and "
               "analysis settings can be adjusted in the app, and the methodology in the appendix explains how each figure is produced.")
    headline = f"First priority: {acts[0].action}" if acts else f"Performance is {verdict}"
    return Section("conclusion", 0, "Conclusion and next steps", paras[0], paragraphs=tuple(paras[1:]) + (closing,), steps=steps,
                   headline=headline, desc=f"Overall assessment ({verdict}) and the first {len(steps) or 'few'} steps to take")


def _forecast_section(fc) -> Section:
    target = clean(fc.value_col) if fc.value_col else "number of rows"
    chosen = next((s for s in fc.scores if s.key == fc.chosen), None)
    label = chosen.label if chosen else clean(fc.chosen)
    base_label = clean(next((s.label for s in fc.scores if s.key == fc.baseline), fc.baseline))
    h = len(fc.forecast)
    unit = _FREQ_WORD.get(str(fc.freq).upper(), "period")
    lead, headline, callouts = "", f"Forecast of {target}", []
    imp = fc.improvement_vs_baseline_pct
    if h and len(fc.history) >= h:
        past = float(fc.history["value"].tail(h).mean())
        fut = float(fc.forecast["forecast"].mean())
        if past:
            chg = (fut - past) / abs(past) * 100
            lead = (f"{label} projects an average of {_num(fut)} per {unit} over the next {_plural(h, unit)}, "
                    f"{abs(chg):.1f}% {'above' if chg >= 0 else 'below'} the average of the last {h} observed {unit}(s) ({_num(past)}).")
            verb = "hold at about" if abs(chg) < 2 else "rise to about" if chg > 0 else "fall to about"
            headline = f"{target[:1].upper() + target[1:]} is expected to {verb} {_num(fut)} per {unit}"
            callouts = [(_num(fut), f"average per {unit}, next {_plural(h, unit)}", "neutral"),
                        (f"{'▲' if chg > 0 else '▼' if chg < 0 else '■'} {abs(chg):.1f}%", f"vs last {h} {unit}s", "neutral")]
    if fc.chosen == fc.baseline or imp is None or imp <= 0:
        lead += " No model beat the simple baseline in validation, so treat this as a rough guide."
        callouts.append(("Low", "confidence: no model beat the baseline", "warn"))
    else:
        lead += f" In validation it was {imp:.0f}% more accurate than the simple baseline ({base_label})."
        callouts.append((f"{imp:.0f}%", "more accurate than baseline", "good"))
    facts = [("Measure", f"{target} ({fc.agg} per {unit})"), ("Horizon", _plural(fc.horizon, unit)), ("Model", label),
             ("Accuracy vs baseline", _pct(imp) + " better" if imp is not None and imp > 0 else "not better than the baseline"),
             ("Range shown", f"{fc.interval_level:.0%} prediction interval"),
             ("Validation", f"{fc.folds} rolling-origin fold(s), chronological")]
    fr = tuple((_period(r.period, fc.freq), _num(r.forecast), _num(r.lower), _num(r.upper)) for r in fc.forecast.head(MAX_ROWS).itertuples())
    score_rows = tuple((s.label + (" (selected)" if s.key == fc.chosen else ""), _num(s.mae), _num(s.mase), _num(s.mape) if s.mape is not None else "n/a")
                       for s in sorted(fc.scores, key=lambda x: (x.mae is None, x.mae or 0.0)) if s.status == "ok")
    tables = [Table("Forecast", ("Period", "Forecast", "Low", "High"), fr, f"First {len(fr)} of {h} {unit}(s)." if h > len(fr) else "", numeric_from=1),
              Table("Models compared in validation (lower error is better)", ("Model", "Mean abs. error", "Scaled error", "% error"),
                    score_rows, "Scaled error below 1 beats repeating the last value.", numeric_from=1)]
    chart_list = []
    png = charts.forecast_chart(fc.history, fc.forecast, target)
    if png:
        hist = fc.history["value"]
        note = (f"History from {_period(fc.history['period'].iloc[0], fc.freq)} to {_period(fc.history['period'].iloc[-1], fc.freq)}: "
                f"peak {_num(hist.max())}, low {_num(hist.min())}. The shaded band is the {fc.interval_level:.0%} likely range of the forecast.")
        chart_list.append(Chart(f"{target}: history and forecast", png, note))
    notes = tuple(clean(t, 220) for t in (*fc.warnings, *fc.limitations))
    return Section("forecast", 0, "Outlook", lead, facts=tuple(facts), tables=tuple(tables), charts=tuple(chart_list), notes=notes,
                   headline=headline, callouts=tuple(callouts),
                   desc=f"{_plural(h, unit)} forecast of {target}; " + ("no model beat the baseline" if not (imp and imp > 0) else f"{imp:.0f}% better than baseline"))


def _flag_total(summary) -> int:
    """Values flagged by the most sensitive method, per column, summed. Summing across methods would count a value twice."""
    return int(summary.groupby("feature")["flagged"].max().sum()) if len(summary) else 0


def _anomaly_section(an) -> Section:
    s = an.summary
    flagged = _flag_total(s)
    ok = s[s["status"] == "ok"] if "status" in s.columns else s
    ncols = s['feature'].nunique() if len(s) else 0
    lead = (f"{flagged:,} value(s) across {ncols} column(s) stand out from the typical range. "
            "An unusual value is not necessarily wrong; nothing was removed from the data.")
    tables, headline, callouts = [], f"{flagged:,} values stand out from the typical range", []
    if len(ok):
        per = ok.groupby("feature").agg(examined=("examined", "max"), flagged=("flagged", "max"), share=("share", "max"))
        per = per.sort_values("share", ascending=False)
        tables.append(Table("Unusual values by column", ("Column", "Values checked", "Unusual", "Share"),
                            tuple((clean(f, 32), f"{int(r.examined):,}", f"{int(r.flagged):,}", _pct(float(r.share) * 100)) for f, r in per.iterrows()),
                            "Highest count across the screening methods, so a value is never counted twice.", numeric_from=1))
        if len(per) and per["flagged"].max() > 0:
            topc = per["flagged"].idxmax()
            headline = f"{flagged:,} unusual values, most of them in {clean(topc)}"
            callouts = [(f"{flagged:,}", "unusual values", "warn" if flagged else "neutral"), (str(ncols), "columns checked", "neutral"),
                        (_pct(float(per.loc[topc, 'share']) * 100), f"of {clean(topc, 30)} is unusual", "neutral")]
    f = an.findings.head(MAX_ROWS)
    if len(f):
        frows = tuple((clean(r.row, 16), clean(r.feature, 28), _num(r.value), clean(r.direction), _num(r.score)) for r in f.itertuples())
        tables.append(Table("Most unusual values", ("Row", "Column", "Value", "Direction", "Score"), frows,
                            f"Top {len(frows)} of {len(an.findings):,} listed, by score.", right=(2, 4)))
    chart_list = []
    if len(s) and flagged:
        per2 = s.groupby("feature")["flagged"].max().sort_values(ascending=False).head(8)
        png = charts.hbar([clean(x) for x in per2.index], list(per2.values), fmt=lambda v: f"{v:,.0f}")
        if png:
            note = (f"{clean(per2.index[0])} accounts for {per2.iloc[0]:,.0f} of the {flagged:,} unusual values "
                    f"({100 * per2.iloc[0] / flagged:.0f}%).")
            chart_list.append(Chart("Unusual values by column", png, note))
    return Section("anomalies", 0, "Unusual values", lead, tables=tuple(tables), charts=tuple(chart_list),
                   notes=tuple(clean(t, 220) for t in (*an.warnings, an.note)), headline=headline, callouts=tuple(callouts),
                   desc=f"{flagged:,} unusual values across {ncols} columns, and where they are")


def _segment_section(sg) -> Section:
    sizes = sg.profile["entities"]
    feats = list(sg.features)
    shown = feats[:4]
    names = getattr(sg, "names", {}) or {}
    entity = clean(sg.assignments.columns[0]) if len(sg.assignments.columns) else "entity"
    strength = "clear" if sg.silhouette >= 0.5 else "moderate" if sg.silhouette >= 0.25 else "weak"
    lead = (f"{int(sizes.sum()):,} {entity} values fall into {sg.k} segments. The split is {strength} (fit {sg.silhouette:.2f} of 1)"
            + (", so treat the segments as a convenient grouping rather than distinct types." if strength == "weak" else "."))
    header = ("Segment", "Count", "Share") + tuple(clean(sg.labels.get(f, f), 22) for f in shown)

    def cell(v, f):
        u = sg.units.get(f, "number")
        return _pct(v) if u == "percent" else _num(v)

    def seg_name(i, n=30):
        return f"{int(i)}. {clean(names.get(i, ''), n)}".rstrip(". ").rstrip()
    rows = tuple((seg_name(i), f"{int(r['entities']):,}", _pct(float(r["share"]) * 100), *[cell(r[f], f) for f in shown])
                 for i, r in sg.profile.iterrows())
    tables = [Table("Segment profiles (average per entity)", header, rows,
                    f"{len(feats) - len(shown)} further measure(s) are shown on the Segmentation page." if len(feats) > len(shown) else "", numeric_from=1)]
    png = charts.hbar([seg_name(i, 24) for i in sizes.index], [float(v) for v in sizes.values], fmt=lambda v: f"{v:,.0f}")
    big = sizes.idxmax()
    share = 100 * float(sizes.max()) / float(sizes.sum() or 1)
    big_name = clean(names.get(big, ""), 30)
    note = f"Segment {int(big)}" + (f" ({big_name})" if big_name else "") + " is the largest, with {int(sizes.max()):,} {entity} values ({share:.0f}%)."
    traits = tuple(f"Segment {int(k)}: {clean(v, 260)}" for k, v in sorted(sg.traits.items()) if v)
    return Section("segments", 0, "Segments", lead, tables=tuple(tables), charts=((Chart("Size of each segment", png, note),) if png else ()),
                   notes=traits + tuple(clean(t, 220) for t in (*sg.warnings, sg.note)),
                   headline=f"{entity[:1].upper() + entity[1:]} values fall into {sg.k} groups; the largest holds {share:.0f}%",
                   callouts=((str(sg.k), "segments", "neutral"), (f"{sg.silhouette:.2f}", f"fit of 1 ({strength})", "neutral"),
                             (f"{share:.0f}%", "in the largest segment", "neutral")),
                   desc=f"{sg.k} groups of {entity} with their profiles; the split is {strength}")


def _driver_section(dr) -> Section:
    from insights.insight_rules import _driver_group_line                  # the same sentence the Insights page shows
    imp = dr.importance[dr.importance["share"] > 0].head(8)
    target = clean(dr.target)
    if getattr(dr, "near_perfect", False):
        verdict = ("The columns predict it almost perfectly, which usually means one of them is part of how it is recorded; "
                   "check the top columns before acting on them.")
    elif dr.reliable:
        verdict = "The model predicts held-out rows clearly better than a simple guess, so the ranking is a useful guide."
    else:
        verdict = "The model does not clearly beat a simple guess, so treat the ranking as weak."
    lead = f"Which columns go with {target}. {verdict}"
    headline = f"No column clearly explains {target}"
    if len(imp):
        top = str(imp.iloc[0]["feature"])
        line, _ = _driver_group_line(dr, top)
        lead += f" The strongest link is {clean(top)} ({float(imp.iloc[0]['share']):.0%} of the explanatory power)" + (f": {clean(line)}." if line else ".")
        if dr.reliable:
            headline = f"{clean(top)} is the strongest factor behind {target}"
    lead += " These are associations, not proven causes."
    rows = []
    for r in imp.itertuples():
        line, _ = _driver_group_line(dr, str(r.feature))
        d = str(r.direction)
        if not line and d.startswith("higher values go with"):
            line = f"higher {r.feature} goes with {'higher' if 'higher target' in d else 'lower'} {dr.target}"
        line = clean(line)
        rows.append((clean(r.feature, 34), _pct(float(r.share) * 100), line[:1].upper() + line[1:] if line else "–"))
    tables = [Table("Columns most associated with " + target, ("Column", "Share", "What it shows"), tuple(rows), right=(1,))]
    m = dr.metrics.iloc[0]
    facts = ((("Model check", f"{clean(m['metric'])}: {_num(m['model'], 2)} (simple guess {_num(m['baseline'], 2)})"),
              ("Tested on", f"{dr.n_test:,} held-out rows ({clean(dr.split, 120)})")))
    png = charts.hbar([clean(f) for f in imp["feature"]], [float(s) * 100 for s in imp["share"]], fmt=lambda v: f"{v:.0f}", unit="%")
    chart_list = ()
    if png and len(imp):
        top2 = float(imp["share"].head(2).sum())
        note = (f"{clean(imp.iloc[0]['feature'])} alone accounts for {float(imp.iloc[0]['share']):.0%} of the explanatory power"
                + (f"; together with {clean(imp.iloc[1]['feature'])}, {top2:.0%}." if len(imp) > 1 else "."))
        chart_list = (Chart("Share of explanatory power, top columns", png, note),)
    excl = (f"{len(dr.excluded)} column(s) were left out because they are identifiers, near-copies of {target} or only known "
            "afterwards; see the Drivers page for the list.",) if dr.excluded else ()
    callouts = ()
    if len(imp):
        callouts = ((f"{float(imp.iloc[0]['share']):.0%}", f"explained by {clean(imp.iloc[0]['feature'], 26)}", "neutral"),
                    (_num(m["model"], 2), f"{clean(m['metric'])} (guess {_num(m['baseline'], 2)})", "good" if dr.reliable else "warn"),
                    (f"{dr.n_test:,}", "rows held out for testing", "neutral"))
    return Section("drivers", 0, f"What drives {target}", lead, facts=facts, tables=tuple(tables), charts=chart_list,
                   notes=excl + tuple(clean(t, 220) for t in (*dr.warnings, dr.note)), headline=headline, callouts=callouts,
                   desc=f"Columns most associated with {target}" + (f"; {clean(str(imp.iloc[0]['feature']))} leads" if len(imp) else ""))


def _cleaning_section(cleaning) -> Section:
    if cleaning is None:
        return _skipped("cleaning", "Transformation log", "No cleaning was applied; the analysis used the original data.", "appendix")
    lead = f"{cleaning.applied_count} of {len(cleaning.log)} step(s) were applied. Rows: {cleaning.rows_before:,} before, {cleaning.rows_after:,} after."
    rows = tuple((str(e.step), clean(e.op), clean(e.column or "–", 28, raw=True), e.status, f"{e.cells_changed:,}", clean(e.detail, 130))
                 for e in cleaning.log[:20])
    note = f"{len(cleaning.log) - len(rows)} further step(s) are in the transformation log CSV download." if len(cleaning.log) > len(rows) else ""
    return Section("cleaning", 0, "Transformation log", lead,
                   tables=(Table("Cleaning steps", ("Step", "Operation", "Column", "Status", "Cells", "Detail"), rows, note, right=(4,)),),
                   part="appendix", desc=f"{cleaning.applied_count} cleaning step(s) applied before the analysis")


GLOSSARY = (
    ("On-time delivery rate", ("on-time",), "Share of deliveries that arrived by the promised date."),
    ("OTIF", ("otif", "in-full"), "On time, in full: share of orders delivered both by the promised date and in the full quantity."),
    ("Fill rate", ("fill rate",), "Share of the ordered quantity that was actually delivered."),
    ("Lead time / shipping time", ("lead time", "shipping time"), "Days between an order (or shipment) and its delivery."),
    ("Average delay", ("delay",), "Average days by which actual delivery exceeded the promised time."),
    ("Cancelled shipments", ("cancel",), "Share of orders recorded as cancelled; they have no delivery outcome."),
    ("Profit margin", ("margin",), "Profit as a share of revenue."),
    ("Average order value", ("average order value",), "Revenue divided by the number of distinct orders."),
    ("Concentration", ("top 10%", "concentration", "share of revenue"), "How much of the total comes from the largest few customers, products or suppliers."),
    ("Repeat customer rate", ("repeat",), "Share of customers with more than one order."),
    ("Reference level", ("reference",), "A threshold set in this app (adjustable in settings) used to flag a metric; not an industry standard."),
    ("Forecast baseline", ("baseline",), "A deliberately simple forecast, such as repeating the last value, that any useful model must beat."),
    ("Prediction interval", ("interval", "likely range"), "The range in which future values are expected to fall with the stated probability."),
    ("Mean absolute error (MAE)", ("mean abs", "mae"), "Average size of forecast errors, in the units of the measure."),
    ("Scaled error (MASE)", ("scaled error",), "Forecast error divided by the error of repeating the last value; below 1 means better than that."),
    ("Unusual value", ("unusual",), "A value far outside the typical range of its column, found with robust statistics; not necessarily an error."),
    ("Segment", ("segment",), "A group of similar entities (customers, products, suppliers) found by clustering their averages."),
    ("Explanatory power", ("explanatory power",), "How much a column helps a model predict the target; it shows association, not cause."),
    ("ROC AUC", ("roc auc", "auc"), "Model quality for yes/no targets: 0.5 is a coin toss, 1.0 is perfect."),
    ("Completeness", ("completeness", "cells filled"), "Share of all cells in the data that are filled in."),
    ("Stockout rate", ("stockout",), "Share of stock records with nothing on hand."),
    ("Days of cover", ("days of cover",), "How many days current stock lasts at the recent rate of sales."),
)


def _glossary_section(text: str) -> Section:
    t = text.lower()
    rows = tuple((term, meaning) for term, keys, meaning in GLOSSARY if any(k in t for k in keys))
    if not rows:
        return _skipped("glossary", "Glossary", "No specialist terms were used.", "appendix")
    return Section("glossary", 0, "Glossary", "Plain-language meaning of the terms used in this report.",
                   tables=(Table("Terms", ("Term", "Meaning"), rows),), part="appendix", desc=f"{len(rows)} terms explained in plain language")


METHOD = (
    "All numbers come from the uploaded data through deterministic calculations. No language model or external service was used, and no value in this report is typed in by hand.",
    "Findings are produced by fixed rules with adjustable thresholds, and each recommended action comes from a fixed template tied to its finding. Owners and timeframes in the action plan are suggestions based on the type and priority of each finding. The thresholds are reference levels of this application, not industry standards.",
    "Trends recompute each metric on every complete month with the same formulas as the scorecard; period comparisons recompute them on each date window.",
    "Forecasts are chosen by chronological, rolling-origin validation against a baseline. A forecast is a projection of past patterns, not a guarantee.",
    "Unusual values are found with robust statistics (median and MAD, IQR, z-score where valid). An unusual value may be a genuine extreme; nothing is deleted.",
    "Segments and driver rankings describe patterns and associations. They do not prove causes, and segments are one reasonable grouping, not the only one.",
    "Column roles are detected from names and values and can be wrong; roles you set manually are marked as such. Column names are shown with underscores replaced by spaces; the appendix lists them exactly as in the data.",
)


# ---- executive summary --------------------------------------------------------------------------
def _tiles(kpis, profile, quality, analysis_rows: int, comp) -> tuple[Tile, ...]:
    if kpis is not None and kpis.ok():
        out = []
        for r in prioritise(kpis, MAX_TILES).headline:
            tone, text = kpi_status(r)
            note = text or ("" if not r.period or r.period == "All rows" else clean(r.period, 40))
            delta, dtone = "", "neutral"
            row = comp.get(r.key) if comp is not None else None
            if row is not None:
                txt, c = _fmt_change(row[2], row[3], row[4])
                if c is not None:
                    delta, dtone = f"{txt} vs prior {comp.window}", _change_tone(r.key, c, row[2])
            out.append(Tile(clean(r.name, 40), format_value(r.value, r.unit), note, tone if text else "neutral", delta, dtone))
        if out:
            return tuple(out)
    return (Tile("Rows analysed", f"{analysis_rows:,}"), Tile("Columns", f"{profile.columns:,}"),
            Tile("Completeness", _pct(quality.completeness_pct) if quality is not None else "–"))


def _summary_text(filename, analysis_rows, profile, period, kpis, rep, comp, n_actions, plan_label) -> tuple[str, tuple[str, ...]]:
    groups = []
    for r in (kpis.ok() if kpis is not None else []):
        w = _GROUP_WORDS.get(r.group)
        if w and w not in groups:
            groups.append(w)
    p1 = (f"This review covers {analysis_rows:,} records from {clean(filename)}" + (f", spanning {period}" if period else "")
          + f", with {profile.columns} fields per record.")
    if groups:
        p1 += f" It assesses {_and(groups[:4])}, and sets out where action is needed."
    paras = []
    if kpis is not None:
        bits = []
        rev, margin = kpis.get("total_revenue"), kpis.get("profit_margin")
        if rev is not None and rev.status == "ok" and rev.value is not None:
            s = f"Revenue totals {format_value(rev.value, rev.unit)}"
            if margin is not None and margin.status == "ok" and margin.value is not None:
                s += f" at a {format_value(margin.value, margin.unit)} profit margin"
            bits.append(s + ".")
        flagged = [(r, *kpi_status(r)) for r in kpis.ok()]
        flagged = [(r, t) for r, tone, t in flagged if tone in ("bad", "warn")]
        if flagged:
            items = [f"{clean(r.name) if k == 0 else _lc(clean(r.name))} is {format_value(r.value, r.unit)} ({t.lower()})"
                     for k, (r, t) in enumerate(flagged[:2])]
            bits.append(_and(items) + ".")
        if comp is not None:
            moves = []
            for key, name, unit, a, b in comp.rows[:5]:
                c, cu = change(unit, a, b)
                if c is not None and abs(c) >= (0.5 if cu == "pts" else 2):
                    moves.append(f"{_lc(clean(name))} {'rose' if c > 0 else 'fell'} {abs(c):.1f}{' points' if cu == 'pts' else '%'}")
            if moves:
                bits.append(f"Over the latest {comp.window} ({comp.recent_label}), compared with the {comp.window} before, " + _and(moves[:3]) + ".")
        if bits:
            paras.append(" ".join(bits))
    ins = list(rep.insights) if rep is not None else []
    if ins:
        risks = [i for i in ins if plan(i).kind == "risk"]
        p3 = (f"The analysis raised {_plural(len(ins), 'finding')}, {len(risks)} of which need attention." if risks else
              f"The analysis raised {_plural(len(ins), 'finding')}; none of them is a risk that needs urgent attention.")
        if risks:
            p3 += f" The most urgent: {_lc(clean(risks[0].title))}."
        if n_actions:
            p3 += (f" The priority actions are listed below; the full plan of {_plural(n_actions, 'action')} with suggested owners and "
                   f"timeframes is in section {plan_label}." if plan_label else "")
        paras.append(p3)
    elif rep is not None:
        paras.append("No finding crossed its threshold: no risks, unusual patterns or data problems were found.")
    return p1, tuple(paras)


def _takeaways(rep) -> tuple[tuple[str, str], ...]:
    if rep is None:
        return ()
    ins = [i for i in rep.insights if i.severity != "info"] or list(rep.insights)
    return tuple((plan(i).kind, _cap(clean(i.title, 160))) for i in ins[:5])


# ---- entry point --------------------------------------------------------------------------------
def build_report(*, filename: str, is_sample: bool, fingerprint: str, profile, quality, analysis_rows: int,
                 kpis=None, insight_report=None, forecast=None, anomalies=None, segmentation=None, drivers=None,
                 cleaning=None, semantic_model=None, unavailable: dict[str, str] | None = None,
                 title: str = "", prepared_by: str = "", generated_at: datetime | None = None,
                 user_settings: tuple[str, ...] = (), df=None, organisation: str = "", logo: bytes | None = None,
                 accent: str = theme.ACCENT, classification: str = "Internal", config=DEFAULT_ANALYTICS) -> ReportData:
    """Assemble a report from computed results. Any result may be None; its section is then listed as not included.
    `df` (the analysed rows) enables the trend, period-comparison and breakdown sections."""
    pal = theme.palette(accent)
    tok_p = theme.use(pal)
    tok_n = _NAMES.set(_name_map(list(df.columns) if df is not None else [c.name for c in profile.columns_profile]))
    try:
        return _build(filename, is_sample, fingerprint, profile, quality, analysis_rows, kpis, insight_report, forecast, anomalies,
                      segmentation, drivers, cleaning, semantic_model, unavailable or {}, title, prepared_by, generated_at, user_settings,
                      df, organisation, logo, pal.accent, classification, config)
    finally:
        _NAMES.reset(tok_n)
        theme.reset(tok_p)


def _build(filename, is_sample, fingerprint, profile, quality, analysis_rows, kpis, insight_report, forecast, anomalies, segmentation,
           drivers, cleaning, semantic_model, unavailable, title, prepared_by, generated_at, user_settings, df, organisation, logo,
           accent, classification, config) -> ReportData:
    when = generated_at or datetime.now()
    domain = (kpis.domain if kpis is not None and kpis.domain else None) or (semantic_model.domain.domain if semantic_model is not None else "general")
    period = ""
    try:
        period = _data_period(profile, semantic_model)
    except Exception:
        log.exception("Data period could not be read")
    ex = Extras()
    if df is not None and kpis is not None:
        try:
            keys = tuple(r.key for r in prioritise(kpis, MAX_TILES).headline)
            ex = gather(df, semantic_model, config, kpis, keys)
        except Exception:
            log.exception("Extra report views failed")

    def guard(key, label, result, fn, part="main"):
        if result is None:
            return _skipped(key, label, unavailable.get(key, "This analysis was not run for this report."), part)
        try:
            return fn(result)
        except Exception:                                  # one section must never stop the report
            log.exception("Report section %s failed", key)
            return _skipped(key, label, "The section could not be built from this result.", part)

    domain_word = {"supply_chain": "supply-chain", "retail": "sales", "sales_crm": "sales", "finance": "financial",
                   "inventory": "inventory", "operations": "operational"}.get(domain, "business")
    rep = insight_report
    parts = [
        guard("insights", "Key findings", rep, _findings_section) if rep is not None
        else _skipped("insights", "Key findings", unavailable.get("insights", "Insights were not generated.")),
        guard("kpis", "Performance scorecard", kpis, _kpi_section),
        guard("trends", "Performance over time", ex if (ex.histories or ex.comparison) else None, lambda e: _trend_section(kpis, e)),
        guard("breakdown", "Breakdown in detail", ex if ex.views else None, _breakdown_section),
        guard("drivers", "Drivers", drivers, _driver_section),
        guard("segments", "Segments", segmentation, _segment_section),
        guard("forecast", "Outlook", forecast, _forecast_section),
        guard("anomalies", "Unusual values", anomalies, _anomaly_section),
        guard("plan", "Recommendations and action plan", rep, lambda r: _plan_section(r, kpis)),
        guard("conclusion", "Conclusion and next steps", (rep,) if (rep is not None or kpis is not None) else None,
              lambda t: _conclusion_section(t[0], kpis, ex, domain_word)),
        guard("dataset", "Data overview", profile, lambda p: _dataset_section(p, semantic_model, cleaning, analysis_rows, domain, period), "appendix"),
        guard("quality", "Data quality", quality, lambda q: _quality_section(profile, q), "appendix"),
        guard("definitions", "Metric definitions", kpis, _definitions_section, "appendix"),
        _cleaning_section(cleaning),
    ]
    if df is None or kpis is None:                         # without the rows these sections are not attempted; say nothing
        for k in ("trends", "breakdown"):
            i = next(n for n, s in enumerate(parts) if s.key == k)
            parts[i] = replace(parts[i], reason="Needs the analysed rows, which were not passed to the report.")
    corpus = " ".join(" ".join((s.lead, s.headline, *s.notes, *(c.body for c in s.cards), *(t.title for t in s.tables),
                                *(" ".join(h) for t in s.tables for h in [t.header]), *(c.title + c.note for c in s.charts)))
                      for s in parts if s.status == "ok")
    glossary = _glossary_section(corpus)
    parts.insert(len(parts) - 1, glossary)
    main_n, app_n, out = 0, 0, []
    for n, s in enumerate(parts, 2):
        label = ""
        if s.status == "ok":
            if s.part == "main":
                main_n += 1
                label = str(main_n)
            else:
                label = chr(ord("A") + app_n)
                app_n += 1
        out.append(replace(s, number=n, label=label))
    method_label = chr(ord("A") + app_n)
    quiet = ("cleaning", "definitions", "glossary", "trends", "breakdown", "plan", "conclusion")
    skipped = tuple((s.title, s.reason) for s in out if s.status == "skipped" and s.key not in quiet)
    included = tuple(s.title for s in out if s.status == "ok")
    ok = {s.key for s in out if s.status == "ok"}
    lab = {s.key: s.label for s in out}

    rep_ok = rep if "insights" in ok else None
    kpis_ok = kpis if "kpis" in ok else None
    tiles = _tiles(kpis_ok, profile, quality, analysis_rows, ex.comparison)
    takeaways = _takeaways(rep_ok)
    actions = tuple(_actions(list(rep_ok.insights), MAX_ACTIONS, kpis_ok)) if rep_ok is not None else ()
    n_all = len(_actions(list(rep_ok.insights), 15, kpis_ok)) if rep_ok is not None else 0
    lede, paras = _summary_text(filename, analysis_rows, profile, period, kpis_ok, rep_ok, ex.comparison, n_all, lab.get("plan", ""))
    summary = tuple(f"{KINDS[k]}: {t}" for k, t in takeaways)
    if skipped:
        summary += ("Not included: " + "; ".join(f"{t.lower()} ({clean(r, 100).rstrip('.')})" for t, r in skipped[:4]) + ".",)
    qual_label = "Appendix " + lab.get("quality", "")
    quality_line = (f"Data quality: rated {quality.status}, {_pct(quality.completeness_pct)} of cells filled, "
                    f"{len(quality.by_severity('high'))} high-severity issue(s). Details in {qual_label.strip()}.") if quality is not None else ""
    headline = tuple((t.label, t.value) for t in tiles)
    rid = hashlib.sha256(f"{fingerprint}|{when.isoformat()}".encode()).hexdigest()[:8].upper()
    scope = (("Not included in this report: " + "; ".join(f"{t} ({clean(r, 120).rstrip('.')})" for t, r in skipped) + ".",)
             if skipped else ())
    meth = METHOD + (f"Rows and columns in this report come from the {'cleaned' if cleaning is not None else 'original'} dataset.",
                     ("Settings chosen on the " + ", ".join(user_settings) + " page(s) were used for those analyses; the others ran with default settings.")
                     if user_settings else "All analyses in this report ran with default settings.") + scope
    dom_label = clean(domain).replace("_", " ").capitalize() if domain else ""
    t = clean(title, 90)
    auto = not t or t == DEFAULT_TITLE
    final_title = AUTO_TITLES.get(domain, AUTO_TITLES["general"]) if auto else t
    subtitle = f"Based on {analysis_rows:,} records from {clean(filename, 100, raw=True)}"
    key_message = takeaways[0][1] if takeaways else (
        "Every metric with a reference level meets it" if kpis_ok is not None else f"{analysis_rows:,} records reviewed")
    cover_png = None
    if ex.cover_series:
        try:
            cover_png = charts.cover_graphic(list(ex.cover_series))
        except Exception:
            log.exception("Cover graphic failed")
    eyebrow = " · ".join(x for x in (dom_label if dom_label.lower() != "general" else "", period) if x)
    return ReportData(final_title, subtitle, clean(prepared_by, 80), clean(filename, 120, raw=True), bool(is_sample),
                      when.strftime("%d %b %Y, %H:%M"), f"{rid[:4]}-{rid[4:]}", headline, summary, tuple(out), meth, included, skipped,
                      lede, tiles, takeaways, actions, period, dom_label, quality_line, method_label, APP_NAME, _sentence(key_message),
                      paras, _about(profile, quality, cleaning, analysis_rows, domain, period), cover_png, clean(organisation, 80),
                      logo if logo else None, accent, clean(classification, 30) or "Internal", eyebrow)
